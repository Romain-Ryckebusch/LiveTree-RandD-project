"""
Demo pipeline: load CSV -> inject gaps -> impute -> predict -> evaluate.

Single-building (Ptot_HA) demo for validating the imputation module.

Usage:
    python run_demo.py --target-date 2026-01-15
    python run_demo.py --target-date 2026-01-15 --block-length 72 --n-blocks 3
    python run_demo.py --target-date 2026-01-15 --gap-mode random --n-points 100
    python run_demo.py --target-date 2026-01-15 --save-plots --save-summary
    python run_demo.py --target-date 2026-01-15 --scenarios scenarios.json --save-plots --save-summary
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pytz

from config import (
    HISTORICAL_CSV,
    WEATHER_CSV,
    BUILDING_COLUMN,
    LOOKBACK_POINTS,
    POINTS_PER_DAY,
    TIMEZONE,
    OUTPUT_DIR,
)
from gap_injector import inject_block_gaps, inject_random_gaps
from imputer import impute, naive_impute
from predict import predict_day
from evaluate import (
    evaluate,
    print_report,
    ScenarioResult,
    save_summary_table,
    compute_aggregate,
)
from plotting import (
    plot_predictions,
    plot_reference,
    plot_history_gaps,
    plot_metrics_bars,
    plot_scenario_comparison,
)


def load_historical_data(source="csv"):
    """Load consumption data from CSV or Cassandra."""
    if source == "cassandra":
        from cassandra_client import load_historical_data_cassandra
        return load_historical_data_cassandra()
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_weather_data(source="csv"):
    """Load weather data from CSV or Cassandra."""
    if source == "cassandra":
        from cassandra_client import load_weather_data_cassandra
        return load_weather_data_cassandra()
    df = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def extract_window(df, target_date, weather_df):
    """
    Extract the 7-day history window and target-day weather for a given date.

    Returns
    -------
    history_df : DataFrame (up to 1008 rows)
    target_timestamps : list of datetime (144)
    weather_temps : array (144)
    actual_target : array (144) or None
    """
    tz = pytz.timezone(TIMEZONE)

    # Work in UTC for filtering to avoid DST issues, then convert
    # target timestamps to local time for feature engineering.
    target_start_local = tz.localize(pd.Timestamp(target_date))
    target_end_local = target_start_local + pd.Timedelta(hours=23, minutes=50)
    history_start_local = target_start_local - pd.Timedelta(days=7)

    # Convert bounds to UTC for comparison
    target_start_utc = target_start_local.astimezone(pytz.utc)
    target_end_utc = target_end_local.astimezone(pytz.utc)
    history_start_utc = history_start_local.astimezone(pytz.utc)

    # Ensure Date column is UTC-aware
    df = df.copy()
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC")
    else:
        df["Date"] = df["Date"].dt.tz_convert("UTC")

    # Extract 7-day history
    mask_hist = (df["Date"] >= history_start_utc) & (df["Date"] < target_start_utc)
    history = df.loc[mask_hist].copy()

    # Build a complete 10-min grid for the 7-day window.
    # In CSV data, every timestamp has a row (missing values are NaN).
    # In Cassandra, missing timestamps have no row at all.
    # Reindexing against a full grid turns missing timestamps into NaN rows,
    # so the imputer can detect and fill them.
    full_grid = pd.date_range(
        history_start_utc, periods=LOOKBACK_POINTS, freq="10min", tz="UTC"
    )
    history = (
        history
        .set_index("Date")
        .reindex(full_grid)
        .rename_axis("Date")
        .reset_index()
    )

    n_missing = history.iloc[:, 1:].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"Detected {n_missing} missing timestamps in 7-day window (real gaps)")

    # Target day timestamps in local time (for feature engineering in predict.py)
    target_timestamps = pd.date_range(
        target_start_local, target_end_local, freq="10min"
    ).tolist()

    # Weather for target day
    weather_df = weather_df.copy()
    if weather_df["Date"].dt.tz is None:
        weather_df["Date"] = weather_df["Date"].dt.tz_localize("UTC")
    else:
        weather_df["Date"] = weather_df["Date"].dt.tz_convert("UTC")

    mask_weather = (weather_df["Date"] >= target_start_utc) & (
        weather_df["Date"] <= target_end_utc
    )
    weather_slice = weather_df.loc[mask_weather]

    if len(weather_slice) >= POINTS_PER_DAY:
        weather_temps = weather_slice["AirTemp"].values[:POINTS_PER_DAY]
    else:
        weather_temps = np.full(POINTS_PER_DAY, 15.0)
        if len(weather_slice) > 0:
            weather_temps[: len(weather_slice)] = weather_slice["AirTemp"].values

    # Actual target day consumption (for evaluation, if available)
    mask_actual = (df["Date"] >= target_start_utc) & (df["Date"] <= target_end_utc)
    actual_slice = df.loc[mask_actual]
    actual_target = None
    if len(actual_slice) >= POINTS_PER_DAY:
        actual_target = actual_slice[BUILDING_COLUMN].values[:POINTS_PER_DAY]

    return history, target_timestamps, weather_temps, actual_target


def _make_label(args):
    """Build a scenario label from gap parameters."""
    if args.gap_mode == "blocks":
        return f"blocks_{args.block_length}x{args.n_blocks}"
    return f"random_{args.n_points}"


def run_reference(args):
    """
    Reference mode: predict on clean data only (no gaps, no imputation).

    Produces the baseline prediction for the target date and saves it as CSV.
    """
    print("=== Reference prediction mode ===")
    hist_df = load_historical_data(args.source)
    weather_df = load_weather_data(args.source)

    print(f"Target date: {args.target_date}")
    print(f"Building: {BUILDING_COLUMN}")

    history, target_ts, weather, actual = extract_window(
        hist_df, args.target_date, weather_df
    )
    print(f"History window: {len(history)} points")

    # Impute real gaps if any (Cassandra data may have missing timestamps)
    real_gaps = history[BUILDING_COLUMN].isna().sum()
    if real_gaps > 0:
        print(f"Imputing {real_gaps} real gaps before prediction")
        real_imputed, _ = impute(
            history[BUILDING_COLUMN],
            history["Date"],
        )
        history[BUILDING_COLUMN] = real_imputed.values

    baseline_pred = predict_day(target_ts, weather, history)
    print(f"Predictions: min={baseline_pred.min():.0f}, max={baseline_pred.max():.0f}")

    # Save as CSV
    out_dir = os.path.join(args.output_dir, f"reference_{args.target_date}")
    os.makedirs(out_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "timestamp": target_ts,
        "predicted_power_W": baseline_pred,
    })
    csv_path = os.path.join(out_dir, "reference_prediction.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    if args.save_plots:
        plot_reference(target_ts, baseline_pred, out_dir, args.target_date)

    print("\nReference prediction complete.")


def run(args, hist_df=None, weather_df=None, quiet=False):
    """
    Run a single scenario: inject gaps, impute, predict, evaluate.

    Parameters
    ----------
    args : argparse.Namespace or similar object with gap config fields.
    hist_df, weather_df : DataFrames, optional
        Pre-loaded data (avoids reloading in multi-scenario mode).
    quiet : bool
        Suppress console output.

    Returns
    -------
    ScenarioResult with all intermediate data and metrics.
    """
    def log(msg):
        if not quiet:
            print(msg)

    source = getattr(args, "source", "csv")
    if hist_df is None:
        log("Loading data...")
        hist_df = load_historical_data(source)
    if weather_df is None:
        weather_df = load_weather_data(source)

    label = getattr(args, "label", None) or _make_label(args)
    log(f"Target date: {args.target_date}")
    log(f"Building: {BUILDING_COLUMN}")

    history, target_ts, weather, actual = extract_window(
        hist_df, args.target_date, weather_df
    )
    log(f"History window: {len(history)} points")

    # If data has real gaps (from Cassandra), impute them first
    # so we have a complete baseline to predict from.
    real_gaps = history[BUILDING_COLUMN].isna().sum()
    if real_gaps > 0:
        log(f"\n=== Imputing {real_gaps} real gaps before baseline ===")
        real_imputed, _real_quality = impute(
            history[BUILDING_COLUMN],
            history["Date"],
            random_seed=args.seed,
        )
        history[BUILDING_COLUMN] = real_imputed.values

    # Save clean history before gap injection
    history_clean = history[BUILDING_COLUMN].values.copy()

    # --- Baseline prediction (clean data, no gaps) ---
    log("\n=== Baseline prediction (no gaps) ===")
    baseline_pred = predict_day(target_ts, weather, history)
    log(f"Baseline predictions: min={baseline_pred.min():.0f}, max={baseline_pred.max():.0f}")

    # --- Inject gaps ---
    log(f"\n=== Injecting gaps (mode={args.gap_mode}) ===")
    history_gapped = history.copy()
    gap_info = []
    if args.gap_mode == "blocks":
        history_gapped, gaps = inject_block_gaps(
            history_gapped,
            BUILDING_COLUMN,
            block_length=args.block_length,
            n_blocks=args.n_blocks,
            seed=args.seed,
        )
        gap_info = gaps
        log(f"Injected {len(gaps)} block(s) of {args.block_length} points each")
        for start, end in gaps:
            log(f"  Gap: rows {start}-{end}")
    else:
        history_gapped, indices = inject_random_gaps(
            history_gapped,
            BUILDING_COLUMN,
            n_points=args.n_points,
            seed=args.seed,
        )
        gap_info = indices
        log(f"Injected {len(indices)} random NaN points")

    history_gapped_values = history_gapped[BUILDING_COLUMN].values.copy()
    nan_count = np.isnan(history_gapped_values).sum()
    log(f"Total NaN in window: {nan_count}/{len(history_gapped)}")

    # --- Naive imputation ---
    naive_method = getattr(args, "naive_method", "linear")
    log(f"\n=== Naive imputation ({naive_method}) ===")
    naive_series, _naive_quality = naive_impute(
        history_gapped[BUILDING_COLUMN], method=naive_method
    )
    history_naive_df = history_gapped.copy()
    history_naive_df[BUILDING_COLUMN] = naive_series.values
    history_naive_values = history_naive_df[BUILDING_COLUMN].values.copy()
    log(f"Remaining NaN after naive imputation: {np.isnan(history_naive_values).sum()}")

    # --- Naive prediction ---
    log("\n=== Naive prediction ===")
    naive_pred = predict_day(target_ts, weather, history_naive_df)
    log(f"Naive predictions: min={naive_pred.min():.0f}, max={naive_pred.max():.0f}")

    # --- Impute ---
    log("\n=== Imputation ===")
    imputed_series, quality = impute(
        history_gapped[BUILDING_COLUMN],
        history_gapped["Date"],
        random_seed=args.seed,
    )
    history_imputed = history_gapped.copy()
    history_imputed[BUILDING_COLUMN] = imputed_series.values
    history_imputed_values = history_imputed[BUILDING_COLUMN].values.copy()
    quality_values = quality.values.copy()

    remaining_nan = np.isnan(history_imputed_values).sum()
    log(f"Remaining NaN after imputation: {remaining_nan}")
    log("Quality flag distribution:")
    for flag, flag_label in [(0, "real"), (1, "linear"), (2, "contextual"), (3, "donor-day")]:
        count = (quality_values == flag).sum()
        if count > 0:
            log(f"  {flag} ({flag_label}): {count}")

    # --- Prediction on imputed data ---
    log("\n=== Imputed prediction ===")
    imputed_pred = predict_day(target_ts, weather, history_imputed)
    log(f"Imputed predictions: min={imputed_pred.min():.0f}, max={imputed_pred.max():.0f}")

    # --- Evaluate ---
    log("\n=== Evaluation ===")
    metrics = evaluate(baseline_pred, imputed_pred, actual, naive_pred=naive_pred)
    if not quiet:
        print_report(metrics)

    return ScenarioResult(
        scenario_label=label,
        target_date=args.target_date,
        gap_mode=args.gap_mode,
        block_length=getattr(args, "block_length", 0),
        n_blocks=getattr(args, "n_blocks", 0),
        n_points=getattr(args, "n_points", 0),
        seed=args.seed,
        baseline_pred=baseline_pred,
        imputed_pred=imputed_pred,
        actual=actual,
        target_timestamps=target_ts,
        history_dates=history["Date"],
        history_clean=history_clean,
        history_gapped=history_gapped_values,
        history_imputed=history_imputed_values,
        quality_flags=quality_values,
        naive_pred=naive_pred,
        history_naive=history_naive_values,
        gap_info=gap_info,
        metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Single-building demo pipeline")
    parser.add_argument(
        "--target-date",
        default=os.environ.get("TARGET_DATE"),
        required="TARGET_DATE" not in os.environ,
        help="Date to predict (YYYY-MM-DD). Needs 7 days of prior data.",
    )
    parser.add_argument(
        "--gap-mode",
        choices=["blocks", "random"],
        default=os.environ.get("GAP_MODE", "blocks"),
        help="Gap injection mode (default: blocks)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=int(os.environ.get("BLOCK_LENGTH", 36)),
        help="Points per gap block (default: 36 = 6 hours)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=int(os.environ.get("N_BLOCKS", 2)),
        help="Number of gap blocks",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=int(os.environ.get("N_POINTS", 50)),
        help="Random gap points",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("DEMO_OUTPUT_DIR", OUTPUT_DIR),
        help="Output directory for plots and summaries",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save PNG plots",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save summary CSV/JSON",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to a JSON file defining multiple gap scenarios to run in batch",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Run baseline prediction only (no gap injection or imputation).",
    )
    parser.add_argument(
        "--naive-method",
        choices=["linear", "zero"],
        default="linear",
        help="Naive imputation method: 'linear' (interpolation) or 'zero' (zero-padding). Default: linear.",
    )
    parser.add_argument(
        "--source",
        choices=["csv", "cassandra"],
        default=os.environ.get("DATA_SOURCE", "csv"),
        help="Data source: 'csv' (local files) or 'cassandra' (database). Default: csv.",
    )
    args = parser.parse_args()

    if args.reference_only:
        run_reference(args)
        return

    if args.scenarios:
        # --- Multi-scenario mode ---
        with open(args.scenarios) as f:
            scenario_defs = json.load(f)

        print(f"Running {len(scenario_defs)} scenarios from {args.scenarios}\n")
        hist_df = load_historical_data()
        weather_df = load_weather_data()

        all_results = []
        for i, sdef in enumerate(scenario_defs):
            # Merge scenario def with CLI defaults
            scenario_args = argparse.Namespace(
                target_date=args.target_date,
                gap_mode=sdef.get("gap_mode", args.gap_mode),
                block_length=sdef.get("block_length", args.block_length),
                n_blocks=sdef.get("n_blocks", args.n_blocks),
                n_points=sdef.get("n_points", args.n_points),
                seed=sdef.get("seed", args.seed),
                label=sdef.get("label", f"scenario_{i+1}"),
                naive_method=args.naive_method,
            )
            print(f"\n{'='*60}")
            print(f"Scenario {i+1}/{len(scenario_defs)}: {scenario_args.label}")
            print(f"{'='*60}")

            result = run(scenario_args, hist_df=hist_df, weather_df=weather_df)
            all_results.append(result)

            if args.save_plots:
                scenario_dir = os.path.join(
                    args.output_dir,
                    f"{args.target_date}_{scenario_args.label}",
                )
                plot_predictions(result, scenario_dir)
                plot_history_gaps(result, scenario_dir)
                plot_metrics_bars(result, scenario_dir)

        if args.save_summary:
            save_summary_table(all_results, args.output_dir)

        if args.save_plots and len(all_results) > 1:
            plot_scenario_comparison(all_results, args.output_dir)

        # Print aggregate stats
        agg = compute_aggregate(all_results)
        if agg:
            print(f"\n{'='*60}")
            print("Aggregate statistics across all scenarios")
            print(f"{'='*60}")
            for key, stats in agg.items():
                print(f"  {key}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
                      f"min={stats['min']:.2f} max={stats['max']:.2f}")

    else:
        # --- Single-scenario mode ---
        result = run(args)

        if args.save_plots:
            scenario_dir = os.path.join(
                args.output_dir,
                f"{args.target_date}_{_make_label(args)}",
            )
            plot_predictions(result, scenario_dir)
            plot_history_gaps(result, scenario_dir)
            plot_metrics_bars(result, scenario_dir)

        if args.save_summary:
            save_summary_table([result], args.output_dir)


if __name__ == "__main__":
    main()
