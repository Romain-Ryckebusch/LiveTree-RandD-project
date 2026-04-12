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
    RECENT_HA_CSV,
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


def load_historical_data():
    # Combines the Jan-Mar HISTORICAL_CSV with the Mar 22 - Apr 10 RECENT_HA_CSV.
    # The two are non-overlapping and only Ptot_HA is reliable in the second.
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    try:
        recent = pd.read_csv(RECENT_HA_CSV, parse_dates=["Date"])
    except FileNotFoundError:
        return df
    if "Ptot_HA" not in recent.columns:
        return df
    recent = recent[["Date", "Ptot_HA"]].dropna(subset=["Ptot_HA"])
    merged = pd.concat([df, recent], ignore_index=True, sort=False)
    merged = (
        merged.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return merged


def load_weather_data():
    df = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def extract_window(df, target_date, weather_df):
    tz = pytz.timezone(TIMEZONE)

    target_start = tz.localize(pd.Timestamp(target_date))
    target_end = target_start + pd.Timedelta(hours=23, minutes=50)
    history_start = target_start - pd.Timedelta(days=7)

    df = df.copy()
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert(tz)

    mask_hist = (df["Date"] >= history_start) & (df["Date"] < target_start)
    history = df.loc[mask_hist].copy().reset_index(drop=True)

    if len(history) < LOOKBACK_POINTS:
        print(
            f"Warning: only {len(history)} history points "
            f"(expected {LOOKBACK_POINTS}). Proceeding anyway."
        )

    if len(history) > LOOKBACK_POINTS:
        history = history.iloc[-LOOKBACK_POINTS:].reset_index(drop=True)

    target_timestamps = pd.date_range(
        target_start, target_end, freq="10min"
    ).tolist()

    weather_df = weather_df.copy()
    if weather_df["Date"].dt.tz is None:
        weather_df["Date"] = weather_df["Date"].dt.tz_localize("UTC").dt.tz_convert(tz)

    mask_weather = (weather_df["Date"] >= target_start) & (
        weather_df["Date"] <= target_end
    )
    weather_slice = weather_df.loc[mask_weather]

    if len(weather_slice) >= POINTS_PER_DAY:
        weather_temps = weather_slice["AirTemp"].values[:POINTS_PER_DAY]
    else:
        weather_temps = np.full(POINTS_PER_DAY, 15.0)
        if len(weather_slice) > 0:
            weather_temps[: len(weather_slice)] = weather_slice["AirTemp"].values

    mask_actual = (df["Date"] >= target_start) & (df["Date"] <= target_end)
    actual_slice = df.loc[mask_actual]
    actual_target = None
    if len(actual_slice) >= POINTS_PER_DAY:
        actual_target = actual_slice[BUILDING_COLUMN].values[:POINTS_PER_DAY]

    return history, target_timestamps, weather_temps, actual_target


def _make_label(args):
    if args.gap_mode == "blocks":
        return f"blocks_{args.block_length}x{args.n_blocks}"
    return f"random_{args.n_points}"


def run_reference(args):
    print("Reference prediction mode")
    hist_df = load_historical_data()
    weather_df = load_weather_data()

    print(f"Target date: {args.target_date}")
    print(f"Building: {BUILDING_COLUMN}")

    history, target_ts, weather, _actual = extract_window(
        hist_df, args.target_date, weather_df
    )
    print(f"History window: {len(history)} points")

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
    def log(msg):
        if not quiet:
            print(msg)

    if hist_df is None:
        log("Loading data...")
        hist_df = load_historical_data()
    if weather_df is None:
        weather_df = load_weather_data()

    label = getattr(args, "label", None) or _make_label(args)
    log(f"Target date: {args.target_date}")
    log(f"Building: {BUILDING_COLUMN}")

    history, target_ts, weather, _actual = extract_window(
        hist_df, args.target_date, weather_df
    )
    log(f"History window: {len(history)} points")

    history_clean = history[BUILDING_COLUMN].values.copy()

    log("\nBaseline prediction (no gaps)")
    baseline_pred = predict_day(target_ts, weather, history)
    log(f"Baseline predictions: min={baseline_pred.min():.0f}, max={baseline_pred.max():.0f}")

    log(f"\nInjecting gaps (mode={args.gap_mode})")
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

    naive_method = getattr(args, "naive_method", "linear")
    log(f"\nNaive imputation ({naive_method})")
    naive_series, _naive_quality = naive_impute(
        history_gapped[BUILDING_COLUMN], method=naive_method
    )
    history_naive_df = history_gapped.copy()
    history_naive_df[BUILDING_COLUMN] = naive_series.values
    history_naive_values = history_naive_df[BUILDING_COLUMN].values.copy()
    log(f"Remaining NaN after naive imputation: {np.isnan(history_naive_values).sum()}")

    log("\nNaive prediction")
    naive_pred = predict_day(target_ts, weather, history_naive_df)
    log(f"Naive predictions: min={naive_pred.min():.0f}, max={naive_pred.max():.0f}")

    log("\nImputation")
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

    log("\nImputed prediction")
    imputed_pred = predict_day(target_ts, weather, history_imputed)
    log(f"Imputed predictions: min={imputed_pred.min():.0f}, max={imputed_pred.max():.0f}")

    log("\nEvaluation")
    metrics = evaluate(baseline_pred, imputed_pred, naive_pred=naive_pred)
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
    args = parser.parse_args()

    if args.reference_only:
        run_reference(args)
        return

    if args.scenarios:
        with open(args.scenarios) as f:
            scenario_defs = json.load(f)

        print(f"Running {len(scenario_defs)} scenarios from {args.scenarios}\n")
        hist_df = load_historical_data()
        weather_df = load_weather_data()

        all_results = []
        for i, sdef in enumerate(scenario_defs):
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
            print(f"\nScenario {i+1}/{len(scenario_defs)}: {scenario_args.label}")

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

        agg = compute_aggregate(all_results)
        if agg:
            print("\nAggregate statistics across all scenarios")
            for key, stats in agg.items():
                print(f"  {key}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
                      f"min={stats['min']:.2f} max={stats['max']:.2f}")

    else:
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
