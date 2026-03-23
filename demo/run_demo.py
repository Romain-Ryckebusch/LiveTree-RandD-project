"""
Demo pipeline: load CSV -> inject gaps -> impute -> predict -> evaluate.

Single-building (Ptot_HA) demo for validating the imputation module.

Usage:
    python run_demo.py --target-date 2026-01-15
    python run_demo.py --target-date 2026-01-15 --block-length 72 --n-blocks 3
    python run_demo.py --target-date 2026-01-15 --gap-mode random --n-points 100
"""
import argparse
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
)
from gap_injector import inject_block_gaps, inject_random_gaps
from imputer import impute
from predict import predict_day
from evaluate import evaluate, print_report


def load_historical_data():
    """Load and prepare the historical consumption CSV."""
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_weather_data():
    """Load the weather CSV."""
    df = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def extract_window(df, target_date, weather_df):
    """
    Extract the 7-day history window and target-day weather for a given date.

    Parameters
    ----------
    df : DataFrame with Date and BUILDING_COLUMN
    target_date : date
    weather_df : DataFrame with Date and AirTemp

    Returns
    -------
    history_df : DataFrame (1008 rows)
    target_timestamps : list of datetime (144)
    weather_temps : array (144)
    actual_target : array (144) or None
    """
    tz = pytz.timezone(TIMEZONE)

    target_start = tz.localize(pd.Timestamp(target_date))
    target_end = target_start + pd.Timedelta(hours=23, minutes=50)
    history_start = target_start - pd.Timedelta(days=7)

    # Localize CSV dates if needed
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert(tz)

    # Extract 7-day history
    mask_hist = (df["Date"] >= history_start) & (df["Date"] < target_start)
    history = df.loc[mask_hist].copy().reset_index(drop=True)

    if len(history) < LOOKBACK_POINTS:
        print(
            f"Warning: only {len(history)} history points "
            f"(expected {LOOKBACK_POINTS}). Proceeding anyway."
        )

    # Truncate or pad to exactly 1008
    if len(history) > LOOKBACK_POINTS:
        history = history.iloc[-LOOKBACK_POINTS:].reset_index(drop=True)

    # Target day timestamps
    target_timestamps = pd.date_range(
        target_start, target_end, freq="10min"
    ).tolist()

    # Weather for target day
    if weather_df["Date"].dt.tz is None:
        weather_df["Date"] = weather_df["Date"].dt.tz_localize("UTC").dt.tz_convert(tz)

    mask_weather = (weather_df["Date"] >= target_start) & (
        weather_df["Date"] <= target_end
    )
    weather_slice = weather_df.loc[mask_weather]

    if len(weather_slice) >= POINTS_PER_DAY:
        weather_temps = weather_slice["AirTemp"].values[:POINTS_PER_DAY]
    else:
        # Fallback: fill with 15 degrees (same as ConsoFile.py)
        weather_temps = np.full(POINTS_PER_DAY, 15.0)
        if len(weather_slice) > 0:
            weather_temps[: len(weather_slice)] = weather_slice["AirTemp"].values

    # Actual target day consumption (for evaluation, if available)
    mask_actual = (df["Date"] >= target_start) & (df["Date"] <= target_end)
    actual_slice = df.loc[mask_actual]
    actual_target = None
    if len(actual_slice) >= POINTS_PER_DAY:
        actual_target = actual_slice[BUILDING_COLUMN].values[:POINTS_PER_DAY]

    return history, target_timestamps, weather_temps, actual_target


def run(args):
    print(f"Loading data...")
    hist_df = load_historical_data()
    weather_df = load_weather_data()

    print(f"Target date: {args.target_date}")
    print(f"Building: {BUILDING_COLUMN}")

    history, target_ts, weather, actual = extract_window(
        hist_df, args.target_date, weather_df
    )
    print(f"History window: {len(history)} points")

    # --- Baseline prediction (clean data, no gaps) ---
    print("\n=== Baseline prediction (no gaps) ===")
    baseline_pred = predict_day(target_ts, weather, history)
    print(f"Baseline predictions: min={baseline_pred.min():.0f}, max={baseline_pred.max():.0f}")

    # --- Inject gaps ---
    print(f"\n=== Injecting gaps (mode={args.gap_mode}) ===")
    history_gapped = history.copy()
    if args.gap_mode == "blocks":
        history_gapped, gaps = inject_block_gaps(
            history_gapped,
            BUILDING_COLUMN,
            block_length=args.block_length,
            n_blocks=args.n_blocks,
            seed=args.seed,
        )
        print(f"Injected {len(gaps)} block(s) of {args.block_length} points each")
        for start, end in gaps:
            print(f"  Gap: rows {start}-{end}")
    else:
        history_gapped, indices = inject_random_gaps(
            history_gapped,
            BUILDING_COLUMN,
            n_points=args.n_points,
            seed=args.seed,
        )
        print(f"Injected {len(indices)} random NaN points")

    nan_count = history_gapped[BUILDING_COLUMN].isna().sum()
    print(f"Total NaN in window: {nan_count}/{len(history_gapped)}")

    # --- Impute ---
    print("\n=== Imputation ===")
    imputed_series, quality = impute(
        history_gapped[BUILDING_COLUMN],
        history_gapped["Date"],
    )
    history_imputed = history_gapped.copy()
    history_imputed[BUILDING_COLUMN] = imputed_series.values

    remaining_nan = history_imputed[BUILDING_COLUMN].isna().sum()
    print(f"Remaining NaN after imputation: {remaining_nan}")
    print(f"Quality flag distribution:")
    for flag, label in [(0, "real"), (1, "linear"), (2, "seasonal"), (3, "backup")]:
        count = (quality == flag).sum()
        if count > 0:
            print(f"  {flag} ({label}): {count}")

    # --- Prediction on imputed data ---
    print("\n=== Imputed prediction ===")
    imputed_pred = predict_day(target_ts, weather, history_imputed)
    print(f"Imputed predictions: min={imputed_pred.min():.0f}, max={imputed_pred.max():.0f}")

    # --- Evaluate ---
    print("\n=== Evaluation ===")
    results = evaluate(baseline_pred, imputed_pred, actual)
    print_report(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Single-building demo pipeline")
    parser.add_argument(
        "--target-date",
        required=True,
        help="Date to predict (YYYY-MM-DD). Needs 7 days of prior data.",
    )
    parser.add_argument(
        "--gap-mode",
        choices=["blocks", "random"],
        default="blocks",
        help="Gap injection mode (default: blocks)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=36,
        help="Points per gap block (default: 36 = 6 hours)",
    )
    parser.add_argument("--n-blocks", type=int, default=2, help="Number of gap blocks")
    parser.add_argument("--n-points", type=int, default=50, help="Random gap points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
