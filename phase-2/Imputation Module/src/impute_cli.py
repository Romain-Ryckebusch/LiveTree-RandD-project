"""CLI wrapper around imputer.impute for the dockerized imputation service.

Reads a 7-day window of holed consumption data from a CSV (--source csv) or
pulls it from Cassandra (--source cassandra), runs the
TemperatureAwareHybridEngine via imputer.impute, and writes the imputed
series plus per-point quality flags to a CSV.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import imputer
from imputer import impute

EXPECTED_ROWS = 1008
EXPECTED_FREQ_SECONDS = 600
FREQ_TOLERANCE_SECONDS = 1


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_input(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        fail(f"input file not found: {path}")
    except Exception as exc:
        fail(f"could not read input CSV {path}: {exc}")

    missing = {"timestamp", "value"} - set(df.columns)
    if missing:
        fail(
            f"input CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    if len(df) != EXPECTED_ROWS:
        fail(
            f"input CSV must have exactly {EXPECTED_ROWS} rows "
            f"(7 days x 144 points/day), got {len(df)}"
        )

    # Keep input timestamp strings for the output (don't reformat via pandas).
    timestamp_strings = df["timestamp"].astype(str).copy()

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as exc:
        fail(f"could not parse 'timestamp' column as datetimes: {exc}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if df["value"].isna().all():
        fail("input CSV has no non-NaN values; nothing to impute from")

    deltas = df["timestamp"].diff().dropna().dt.total_seconds()
    if (deltas <= 0).any():
        fail("input CSV timestamps must be strictly increasing")
    off_grid = (deltas - EXPECTED_FREQ_SECONDS).abs() > FREQ_TOLERANCE_SECONDS
    if off_grid.any():
        n_bad = int(off_grid.sum())
        fail(
            f"input CSV timestamps must be on a 10-minute grid; "
            f"{n_bad} interval(s) deviate by more than {FREQ_TOLERANCE_SECONDS}s"
        )

    return df, timestamp_strings


def load_cassandra_window(target_date):
    """Pull history + weather from Cassandra and extract the 7-day window.

    Returns the same (df, timestamp_strings) tuple as load_input.
    Also seeds imputer's history cache from the Cassandra pull.
    """
    try:
        from cassandra_client import (
            load_historical_data_cassandra,
            load_weather_data_cassandra,
        )
    except ImportError as exc:
        fail(f"could not import cassandra_client: {exc}")

    try:
        hist_df = load_historical_data_cassandra()
    except Exception as exc:
        fail(f"could not load history from Cassandra: {type(exc).__name__}: {exc}")

    try:
        weather_df = load_weather_data_cassandra()
    except Exception as exc:
        fail(f"could not load weather from Cassandra: {type(exc).__name__}: {exc}")

    from config import BUILDING_COLUMN
    from window import extract_window

    try:
        history, _target_ts, _weather, _actual = extract_window(
            hist_df, target_date, weather_df
        )
    except Exception as exc:
        fail(f"extract_window failed for {target_date}: {type(exc).__name__}: {exc}")

    if len(history) != EXPECTED_ROWS:
        fail(
            f"Cassandra window for {target_date} has {len(history)} rows, "
            f"expected {EXPECTED_ROWS}"
        )
    if history[BUILDING_COLUMN].isna().all():
        fail(
            f"Cassandra window for {target_date} has no Ptot_HA values at all"
        )

    # Seed the engine's 56-day history from the Cassandra pull.
    imputer.set_history_source(
        hist_df.rename(columns={"Date": "Timestamp"})[["Timestamp", BUILDING_COLUMN]]
    )

    timestamp_strings = history["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").copy()
    df = pd.DataFrame({
        "timestamp": history["Date"],
        "value": history[BUILDING_COLUMN].astype(float),
    })
    return df, timestamp_strings


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Impute gaps in a 7-day (1008-point, 10-min interval) window of "
            "consumption data using the TemperatureAwareHybridEngine."
        )
    )
    parser.add_argument(
        "--source", choices=["csv", "cassandra"], default="csv",
        help="Input data source (default: csv).",
    )
    parser.add_argument(
        "--input",
        help="CSV mode only: input CSV with columns 'timestamp' (ISO 8601) and 'value' (float, NaN for gaps). Must be 1008 rows.",
    )
    parser.add_argument(
        "--target-date",
        help="Cassandra mode only: YYYY-MM-DD. The 7-day window ending the day before this date will be pulled from Cassandra.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path. Will contain columns 'timestamp', 'value' (imputed), 'quality' (0=real, 1=linear, 2=contextual, 3=donor-day).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional random seed for deterministic imputation.",
    )
    args = parser.parse_args()

    if args.source == "csv":
        if not args.input:
            fail("--input is required when --source=csv")
        df, timestamp_strings = load_input(args.input)
    else:
        if not args.target_date:
            fail("--target-date is required when --source=cassandra")
        df, timestamp_strings = load_cassandra_window(args.target_date)

    n_gaps = int(df["value"].isna().sum())

    try:
        imputed, quality = impute(df["value"], df["timestamp"], random_seed=args.seed)
    except Exception as exc:
        fail(f"impute() raised {type(exc).__name__}: {exc}")

    remaining = int(np.isnan(imputed.to_numpy()).sum())
    if remaining:
        fail(f"imputation left {remaining} NaN value(s)")

    out = pd.DataFrame({
        "timestamp": timestamp_strings,
        "value": imputed.to_numpy(),
        "quality": quality.to_numpy().astype(int),
    })
    try:
        out.to_csv(args.output, index=False, line_terminator="\n")
    except Exception as exc:
        fail(f"could not write output CSV {args.output}: {exc}")

    print(f"[OK] Imputed {n_gaps} gap point(s) -> {args.output}")


if __name__ == "__main__":
    main()
