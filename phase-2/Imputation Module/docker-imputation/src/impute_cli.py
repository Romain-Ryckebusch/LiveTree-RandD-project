"""CLI wrapper around demo.imputer.impute for the dockerized imputation service.

Reads a 7-day window of holed consumption data from a CSV, runs the
TemperatureAwareHybridEngine via imputer.impute, writes the imputed series
plus per-point quality flags to a CSV.
"""
import argparse
import sys

import numpy as np
import pandas as pd

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

    # Preserve original timestamp strings for the output so the container
    # echoes the caller's format verbatim instead of pandas's default.
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Impute gaps in a 7-day (1008-point, 10-min interval) window of "
            "consumption data using the TemperatureAwareHybridEngine."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV with columns 'timestamp' (ISO 8601) and 'value' (float, NaN for gaps). Must be 1008 rows.",
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

    df, timestamp_strings = load_input(args.input)
    n_gaps = int(df["value"].isna().sum())

    try:
        imputed, quality = impute(df["value"], df["timestamp"], random_seed=args.seed)
    except Exception as exc:
        fail(f"impute() raised {type(exc).__name__}: {exc}")

    remaining = int(np.isnan(imputed.to_numpy()).sum())
    if remaining:
        fail(f"imputation left {remaining} NaN value(s); refusing to write output")

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
