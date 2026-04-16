"""CLI wrapper around imputer.impute for the dockerized imputation service.

Reads a 7-day window of holed consumption data from a CSV (--source csv) or
pulls it from Cassandra (--source cassandra), runs the
TemperatureAwareHybridEngine via imputer.impute, and writes the imputed
series plus per-point quality flags to a CSV. --building picks the Cassandra
column to reconstruct; Ptot_Campus is the sum of the four buildings.
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


def parse_test_gap(raw_start, raw_end):
    """Parse one --test-gap pair as pandas Timestamps.

    Bounds are treated as naive Europe/Paris local time if no tz is supplied.
    Actual tz alignment against the DataFrame's timestamp column happens in
    apply_test_gaps. Hard-fails on unparseable input or start > end.
    """
    try:
        start = pd.Timestamp(raw_start)
        end = pd.Timestamp(raw_end)
    except Exception as exc:
        fail(f"could not parse --test-gap '{raw_start}' '{raw_end}': {exc}")
    if start > end:
        fail(
            f"--test-gap start must be <= end, got '{raw_start}' > '{raw_end}'"
        )
    return start, end


def apply_test_gaps(df, gaps, value_col="value", source="csv"):
    """Mask df[value_col] where timestamp falls in any [start, end] (inclusive).

    Returns (df_masked, ground_truth, mask):
        df_masked: copy of df with value_col NaN'd on masked rows.
        ground_truth: np.ndarray[float] aligned to df.index — NaN outside the
            mask, original values inside (incl. NaN if the source was already
            NaN).
        mask: np.ndarray[bool] — True where any gap covers the row.

    Handles naive (CSV) vs tz-aware (Cassandra, UTC) timestamps by detecting
    df[timestamp].dt.tz and localizing user bounds with config.TIMEZONE
    (Europe/Paris) before converting when needed.
    """
    from config import TIMEZONE

    df = df.copy()
    ts = df["timestamp"]
    tz = ts.dt.tz
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    ts_min = ts.min()
    ts_max = ts.max()

    for i, (raw_start, raw_end) in enumerate(gaps):
        start = pd.Timestamp(raw_start)
        end = pd.Timestamp(raw_end)
        if tz is None:
            if start.tz is not None or end.tz is not None:
                fail(
                    f"--test-gap #{i+1}: DataFrame timestamps are naive but "
                    f"the supplied bound is timezone-aware"
                )
        else:
            if start.tz is None:
                start = start.tz_localize(TIMEZONE)
            if end.tz is None:
                end = end.tz_localize(TIMEZONE)
            start = start.tz_convert(tz)
            end = end.tz_convert(tz)

        if end < ts_min or start > ts_max:
            fail(
                f"--test-gap #{i+1} [{raw_start}, {raw_end}] is fully outside "
                f"the 7-day window [{ts_min}, {ts_max}]"
            )
        if start < ts_min or end > ts_max:
            print(
                f"[WARN] --test-gap #{i+1} [{raw_start}, {raw_end}] partially "
                f"outside window [{ts_min}, {ts_max}]; clipping.",
                file=sys.stderr,
            )
            start = max(start, ts_min)
            end = min(end, ts_max)

        gap_mask = ((ts >= start) & (ts <= end)).to_numpy()
        if gap_mask.any() and df.loc[gap_mask, value_col].isna().all():
            print(
                f"[WARN] --test-gap #{i+1} [{raw_start}, {raw_end}]: all "
                f"{int(gap_mask.sum())} targeted row(s) were already NaN.",
                file=sys.stderr,
            )
        mask |= gap_mask

    if mask.all():
        if source == "csv":
            fail(
                "--test-gap masks the entire 7-day window; CSV mode has no "
                "history to reconstruct from."
            )
        print(
            "[WARN] --test-gap masks the entire 7-day window; reconstruction "
            "will rely solely on the 56-day Cassandra history.",
            file=sys.stderr,
        )

    ground_truth = np.full(n, np.nan, dtype=float)
    ground_truth[mask] = df.loc[mask, value_col].to_numpy(dtype=float)
    df.loc[mask, value_col] = np.nan
    return df, ground_truth, mask


def write_test_report(
    path, timestamp_strings, ground_truth, imputed, quality, mask,
):
    """Write per-masked-point report CSV and print summary metrics.

    Report schema: timestamp, ground_truth, imputed, quality, abs_error.
    Metrics (MAE, RMSE, max_err) are computed over masked rows where
    ground_truth is not NaN, written as '# key=value' footer comments, and
    echoed on stdout.
    """
    masked_idx = np.flatnonzero(mask)
    gt = ground_truth[masked_idx]
    im = imputed[masked_idx]
    qu = quality[masked_idx]
    abs_err = np.abs(im - gt)
    valid = ~np.isnan(gt)

    report_df = pd.DataFrame({
        "timestamp": timestamp_strings.iloc[masked_idx].to_numpy(),
        "ground_truth": gt,
        "imputed": im,
        "quality": qu.astype(int),
        "abs_error": abs_err,
    })

    if valid.any():
        mae = float(np.nanmean(abs_err[valid]))
        rmse = float(np.sqrt(np.nanmean(abs_err[valid] ** 2)))
        max_err = float(np.nanmax(abs_err[valid]))
    else:
        mae = rmse = max_err = float("nan")

    try:
        with open(path, "w", newline="\n") as fh:
            report_df.to_csv(fh, index=False, line_terminator="\n")
            fh.write(f"# MAE={mae:.4f}\n")
            fh.write(f"# RMSE={rmse:.4f}\n")
            fh.write(f"# max_err={max_err:.4f}\n")
            fh.write(f"# n_points={len(report_df)}\n")
            fh.write(f"# n_ground_truth={int(valid.sum())}\n")
    except Exception as exc:
        fail(f"could not write test report CSV {path}: {exc}")

    print(
        f"[TEST] n={len(report_df)} (gt={int(valid.sum())})  "
        f"MAE={mae:.2f}  RMSE={rmse:.2f}  max={max_err:.2f}"
    )


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


def load_cassandra_window(target_date, building_column):
    """Pull history + weather from Cassandra and extract the 7-day window.

    Returns the same (df, timestamp_strings) tuple as load_input, and seeds
    imputer's history cache for ``building_column``. For Ptot_Campus the
    column is built as the sum of the four buildings with skipna=False, so a
    single missing component leaves the row missing.
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

    from config import CAMPUS_COMPONENTS
    from window import extract_window

    if building_column == "Ptot_Campus":
        missing_components = [c for c in CAMPUS_COMPONENTS if c not in hist_df.columns]
        if missing_components:
            fail(
                f"Cassandra is missing Campus component columns: {missing_components}"
            )
        hist_df = hist_df.copy()
        hist_df["Ptot_Campus"] = hist_df[CAMPUS_COMPONENTS].sum(axis=1, skipna=False)
    elif building_column not in hist_df.columns:
        fail(
            f"Cassandra conso table does not contain column '{building_column}'. "
            f"Available: {sorted(c for c in hist_df.columns if c.startswith('Ptot_'))}"
        )

    try:
        history, _target_ts, _weather, _actual = extract_window(
            hist_df, target_date, weather_df, building_column=building_column
        )
    except Exception as exc:
        fail(f"extract_window failed for {target_date}: {type(exc).__name__}: {exc}")

    if len(history) != EXPECTED_ROWS:
        fail(
            f"Cassandra window for {target_date} has {len(history)} rows, "
            f"expected {EXPECTED_ROWS}"
        )
    if history[building_column].isna().all():
        fail(
            f"Cassandra window for {target_date} has no {building_column} values at all"
        )

    # Seed the engine's 56-day history from the Cassandra pull.
    imputer.set_history_source(
        hist_df.rename(columns={"Date": "Timestamp"})[["Timestamp", building_column]],
        building_column=building_column,
    )

    timestamp_strings = history["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").copy()
    df = pd.DataFrame({
        "timestamp": history["Date"],
        "value": history[building_column].astype(float),
    })
    return df, timestamp_strings


def main():
    from config import BUILDINGS, BUILDING_COLUMN

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
        "--building", choices=BUILDINGS, default=BUILDING_COLUMN,
        help=(
            "Cassandra mode: which building/column to reconstruct "
            f"(default: {BUILDING_COLUMN}). Ignored in CSV mode."
        ),
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path. Will contain columns 'timestamp', 'value' (imputed), 'quality' (0=real, 1=linear, 2=contextual, 3=donor-day).",
    )
    parser.add_argument(
        "--plot",
        help="Optional PNG path. If given, renders a reconstruction overlay plot alongside the output CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional random seed for deterministic imputation.",
    )
    parser.add_argument(
        "--test-gap", nargs=2, metavar=("START", "END"),
        action="append", default=None,
        help=(
            "Test mode: mask rows in [START, END] (inclusive) with NaN in "
            "memory before imputation. START/END are naive Europe/Paris "
            "datetimes, e.g. '2026-04-10 08:00'. Repeatable. The source "
            "(CSV file or Cassandra cluster) is never modified."
        ),
    )
    parser.add_argument(
        "--test-report",
        help=(
            "Test mode only: path to a CSV with per-point ground-truth vs "
            "imputed values for masked rows, plus MAE/RMSE/max footer."
        ),
    )
    args = parser.parse_args()

    if args.source == "csv":
        if not args.input:
            fail("--input is required when --source=csv")
        if args.building != BUILDING_COLUMN:
            print(
                f"[NOTE] --building={args.building} is ignored in CSV mode "
                f"(CSV uses a generic 'value' column)."
            )
        df, timestamp_strings = load_input(args.input)
        building_column = BUILDING_COLUMN
    else:
        if not args.target_date:
            fail("--target-date is required when --source=cassandra")
        df, timestamp_strings = load_cassandra_window(args.target_date, args.building)
        building_column = args.building

    gt_values = None
    test_mask = None
    test_gaps = None
    if args.test_gap:
        if args.test_report is None:
            print(
                "[NOTE] --test-gap is set but no --test-report; output "
                "'quality' column still flags imputed points.",
                file=sys.stderr,
            )
        test_gaps = [parse_test_gap(s, e) for s, e in args.test_gap]
        df, gt_values, test_mask = apply_test_gaps(
            df, test_gaps, value_col="value", source=args.source,
        )
    elif args.test_report:
        fail("--test-report requires at least one --test-gap")

    n_gaps = int(df["value"].isna().sum())

    try:
        imputed, quality = impute(
            df["value"], df["timestamp"],
            random_seed=args.seed, building_column=building_column,
        )
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

    if test_mask is not None:
        n_synthetic = int(test_mask.sum())
        print(
            f"[OK] Imputed {n_gaps} gap point(s) "
            f"({n_synthetic} synthetic from --test-gap) -> {args.output}"
        )
    else:
        print(f"[OK] Imputed {n_gaps} gap point(s) -> {args.output}")

    if args.test_report:
        write_test_report(
            args.test_report,
            timestamp_strings,
            gt_values,
            imputed.to_numpy(),
            quality.to_numpy().astype(int),
            test_mask,
        )
        print(f"[OK] Test report -> {args.test_report}")

    if args.plot:
        try:
            from plot_reconstruction import render
            render(
                args.output, args.plot, building_column,
                masked_ranges=test_gaps,
            )
        except Exception as exc:
            fail(f"plot_reconstruction.render() raised {type(exc).__name__}: {exc}")
        print(f"[OK] Plot -> {args.plot}")


if __name__ == "__main__":
    main()
