#!/usr/bin/env python3
"""
Plot consumption time-series from one or two parquet files.

Features:
- Plot one or several building columns over time
- Optional comparison: overlay a second parquet file (e.g. reconstructed data)
- Optional date filtering and resampling
- Holiday / closed-day shading based on is_holiday / is_closed columns (on by default)
"""

import argparse
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Adapt if your schema differs
DEFAULT_CONSUMPTION_COLS = ["HA", "HEI1", "HEI2", "RIZOMM", "Campus"]


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "DateTime" not in df.columns:
        raise ValueError(f"{path} must contain a 'DateTime' column.")

    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")
    return df


def select_target_columns(df: pd.DataFrame, spec: str) -> List[str]:
    """
    spec: 'all' or comma-separated list of column names.
    Returns the list of columns that actually exist in df.
    """
    if spec.lower() == "all":
        candidate = [c for c in DEFAULT_CONSUMPTION_COLS if c in df.columns]
    else:
        candidate = [c.strip() for c in spec.split(",") if c.strip()]

    cols = [c for c in candidate if c in df.columns]
    if not cols:
        raise ValueError(
            f"No valid target columns found for spec '{spec}'. "
            f"Available columns: {list(df.columns)}"
        )
    return cols


def filter_and_resample(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    resample_freq: Optional[str],
) -> pd.DataFrame:
    if start is not None:
        start_dt = pd.to_datetime(start)
        df = df[df.index >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(end)
        df = df[df.index <= end_dt]

    if resample_freq:
        # Simple mean aggregation when resampling
        df = df.resample(resample_freq).mean()

    return df


def compute_spans(bool_series: pd.Series):
    """
    From a boolean series indexed by DateTime, returns a list of (start, end) intervals
    where the value is True.
    """
    if bool_series.empty:
        return []

    values = bool_series.to_numpy()
    index = bool_series.index

    spans = []
    in_span = False
    start = None

    for i, v in enumerate(values):
        if v and not in_span:
            in_span = True
            start = index[i]
        elif (not v) and in_span:
            end = index[i]
            spans.append((start, end))
            in_span = False

    if in_span:
        spans.append((start, index[-1]))

    return spans


def apply_shading(ax, spans, label: str, alpha: float):
    """
    Shade vertical regions on the plot for given (start, end) spans.
    Label is used only for the first span to avoid legend clutter.
    """
    for i, (start, end) in enumerate(spans):
        lbl = label if i == 0 else None
        ax.axvspan(start, end, alpha=alpha, label=lbl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot consumption data from parquet files, with optional comparison and shading."
    )

    parser.add_argument(
        "file1",
        help="Path to first parquet file (e.g. original data).",
    )
    parser.add_argument(
        "file2",
        nargs="?",
        default=None,
        help="Optional path to second parquet file (e.g. reconstructed data).",
    )

    parser.add_argument(
        "-c",
        "--columns",
        default="all",
        help=(
            "Which consumption columns to plot: 'all' or comma-separated list "
            f"(default: 'all' -> {DEFAULT_CONSUMPTION_COLS})."
        ),
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start datetime (e.g. '2021-01-01' or '2021-01-01T00:00').",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end datetime.",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Optional resampling frequency (e.g. '30T', '1H', '1D').",
    )

    parser.add_argument(
        "--no-shading",
        action="store_true",
        help="Disable all shading (holidays and closed days).",
    )
    parser.add_argument(
        "--no-holiday-shading",
        action="store_true",
        help="Disable holiday shading only.",
    )
    parser.add_argument(
        "--no-closed-shading",
        action="store_true",
        help="Disable closed-day shading only.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save figure to this path instead of showing (e.g. 'plot.png').",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df1 = load_parquet(args.file1)
    df1 = filter_and_resample(df1, args.start, args.end, args.resample)

    df2 = None
    if args.file2 is not None:
        df2 = load_parquet(args.file2)
        df2 = filter_and_resample(df2, args.start, args.end, args.resample)

    cols = select_target_columns(df1, args.columns)
    print(f"Plotting columns: {cols}")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot file1 (solid)
    for col in cols:
        if col not in df1.columns:
            print(f"Warning: column '{col}' not found in {args.file1}, skipping.")
            continue
        ax.plot(df1.index, df1[col], label=f"{col} (file1)")

    # Plot file2 (dashed) if present
    if df2 is not None:
        for col in cols:
            if col not in df2.columns:
                print(f"Warning: column '{col}' not found in {args.file2}, skipping.")
                continue
            ax.plot(df2.index, df2[col], linestyle="--", label=f"{col} (file2)")

    # Axes labels and title
    ax.set_xlabel("DateTime")
    ax.set_ylabel("Consumption")

    if args.title:
        ax.set_title(args.title)
    else:
        if df2 is None:
            ax.set_title("Consumption time series")
        else:
            ax.set_title("Consumption time series (comparison)")

    # Date formatting on x-axis
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Shading (from file1 metadata)
    if not args.no_shading:
        if not args.no_holiday_shading and "is_holiday" in df1.columns:
            holiday_spans = compute_spans(df1["is_holiday"].astype(bool))
            apply_shading(ax, holiday_spans, "Holiday", alpha=0.1)
        if not args.no_closed_shading and "is_closed" in df1.columns:
            closed_spans = compute_spans(df1["is_closed"].astype(bool))
            apply_shading(ax, closed_spans, "Closed", alpha=0.06)

    # Legend (deduplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = {}
        for h, lab in zip(handles, labels):
            unique[lab] = h
        ax.legend(unique.values(), unique.keys(), loc="best")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
