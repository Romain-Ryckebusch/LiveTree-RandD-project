"""PNG reconstruction overlay for the imputer's output CSV.

Reads the CSV produced by impute_cli (columns: timestamp, value, quality)
and renders a 7-day overlay showing real measurements vs. imputed values,
color-coded by the quality flag.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


QUALITY_LABELS = {
    0: ("Real", "tab:blue"),
    1: ("Naive linear", "tab:green"),
    2: ("Linear / contextual", "tab:orange"),
    3: ("Donor-day ensemble", "tab:red"),
}


def _to_paris_naive(ts):
    t = pd.Timestamp(ts)
    if t.tz is None:
        return t
    return t.tz_convert("Europe/Paris").tz_localize(None)


def render(
    csv_path: str,
    png_path: str,
    building_column: str,
    masked_ranges=None,
) -> None:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    df["quality"] = df["quality"].astype(int)

    window_start = df["timestamp"].min().date()
    window_end = df["timestamp"].max().date()
    n_real = int((df["quality"] == 0).sum())
    n_imputed = int(len(df) - n_real)

    fig, ax = plt.subplots(figsize=(14, 5))

    if masked_ranges:
        for raw_start, raw_end in masked_ranges:
            s = _to_paris_naive(raw_start)
            e = _to_paris_naive(raw_end)
            ax.axvspan(s, e, color="0.85", alpha=0.4, zorder=0, label="_nolegend_")

    ax.plot(
        df["timestamp"], df["value"],
        color="0.7", linewidth=1.0, zorder=1, label="Reconstructed series",
    )
    for flag, (label, color) in QUALITY_LABELS.items():
        mask = df["quality"] == flag
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "timestamp"], df.loc[mask, "value"],
            s=12, color=color, label=f"{label} (n={int(mask.sum())})",
            zorder=2, edgecolors="none",
        )

    ax.set_title(
        f"Reconstruction — {building_column} — "
        f"{window_start} -> {window_end}  "
        f"({n_imputed} imputed / {len(df)} pts)"
    )
    ax.set_xlabel("Local time (Europe/Paris)")
    ax.set_ylabel("Power (W)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    handles, labels = ax.get_legend_handles_labels()
    if masked_ranges:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="0.85", alpha=0.4,
                             label="Test-mode masked range"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
