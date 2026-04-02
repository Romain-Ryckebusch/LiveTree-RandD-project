"""
Plotting module for the evaluation framework.

Generates matplotlib figures for prediction comparison, gap visualization,
metrics bar charts, and multi-scenario degradation analysis.
All plots save to PNG and close the figure to avoid memory leaks in batch mode.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import BUILDING_COLUMN


# ---------------------------------------------------------------------------
# Quality flag colors
# ---------------------------------------------------------------------------

QUALITY_COLORS = {
    0: ("white", "Real data"),
    1: ("#fee08b", "Linear interpolation"),
    2: ("#fdae61", "Contextual reconstruction"),
    3: ("#d73027", "Donor-day reconstruction"),
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot A: Prediction overlay
# ---------------------------------------------------------------------------

def plot_predictions(result, output_dir):
    """
    Time series overlay of baseline vs imputed predictions (+ actual if available).

    Parameters
    ----------
    result : ScenarioResult
    output_dir : str
    """
    _ensure_dir(output_dir)

    timestamps = result.target_timestamps
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(timestamps, result.baseline_pred, label="Baseline (no gaps)",
            color="#2166ac", linewidth=1.5)
    ax.plot(timestamps, result.imputed_pred, label="Imputed",
            color="#d6604d", linewidth=1.5, linestyle="--")

    if result.actual is not None:
        ax.plot(timestamps, result.actual, label="Actual",
                color="#4daf4a", linewidth=1.2, linestyle=":")

    if result.naive_pred is not None:
        ax.plot(timestamps, result.naive_pred, label="Naive",
                color="#e08214", linewidth=1.5, linestyle="-.")

    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Predictions for {result.target_date},{result.scenario_label}")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(output_dir, "predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot A2: Reference prediction (clean data, no gaps)
# ---------------------------------------------------------------------------

def plot_reference(timestamps, baseline_pred, output_dir, target_date):
    """
    Plot the reference (clean-data) prediction for a target date.

    Parameters
    ----------
    timestamps : list of datetime
    baseline_pred : np.ndarray (144,)
    output_dir : str
    target_date : str
    """
    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, baseline_pred, label="Reference prediction (no gaps)",
            color="#2166ac", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Reference prediction for {target_date} ({BUILDING_COLUMN})")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(output_dir, "reference_prediction.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot B: History window with gaps and imputation
# ---------------------------------------------------------------------------

def plot_history_gaps(result, output_dir):
    """
    7-day history window showing clean data, gaps, and color-coded imputation.

    Parameters
    ----------
    result : ScenarioResult
    output_dir : str
    """
    if result.history_dates is None or result.history_clean is None:
        return

    _ensure_dir(output_dir)

    dates = result.history_dates
    if hasattr(dates, "values"):
        dates = dates.values

    fig, ax = plt.subplots(figsize=(14, 5))

    # Clean history as light background reference
    ax.plot(dates, result.history_clean, color="#bdbdbd",
            linewidth=0.8, label="Clean data", zorder=1)

    # Imputed history
    if result.history_imputed is not None:
        ax.plot(dates, result.history_imputed, color="#2166ac",
                linewidth=1.2, label="Imputed data", zorder=2)

    # Color-coded background spans for imputed regions
    if result.quality_flags is not None:
        flags = result.quality_flags
        i = 0
        while i < len(flags):
            if flags[i] > 0:
                flag_val = flags[i]
                start = i
                while i < len(flags) and flags[i] == flag_val:
                    i += 1
                end = min(i, len(dates) - 1)
                color, label = QUALITY_COLORS.get(flag_val, ("#999999", "Unknown"))
                ax.axvspan(dates[start], dates[end], alpha=0.3,
                           color=color, label=label, zorder=0)
            else:
                i += 1

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=8)

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{BUILDING_COLUMN} (W)")
    ax.set_title(f"7-day history,{result.scenario_label}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(output_dir, "history_gaps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot C: Metrics bar chart (single scenario)
# ---------------------------------------------------------------------------

def plot_metrics_bars(result, output_dir):
    """
    Grouped bar chart of MAE, RMSE, MAPE for one scenario.

    Parameters
    ----------
    result : ScenarioResult
    output_dir : str
    """
    _ensure_dir(output_dir)
    metrics = result.metrics
    if not metrics:
        return

    comparisons = list(metrics.keys())
    metric_names = ["mae", "rmse", "mape"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAE and RMSE on the left axis (same unit: watts)
    x = np.arange(len(comparisons))
    width = 0.35
    mae_vals = [metrics[c].get("mae", 0) for c in comparisons]
    rmse_vals = [metrics[c].get("rmse", 0) for c in comparisons]

    ax1.bar(x - width / 2, mae_vals, width, label="MAE", color="#2166ac")
    ax1.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#d6604d")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in comparisons], fontsize=8)
    ax1.set_ylabel("Error (W)")
    ax1.set_title("MAE & RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # MAPE on the right (percentage)
    mape_vals = [metrics[c].get("mape", 0) for c in comparisons]
    ax2.bar(x, mape_vals, width * 2, color="#4daf4a")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in comparisons], fontsize=8)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("MAPE")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Metrics,{result.scenario_label}", fontsize=12)
    fig.tight_layout()

    path = os.path.join(output_dir, "metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot D: Multi-scenario comparison
# ---------------------------------------------------------------------------

def plot_scenario_comparison(results, output_dir, metric="mae"):
    """
    Grouped bar chart comparing smart vs naive imputation across scenarios.

    Parameters
    ----------
    results : list of ScenarioResult
    output_dir : str
    metric : str
        Which metric to plot (default: "mae"). Uses baseline_vs_imputed
        and baseline_vs_naive comparisons.
    """
    if not results:
        return

    _ensure_dir(output_dir)

    labels = []
    smart_vals = []
    naive_vals = []
    for r in results:
        bvi = r.metrics.get("baseline_vs_imputed", {})
        bvn = r.metrics.get("baseline_vs_naive", {})
        s_val = bvi.get(metric, float("nan"))
        n_val = bvn.get(metric, float("nan"))
        if not np.isnan(s_val):
            labels.append(r.scenario_label)
            smart_vals.append(s_val)
            naive_vals.append(n_val if not np.isnan(n_val) else 0)

    if not smart_vals:
        return

    has_naive = any(v > 0 for v in naive_vals)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    x = np.arange(len(labels))

    if has_naive:
        width = 0.35
        bars_smart = ax.bar(x - width / 2, smart_vals, width,
                            label="Contextual imputation", color="#2166ac")
        bars_naive = ax.bar(x + width / 2, naive_vals, width,
                            label="Naive imputation", color="#e08214")
        for bar, val in zip(bars_smart, smart_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_naive, naive_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        bars = ax.bar(x, smart_vals, color="#2166ac", width=0.6)
        for bar, val in zip(bars, smart_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    unit = "%" if metric == "mape" else "W"
    ax.set_ylabel(f"{metric.upper()} ({unit})")
    ax.set_title(f"Prediction degradation by gap scenario ({metric.upper()})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(output_dir, "scenario_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
