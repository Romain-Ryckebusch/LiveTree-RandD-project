#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_align(
    original_path: str,
    imputed_path: str,
    cols: List[str],
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """
    Load original and imputed parquet files and align them on datetime and columns.
    Returns a DataFrame with columns:
      datetime_col, and for each col in cols: col + "_orig", col + "_imp"
    """
    df_orig = pd.read_parquet(original_path)
    df_imp = pd.read_parquet(imputed_path)

    df_orig[datetime_col] = pd.to_datetime(df_orig[datetime_col], utc=True)
    df_imp[datetime_col] = pd.to_datetime(df_imp[datetime_col], utc=True)

    needed_cols = [datetime_col] + cols

    missing_orig = [c for c in needed_cols if c not in df_orig.columns]
    missing_imp = [c for c in needed_cols if c not in df_imp.columns]
    if missing_orig:
        raise ValueError(f"Missing columns in original file: {missing_orig}")
    if missing_imp:
        raise ValueError(f"Missing columns in imputed file: {missing_imp}")

    df_o = df_orig[needed_cols]
    df_i = df_imp[needed_cols]

    df = pd.merge(
        df_o,
        df_i,
        on=datetime_col,
        suffixes=("_orig", "_imp"),
        how="inner",
    )

    df = df.sort_values(datetime_col).reset_index(drop=True)
    return df


def compute_metrics_for_column(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
    window_points: int = 6,
) -> Dict[str, Any]:
    """
    Compute metrics for a single column:
    - global RMSE over replaced part
    - RMSE (%) over replaced part relative to mean original
    - per-day RMSE over replaced part
    - sliding-window RMSE over 1-day window (window_points)
    - correlation coefficient over replaced part
    - index range where any replacement occurred (idx_min, idx_max)
    """
    y_true = df[f"{col}_orig"].to_numpy(dtype=float)
    y_pred = df[f"{col}_imp"].to_numpy(dtype=float)

    # Valid where both are finite numbers
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    # "Replaced" where they differ (with tolerance)
    diff_mask = valid_mask & (~np.isclose(y_true, y_pred, rtol=1e-9, atol=1e-12))
    n_replaced = int(diff_mask.sum())

    metrics: Dict[str, Any] = {
        "column": col,
        "n_points_total": int(len(y_true)),
        "n_points_replaced": n_replaced,
    }

    if n_replaced == 0:
        # Nothing was replaced, skip detailed metrics
        metrics["rmse_replaced"] = np.nan
        metrics["rmse_replaced_pct"] = np.nan
        metrics["corr_replaced"] = np.nan
        metrics["rmse_per_day"] = pd.DataFrame(
            columns=["date", "rmse", "rmse_pct", "n_points_replaced"]
        )
        metrics["sliding_rmse"] = pd.DataFrame(
            {datetime_col: df[datetime_col], f"{col}_rmse_1d": np.nan}
        )
        metrics["idx_min"] = None
        metrics["idx_max"] = None
        metrics["diff_mask"] = diff_mask
        return metrics

    # Index range where anything was replaced
    replaced_indices = np.where(diff_mask)[0]
    idx_min = int(replaced_indices.min())
    idx_max = int(replaced_indices.max())

    metrics["idx_min"] = idx_min
    metrics["idx_max"] = idx_max
    metrics["diff_mask"] = diff_mask

    # --- Global RMSE over replaced part ---
    err_repl = y_pred[diff_mask] - y_true[diff_mask]
    rmse_global = float(np.sqrt(np.mean(err_repl**2)))
    metrics["rmse_replaced"] = rmse_global

    # RMSE percentage relative to mean original over replaced part
    mean_true_repl = float(np.mean(y_true[diff_mask]))
    if abs(mean_true_repl) > 1e-12:
        rmse_pct = 100.0 * rmse_global / mean_true_repl
    else:
        rmse_pct = np.nan
    metrics["rmse_replaced_pct"] = rmse_pct

    # --- Correlation over replaced part ---
    if n_replaced > 1:
        corr = float(np.corrcoef(y_true[diff_mask], y_pred[diff_mask])[0, 1])
    else:
        corr = np.nan
    metrics["corr_replaced"] = corr

    # --- RMSE per day (only days with at least one replaced point) ---
    df_col = pd.DataFrame(
        {
            datetime_col: df[datetime_col].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "replaced": diff_mask,
        }
    )
    df_col["date"] = df_col[datetime_col].dt.normalize()

    per_day_rows = []
    for date, g in df_col.groupby("date"):
        mask_day = g["replaced"].to_numpy()
        if not mask_day.any():
            continue
        true_day = g["y_true"].to_numpy()[mask_day]
        pred_day = g["y_pred"].to_numpy()[mask_day]
        e_day = pred_day - true_day
        rmse_day = float(np.sqrt(np.mean(e_day**2)))
        mean_true_day = float(np.mean(true_day))
        if abs(mean_true_day) > 1e-12:
            rmse_day_pct = 100.0 * rmse_day / mean_true_day
        else:
            rmse_day_pct = np.nan
        per_day_rows.append(
            {
                "date": date,
                "rmse": rmse_day,
                "rmse_pct": rmse_day_pct,
                "n_points_replaced": int(mask_day.sum()),
            }
        )

    rmse_per_day_df = pd.DataFrame(per_day_rows).sort_values("date").reset_index(
        drop=True
    )
    metrics["rmse_per_day"] = rmse_per_day_df

    # --- Sliding RMSE window (1 hour = window_points samples) ---
    # Only replaced points contribute; others are NaN
    se = (y_pred - y_true) ** 2
    se[~diff_mask] = np.nan  # only replaced part contributes
    se_series = pd.Series(se, index=df[datetime_col])

    rmse_sliding = np.sqrt(
        se_series.rolling(window=window_points, min_periods=1).mean()
    )

    sliding_df = pd.DataFrame(
        {
            datetime_col: df[datetime_col],
            f"{col}_rmse_1d": rmse_sliding.to_numpy(),
        }
    )
    metrics["sliding_rmse"] = sliding_df

    return metrics


def plot_column(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    col: str,
    datetime_col: str = "DateTime",
    window_points: int = 6,
    save_dir: Path | None = None,
    show: bool = True,
):
    """
    Plot original, imputed, and sliding 1-day RMSE for one column,
    restricted to the time span where data was replaced.
    """
    idx_min = metrics["idx_min"]
    idx_max = metrics["idx_max"]

    if idx_min is None or idx_max is None:
        # Nothing replaced, nothing to plot
        return

    # Restrict to the replaced span
    df_slice = df.iloc[idx_min : idx_max + 1].copy()
    time = df_slice[datetime_col]
    y_true = df_slice[f"{col}_orig"]
    y_pred = df_slice[f"{col}_imp"]

    sliding_df = metrics["sliding_rmse"].iloc[idx_min : idx_max + 1]
    rmse_series = sliding_df[f"{col}_rmse_1d"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(time, y_true, label="Original")
    ax1.plot(time, y_pred, label="Imputed")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Consumption")

    ax2 = ax1.twinx()
    ax2.plot(time, rmse_series, label="Sliding RMSE (1 hour)", linestyle=":")
    ax2.set_ylabel("RMSE")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    rmse_val = metrics["rmse_replaced"]
    rmse_pct = metrics["rmse_replaced_pct"]
    title = (
        f"{col} - Original vs Imputed (replaced span)\n"
        f"RMSE = {rmse_val:.4f} ({rmse_pct:.2f} % of mean original on replaced part)"
    )
    ax1.set_title(title)

    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{col}_comparison.png"
        fig.savefig(out_file, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and imputed parquet files, compute RMSE/correlation "
            "on the replaced part, and plot original/imputed/sliding-RMSE curves."
        )
    )
    parser.add_argument("original", help="Path to original (ground truth) parquet file")
    parser.add_argument("imputed", help="Path to imputed parquet file")
    parser.add_argument(
        "--cols",
        nargs="+",
        required=True,
        help="Columns to evaluate (e.g. HA HEI1 HEI2 RIZOMM Campus)",
    )
    parser.add_argument(
        "--datetime-col",
        default="DateTime",
        help="Datetime column name (default: DateTime)",
    )
    parser.add_argument(
        "--window-points",
        type=int,
        default=6,
        help="Sliding window size in points for 1-day RMSE (default: 6 for 10-min data)",
    )
    parser.add_argument(
        "--sliding-output",
        help=(
            "Optional CSV file to write sliding RMSE time series "
            "(all columns concatenated on DateTime)"
        ),
        default=None,
    )
    parser.add_argument(
        "--plot-dir",
        help="Optional directory to save plots as PNG (one per column)",
        default=None,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively (only save if --plot-dir is set)",
    )

    args = parser.parse_args()

    # Load and align
    df = load_and_align(args.original, args.imputed, args.cols, args.datetime_col)

    # Compute metrics per column
    all_metrics: Dict[str, Any] = {}
    sliding_all = pd.DataFrame({args.datetime_col: df[args.datetime_col]})

    print("=== Global metrics over replaced part ===")
    for col in args.cols:
        metrics = compute_metrics_for_column(
            df, col, datetime_col=args.datetime_col, window_points=args.window_points
        )
        all_metrics[col] = metrics

        print(f"\nColumn: {col}")
        print(f"  Total points:     {metrics['n_points_total']}")
        print(f"  Replaced points:  {metrics['n_points_replaced']}")

        rmse_val = metrics["rmse_replaced"]
        rmse_pct = metrics["rmse_replaced_pct"]
        print(
            f"  RMSE (replaced):  {rmse_val:.4f} "
            f"({rmse_pct:.2f} % of mean original on replaced part)"
        )
        print(f"  Corr (replaced):  {metrics['corr_replaced']:.4f}")

        # Per-day RMSE summary
        rmse_day_df = metrics["rmse_per_day"]
        if not rmse_day_df.empty:
            print("  Per-day RMSE (first 10 days):")
            print(rmse_day_df.head(10).to_string(index=False))
        else:
            print("  Per-day RMSE: no days with replaced points.")

        # Sliding RMSE summary
        sliding_df = metrics["sliding_rmse"]
        # merge into global sliding dataframe
        sliding_all = sliding_all.merge(
            sliding_df, on=args.datetime_col, how="left"
        )

        rmse_vals = sliding_df.iloc[:, 1].dropna().to_numpy()
        if rmse_vals.size > 0:
            mean_rmse = float(np.mean(rmse_vals))
            median_rmse = float(np.median(rmse_vals))
            max_rmse = float(np.max(rmse_vals))
            p95_rmse = float(np.percentile(rmse_vals, 95))
            print(
                f"  Sliding RMSE {args.window_points} pts (over replaced part): "
                f"mean={mean_rmse:.4f}, median={median_rmse:.4f}, "
                f"p95={p95_rmse:.4f}, max={max_rmse:.4f}"
            )
        else:
            print(
                f"  Sliding RMSE {args.window_points} pts: no windows with replaced points."
            )

        # Plot for this column over replaced span only
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else None
        plot_column(
            df,
            metrics,
            col,
            datetime_col=args.datetime_col,
            window_points=args.window_points,
            save_dir=plot_dir,
            show=not args.no_show,
        )

    # Optionally write sliding RMSE time series
    if args.sliding_output is not None:
        out_path = Path(args.sliding_output)
        sliding_all.to_csv(out_path, index=False)
        print(f"\nSliding RMSE time series written to: {out_path}")


if __name__ == "__main__":
    main()

