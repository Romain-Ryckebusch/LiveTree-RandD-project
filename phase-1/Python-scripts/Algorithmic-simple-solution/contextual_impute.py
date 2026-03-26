#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

EDGE_SCALE_MIN = 0.1
EDGE_SCALE_MAX = 10.0


# ====================
# Logging setup
# ====================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure root logger with given level and optional log file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if log_file:
        # Add file handler in addition to stderr
        fh = logging.FileHandler(log_file)
        fh.setLevel(numeric_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(fh)


# ============
# 1. Context
# ============

def prepare_context(df: pd.DataFrame, datetime_col: str = "DateTime") -> pd.DataFrame:
    """
    Ensure calendar/context columns exist:
    - dayofweek, month, hour, minute
    - is_weekend, is_night
    - is_closed, is_holiday, is_special_day (default False if absent)
    - day_type (open_workday / closed_weekday / weekend / holiday)
    - slot (10-min slot index 0..143)
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.sort_values(datetime_col).reset_index(drop=True)
    dt = df[datetime_col].dt

    if "dayofweek" not in df.columns:
        df["dayofweek"] = dt.dayofweek
    if "month" not in df.columns:
        df["month"] = dt.month
    if "hour" not in df.columns:
        df["hour"] = dt.hour
    if "minute" not in df.columns:
        df["minute"] = dt.minute

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["dayofweek"] >= 5
    if "is_night" not in df.columns:
        df["is_night"] = (df["hour"] >= 22) | (df["hour"] < 6)

    for col in ["is_closed", "is_holiday", "is_special_day"]:
        if col not in df.columns:
            df[col] = False

    def _day_type(row: pd.Series) -> str:
        if row["is_holiday"] or row["is_special_day"]:
            return "holiday"
        if row["is_weekend"]:
            return "weekend"
        if row["is_closed"]:
            return "closed_weekday"
        return "open_workday"

    df["day_type"] = df.apply(_day_type, axis=1)
    df["slot"] = df["hour"] * 6 + df["minute"] // 10

    logger.info(
        "Prepared context: %d rows, from %s to %s",
        len(df),
        df[datetime_col].min(),
        df[datetime_col].max(),
    )
    return df


# ===============================
# 2. Slot-level contextual means
# ===============================

def build_profiles(df: pd.DataFrame, col: str):
    """
    Slot baselines:
      (day_type, month, slot) -> mean
      (day_type, slot)        -> mean
      (slot)                  -> mean
      global_mean             -> scalar
    """
    non_missing = df[df[col].notna()]
    if non_missing.empty:
        logger.warning("Column %s has no non-missing data; profiles empty.", col)
        return {
            "dms": pd.Series(dtype=float),
            "ds": pd.Series(dtype=float),
            "s": pd.Series(dtype=float),
            "global_mean": 0.0,
        }

    dms = non_missing.groupby(["day_type", "month", "slot"])[col].mean()
    ds = non_missing.groupby(["day_type", "slot"])[col].mean()
    s = non_missing.groupby(["slot"])[col].mean()
    global_mean = non_missing[col].mean()

    logger.info(
        "Built profiles for %s: %d (day_type,month,slot); %d (day_type,slot); %d (slot)",
        col,
        len(dms),
        len(ds),
        len(s),
    )

    return {"dms": dms, "ds": ds, "s": s, "global_mean": float(global_mean)}


def make_baseline_getter(profiles):
    dms = profiles["dms"]
    ds = profiles["ds"]
    s = profiles["s"]
    global_mean = profiles["global_mean"]

    def get_baseline(day_type: str, month: int, slot: int) -> float:
        # Most specific
        val = dms.get((day_type, month, slot), np.nan)
        if not np.isnan(val):
            return float(val)
        # Without month
        val = ds.get((day_type, slot), np.nan)
        if not np.isnan(val):
            return float(val)
        # Only slot
        val = s.get(slot, np.nan)
        if not np.isnan(val):
            return float(val)
        # Fallback
        return float(global_mean)

    return get_baseline


# ==========================================
# 3. Library of full historical "donor days"
# ==========================================

def build_daily_library(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
    max_missing_frac: float = 0.05,
) -> Dict[Tuple[str, int, int], List[Dict[str, object]]]:
    """
    Build a library of historical full days for one column.
    Key: (day_type, dayofweek, month) -> list of dicts:
        {"date": date, "values": np.ndarray}
    Only days with <= max_missing_frac missing values and no NaNs
    (after dropping) are kept as donors.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df["date"] = df[datetime_col].dt.normalize()

    library: Dict[Tuple[str, int, int], List[Dict[str, object]]] = defaultdict(list)
    total_days = 0
    donor_days = 0

    for date, g in df.groupby("date"):
        total_days += 1
        vals = g[col].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        if np.isnan(vals).mean() > max_missing_frac:
            continue
        if np.isnan(vals).any():
            # For simplicity: reject days with any missing.
            continue

        day_type = g["day_type"].iloc[0]
        dow = int(g["dayofweek"].iloc[0])
        month = int(g["month"].iloc[0])
        key = (day_type, dow, month)
        library[key].append({"date": date, "values": vals})
        donor_days += 1

    logger.info(
        "Built donor-day library for %s: %d donor days out of %d total days.",
        col,
        donor_days,
        total_days,
    )
    if donor_days > 0:
        for key, lst in library.items():
            logger.debug(
                "  Context %s: %d donor days (example date: %s)",
                key,
                len(lst),
                lst[0]["date"],
            )

    return library


def build_daily_stats(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """
    Build daily stats for a column:
      - date
      - daily_mean
      - coverage (fraction of non-missing points in the day)
      - day_type, dayofweek, month

    Used to find the last fully-observed day for a given context.
    """
    df_tmp = df[[datetime_col, "day_type", "dayofweek", "month", col]].copy()
    df_tmp[datetime_col] = pd.to_datetime(df_tmp[datetime_col], utc=True)
    df_tmp["date"] = df_tmp[datetime_col].dt.normalize()

    grouped = df_tmp.groupby("date")

    stats = grouped[col].agg(["mean", "count"]).rename(
        columns={"mean": "daily_mean", "count": "n_obs"}
    )
    stats["n_total"] = grouped[col].size()
    # avoid division by zero
    stats["coverage"] = stats["n_obs"] / stats["n_total"].where(
        stats["n_total"] > 0, np.nan
    )

    stats["day_type"] = grouped["day_type"].first()
    stats["dayofweek"] = grouped["dayofweek"].first()
    stats["month"] = grouped["month"].first()

    stats.reset_index(inplace=True)  # 'date' becomes a column
    stats = stats.sort_values("date").reset_index(drop=True)

    logger.info(
        "Built daily stats for %s: %d days with any data (coverage>0).",
        col,
        int((stats["coverage"] > 0).sum()),
    )
    return stats


def last_context_daily_mean(
    daily_stats: pd.DataFrame,
    target_date: pd.Timestamp,
    day_type: str,
    min_coverage: float = 0.7,
) -> Optional[float]:
    """
    For a given target_date and day_type, return the daily mean of the last
    earlier day with the same day_type and sufficient coverage.

    Example: for an open_workday in a gap at the end, this finds the last
    fully-observed open_workday and uses its mean to calibrate scaling.
    """
    mask = (
        (daily_stats["date"] < target_date)
        & (daily_stats["day_type"] == day_type)
        & (daily_stats["coverage"] >= min_coverage)
    )
    if not mask.any():
        return None

    # daily_stats is sorted by date; take the last matching row
    last_row = daily_stats.loc[mask].iloc[-1]
    return float(last_row["daily_mean"])



def choose_donor_day(
    library: Dict[Tuple[str, int, int], List[Dict[str, object]]],
    day_type: str,
    dow: int,
    month: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[pd.Timestamp], Optional[str]]:
    """
    Select a donor day with closest context, in this order:
      (day_type, dow, month)
      (day_type, dow)
      (day_type)
      any

    Returns (values, date, match_level).
    """
    candidates: List[Dict[str, object]] = []

    # Exact (day_type, dow, month)
    for (dt, d, m), arrs in library.items():
        if dt == day_type and d == dow and m == month:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type+dow+month"

    # Same (day_type, dow)
    candidates = []
    for (dt, d, m), arrs in library.items():
        if dt == day_type and d == dow:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type+dow"

    # Same day_type
    candidates = []
    for (dt, d, m), arrs in library.items():
        if dt == day_type:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type"

    # Any donor
    candidates = []
    for arrs in library.values():
        candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "any"

    return None, None, None


# =========================
# 4. Gap detection utility
# =========================

def find_gaps(is_na: np.ndarray) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    n = len(is_na)
    i = 0
    while i < n:
        if is_na[i]:
            start = i
            while i + 1 < n and is_na[i + 1]:
                i += 1
            end = i
            gaps.append((start, end))
        i += 1
    return gaps


# =========================
# 5. Scale estimation
# =========================

def estimate_scale_for_gap(
    start: int,
    end: int,
    values: np.ndarray,
    is_na: np.ndarray,
    day_type_arr: np.ndarray,
    month_arr: np.ndarray,
    slot_arr: np.ndarray,
    get_baseline,
    max_context_points: int = 144,  # ~ one day of 10-min points
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> float:
    """
    Estimate a scaling factor for the gap [start, end] using many surrounding
    real points (not just the immediate neighbors).

    We look up to max_context_points points backwards and forwards, compute
    (observed / baseline) ratios, and return a clipped median.
    """
    n = len(values)
    ratios = []

    # Look backwards
    i = start - 1
    steps = 0    # type: ignore[assignment]
    while i >= 0 and steps < max_context_points:
        if not is_na[i]:
            dt = day_type_arr[i]
            m = int(month_arr[i])
            s = int(slot_arr[i])
            base = get_baseline(dt, m, s)
            if base > 0:
                ratios.append(values[i] / base)
        i -= 1
        steps += 1

    # Look forwards
    j = end + 1
    steps = 0
    while j < n and steps < max_context_points:
        if not is_na[j]:
            dt = day_type_arr[j]
            m = int(month_arr[j])
            s = int(slot_arr[j])
            base = get_baseline(dt, m, s)
            if base > 0:
                ratios.append(values[j] / base)
        j += 1
        steps += 1

    if ratios:
        scale = float(np.median(ratios))  # robust to outliers
        logger.debug(
            "Estimated scale from %d surrounding points: median=%.4f (clipped to [%.2f, %.2f])",
            len(ratios),
            scale,
            scale_min,
            scale_max,
        )
    else:
        scale = 1.0
        logger.debug("No surrounding points found for scale; defaulting to 1.0")

    scale = max(scale_min, min(scale_max, scale))
    return scale


def apply_edge_continuity(
    gap_values: np.ndarray,
    prev_value: Optional[float],
    next_value: Optional[float],
    *,
    edge_scale_min: float = EDGE_SCALE_MIN,
    edge_scale_max: float = EDGE_SCALE_MAX,
) -> np.ndarray:
    """
    Multiplicatively rescale a reconstructed gap so its two edges remain
    continuous with the nearest observed samples.

    This is mainly a guardrail for short-history reconstructions where a coarse
    fallback profile can have the right class but the wrong level (for example a
    Saturday-like weekend profile used inside a Sunday block).
    """
    adjusted = np.asarray(gap_values, dtype=float).copy()
    if adjusted.size == 0:
        return adjusted

    eps = 1e-9
    start_scale = None
    end_scale = None

    if prev_value is not None and np.isfinite(prev_value) and abs(adjusted[0]) > eps:
        start_scale = float(np.clip(prev_value / adjusted[0], edge_scale_min, edge_scale_max))

    if next_value is not None and np.isfinite(next_value) and abs(adjusted[-1]) > eps:
        end_scale = float(np.clip(next_value / adjusted[-1], edge_scale_min, edge_scale_max))

    if start_scale is None and end_scale is None:
        return adjusted
    if start_scale is None:
        start_scale = end_scale
    if end_scale is None:
        end_scale = start_scale

    ramp = np.linspace(start_scale, end_scale, adjusted.size)
    adjusted *= ramp
    adjusted[adjusted < 0] = 0.0
    return adjusted


# ================================
# 6. Imputation for a single series
# ================================

def impute_series_with_profiles(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
    small_gap_max: int = 3,     # <= 30 minutes
    large_gap_min: int = 144,   # >= 1 day at 10-min resolution
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    rel_noise: float = 0.03,    # relative noise amplitude (3% by default)
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    if rng is None:
        rng = np.random.default_rng()

    series = df[col].copy()
    values = series.to_numpy(dtype=float)
    is_na = series.isna().to_numpy()
    n = len(series)

    logger.info("Imputing column %s: %d total points, %d missing.", col, n, int(is_na.sum()))

    profiles = build_profiles(df, col)
    get_baseline = make_baseline_getter(profiles)

    if n == 0 or np.all(is_na):
        logger.warning("Column %s: empty or entirely missing; nothing to impute.", col)
        return series

    day_type_arr = df["day_type"].to_numpy()
    month_arr = df["month"].to_numpy()
    slot_arr = df["slot"].to_numpy()
    dow_arr = df["dayofweek"].to_numpy()
    date_arr = df[datetime_col].dt.normalize().to_numpy()

    # Library of full days for long gaps
    library = build_daily_library(df, col, datetime_col=datetime_col)

    # Daily stats for context-aware scaling (last similar fully observed day)
    daily_stats = build_daily_stats(df, col, datetime_col=datetime_col)

    gaps = find_gaps(is_na)
    n_gaps = len(gaps)
    gap_lengths = [end - start + 1 for start, end in gaps]
    logger.info(
        "Column %s: found %d gaps (min=%s, max=%s, median=%s).",
        col,
        n_gaps,
        min(gap_lengths) if gap_lengths else 0,
        max(gap_lengths) if gap_lengths else 0,
        np.median(gap_lengths) if gap_lengths else 0,
    )

    small_count = sum(1 for L in gap_lengths if L <= small_gap_max)
    long_count = sum(1 for L in gap_lengths if L >= large_gap_min)
    medium_count = n_gaps - small_count - long_count
    logger.info(
        "Column %s: small gaps=%d, medium gaps=%d, long gaps=%d.",
        col,
        small_count,
        medium_count,
        long_count,
    )

    for (start, end), gap_len in zip(gaps, gap_lengths):
        gap_start_time = df[datetime_col].iloc[start]
        gap_end_time = df[datetime_col].iloc[end]

        if gap_len <= small_gap_max:
            gap_type = "small"
        elif gap_len >= large_gap_min:
            gap_type = "long"
        else:
            gap_type = "medium"

        logger.debug(
            "Column %s: %s gap [%d, %d] (%d pts) from %s to %s.",
            col,
            gap_type,
            start,
            end,
            gap_len,
            gap_start_time,
            gap_end_time,
        )

        # For small gaps, we still want raw neighbor indices for interpolation
        prev_idx: Optional[int] = None
        next_idx: Optional[int] = None
        if start > 0 and not is_na[start - 1]:
            prev_idx = start - 1
        if end + 1 < n and not is_na[end + 1]:
            next_idx = end + 1

        # 1) Small gaps: linear interpolation if both neighbors exist
        if gap_len <= small_gap_max and prev_idx is not None and next_idx is not None:
            y_prev = values[prev_idx]
            y_next = values[next_idx]
            step = (y_next - y_prev) / (gap_len + 1)
            logger.debug(
                "Column %s: small gap -> linear interpolation between %.4f and %.4f.",
                col,
                y_prev,
                y_next,
            )
            for k in range(gap_len):
                v = y_prev + step * (k + 1)
                values[start + k] = max(v, 0.0)
            continue

        # 2) For medium/large gaps, estimate scale from many surrounding points
        scale = estimate_scale_for_gap(
            start=start,
            end=end,
            values=values,
            is_na=is_na,
            day_type_arr=day_type_arr,
            month_arr=month_arr,
            slot_arr=slot_arr,
            get_baseline=get_baseline,
            max_context_points=144,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        logger.info(
            "Column %s: %s gap [%s to %s], length=%d -> scale=%.4f.",
            col,
            gap_type,
            gap_start_time,
            gap_end_time,
            gap_len,
            scale,
        )

        # 3) Long gaps: donor days + noise + context-aware daily scaling
        if gap_len >= large_gap_min and len(library) > 0:
            gap_dates = np.unique(date_arr[start: end + 1])
            logger.info(
                "Column %s: long gap spans %d day(s): %s.",
                col,
                len(gap_dates),
                [str(d) for d in gap_dates],
            )

            for d in gap_dates:
                idxs_all = np.arange(start, end + 1)
                mask_day = date_arr[start: end + 1] == d
                idxs = idxs_all[mask_day]
                if len(idxs) == 0:
                    continue

                dt_i = day_type_arr[idxs[0]]
                dow_i = int(dow_arr[idxs[0]])
                m_i = int(month_arr[idxs[0]])

                donor_vals, donor_date, match_level = choose_donor_day(
                    library, dt_i, dow_i, m_i, rng
                )

                logger.info(
                    "Column %s: date %s (day_type=%s, dow=%d, month=%d) -> "
                    "donor date=%s, match_level=%s.",
                    col,
                    d,
                    dt_i,
                    dow_i,
                    m_i,
                    donor_date,
                    match_level,
                )

                # --- new: context-aware daily scaling ---
                # Use the last fully-observed day with same day_type to calibrate
                context_scale = None
                if donor_vals is not None:
                    mean_donor = float(np.mean(donor_vals))
                    # 'd' is numpy.datetime64; convert to Timestamp for clarity
                    mean_last = last_context_daily_mean(
                        daily_stats,
                        target_date=pd.Timestamp(d),
                        day_type=dt_i,
                        min_coverage=0.7,
                    )
                    if mean_last is not None and mean_donor > 0:
                        context_scale = mean_last / mean_donor
                        context_scale = max(scale_min, min(scale_max, context_scale))
                        logger.info(
                            "Column %s: context scale for date %s (day_type=%s): "
                            "mean_last=%.4f, mean_donor=%.4f, scale_context=%.4f",
                            col,
                            d,
                            dt_i,
                            mean_last,
                            mean_donor,
                            context_scale,
                        )
                    else:
                        logger.debug(
                            "Column %s: no valid context scale for date %s (day_type=%s).",
                            col,
                            d,
                            dt_i,
                        )

                if dt_i != "open_workday":
                    context_scale = None
                    
                # Blend gap-level scale and context scale (bias towards context)
                if context_scale is not None:
                    effective_scale = 0.2 * scale + 0.8 * context_scale
                else:
                    effective_scale = scale

                logger.info(
                    "Column %s: date %s -> effective_scale=%.4f (gap_scale=%.4f, context_scale=%s)",
                    col,
                    d,
                    effective_scale,
                    scale,
                    "None" if context_scale is None else f"{context_scale:.4f}",
                )

                # --- impute this day using the effective_scale ---
                for i in idxs:
                    s_i = int(slot_arr[i])
                    if donor_vals is not None and 0 <= s_i < len(donor_vals):
                        base = donor_vals[s_i]
                    else:
                        base = get_baseline(dt_i, m_i, s_i)

                    v = effective_scale * base
                    if rel_noise > 0:
                        noise = rng.normal(1.0, rel_noise)
                        noise = np.clip(noise, 1.0 - 3 * rel_noise, 1.0 + 3 * rel_noise)
                        v *= noise

                    values[i] = max(v, 0.0)

            prev_value = values[prev_idx] if prev_idx is not None else None
            next_value = values[next_idx] if next_idx is not None else None
            values[start: end + 1] = apply_edge_continuity(
                values[start: end + 1],
                prev_value=prev_value,
                next_value=next_value,
            )

            continue  # skip medium-gap branch for this gap


        # 4) Medium gaps: contextual baseline + scaling + small noise
        logger.debug(
            "Column %s: medium gap [%s to %s] -> contextual baseline + scale + noise.",
            col,
            gap_start_time,
            gap_end_time,
        )
        for i in range(start, end + 1):
            dt_i = day_type_arr[i]
            m_i = int(month_arr[i])
            s_i = int(slot_arr[i])

            base = get_baseline(dt_i, m_i, s_i)
            v = scale * base

            if rel_noise > 0:
                noise = rng.normal(1.0, rel_noise)
                noise = np.clip(noise, 1.0 - 3 * rel_noise, 1.0 + 3 * rel_noise)
                v *= noise

            values[i] = max(v, 0.0)

        prev_value = values[prev_idx] if prev_idx is not None else None
        next_value = values[next_idx] if next_idx is not None else None
        values[start: end + 1] = apply_edge_continuity(
            values[start: end + 1],
            prev_value=prev_value,
            next_value=next_value,
        )

    logger.info("Column %s: imputation completed.", col)
    return pd.Series(values, index=series.index, name=series.name)


# ==================================
# 7. Imputation for the full dataset
# ==================================

def impute_consumption_dataframe(
    df: pd.DataFrame,
    building_cols: List[str],
    datetime_col: str = "DateTime",
    small_gap_max: int = 3,
    large_gap_min: int = 144,
    rel_noise: float = 0.03,
) -> pd.DataFrame:
    """
    Impute missing consumption values for given columns.

    building_cols: e.g. ["HA", "HEI1", "HEI2", "RIZOMM", "Campus"]
    large_gap_min: threshold (in points) above which we treat a gap as "long"
                   and use donor days (default 144 -> 1 day at 10-min res).
    """
    df_ctx = prepare_context(df, datetime_col=datetime_col)
    rng = np.random.default_rng()
    result = df_ctx.copy()

    for col in building_cols:
        if col not in result.columns:
            logger.warning("Requested column %s not found in dataframe; skipping.", col)
            continue
        logger.info("=== Starting imputation for column %s ===", col)
        result[col] = impute_series_with_profiles(
            result,
            col,
            datetime_col=datetime_col,
            small_gap_max=small_gap_max,
            large_gap_min=large_gap_min,
            rel_noise=rel_noise,
            rng=rng,
        )

    return result


# ===========================
# 8. CLI / main entry point
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Contextual imputation of missing consumption data (with day-type-aware donor days) "
            "and detailed logging."
        )
    )
    parser.add_argument("input_parquet", help="Input parquet file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output parquet file path (default: <name>_imputed.parquet)",
        default=None,
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        required=True,
        help="Consumption columns to impute (e.g. HA HEI1 HEI2 RIZOMM Campus)",
    )
    parser.add_argument(
        "--datetime-col",
        default="DateTime",
        help="Name of the datetime column (default: DateTime)",
    )
    parser.add_argument(
        "--small-gap-max",
        type=int,
        default=3,
        help="Max length (in points) for linear interpolation (default: 3)",
    )
    parser.add_argument(
        "--large-gap-min",
        type=int,
        default=144,
        help="Min length (in points) to treat as long gap and use donor days (default: 144 ≈ 1 day)",
    )
    parser.add_argument(
        "--rel-noise",
        type=float,
        default=0.03,
        help="Relative noise level for variability (default: 0.03 = 3 percent)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...). Default: INFO",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. If set, logs are written both to stderr and to this file.",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("Reading input parquet: %s", args.input_parquet)
    df_in = pd.read_parquet(args.input_parquet)

    df_out = impute_consumption_dataframe(
        df_in,
        building_cols=args.cols,
        datetime_col=args.datetime_col,
        small_gap_max=args.small_gap_max,
        large_gap_min=args.large_gap_min,
        rel_noise=args.rel_noise,
    )

    if args.output is None:
        p = Path(args.input_parquet)
        out_path = p.with_name(p.stem + "_imputed" + p.suffix)
    else:
        out_path = Path(args.output)

    logger.info("Writing output parquet: %s", out_path)
    df_out.to_parquet(out_path, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()

