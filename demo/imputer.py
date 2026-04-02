"""
Contextual imputation module for the demo.

All gap sizes use a cosine-blended mix of linear interpolation (strong at
boundaries) and a contextual profile (growing in the centre).  The contextual
contribution scales with gap length so short gaps are almost purely linear
while long gaps benefit from daily-shape information.

Gap handling:
1. Medium gaps (< 144 points): contextual slot baselines, level-corrected to
   match boundary-implied mean, blended with linear interpolation.
2. Long gaps (>= 144 points / 1 day): multi-donor-day average, faded level
   correction, blended with linear interpolation.

Quality flags returned by ``impute``:
    0 = real/original value
    2 = contextual profile reconstruction
    3 = donor-day reconstruction
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LONG_GAP_MIN = 144
SCALE_MIN = 0.5
SCALE_MAX = 2.0
MAX_CONTEXT_POINTS = 144
MAX_DONOR_DAYS = 5


# ====================
# Context preparation
# ====================

def _prepare_context(df: pd.DataFrame, datetime_col: str = "DateTime") -> pd.DataFrame:
    """
    Build the same contextual columns as the reference implementation.

    The demo only supplies timestamps + one consumption series, so context fields
    that are not present in the input are derived or defaulted.
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
    return df


# =========================
# Contextual slot profiles
# =========================

def _build_profiles(df: pd.DataFrame, col: str) -> Dict[str, object]:
    non_missing = df[df[col].notna()]
    if non_missing.empty:
        return {
            "dms": pd.Series(dtype=float),
            "ds": pd.Series(dtype=float),
            "s": pd.Series(dtype=float),
            "global_mean": 0.0,
        }

    return {
        "dms": non_missing.groupby(["day_type", "month", "slot"])[col].mean(),
        "ds": non_missing.groupby(["day_type", "slot"])[col].mean(),
        "s": non_missing.groupby(["slot"])[col].mean(),
        "global_mean": float(non_missing[col].mean()),
    }


def _make_baseline_getter(profiles):
    dms = profiles["dms"]
    ds = profiles["ds"]
    s = profiles["s"]
    global_mean = profiles["global_mean"]

    def get_baseline(day_type: str, month: int, slot: int) -> float:
        val = dms.get((day_type, month, slot), np.nan)
        if not np.isnan(val):
            return float(val)

        val = ds.get((day_type, slot), np.nan)
        if not np.isnan(val):
            return float(val)

        val = s.get(slot, np.nan)
        if not np.isnan(val):
            return float(val)

        return float(global_mean)

    return get_baseline


# ===========================
# Donor-day library and stats
# ===========================

def _build_daily_library(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
    max_missing_frac: float = 0.05,
) -> Dict[Tuple[str, int, int], List[Dict[str, object]]]:
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df["date"] = df[datetime_col].dt.normalize()

    library: Dict[Tuple[str, int, int], List[Dict[str, object]]] = defaultdict(list)

    for date, g in df.groupby("date"):
        vals = g[col].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        if np.isnan(vals).mean() > max_missing_frac:
            continue
        if np.isnan(vals).any():
            continue

        day_type = g["day_type"].iloc[0]
        dow = int(g["dayofweek"].iloc[0])
        month = int(g["month"].iloc[0])
        library[(day_type, dow, month)].append({"date": date, "values": vals})

    return library


def _build_daily_stats(
    df: pd.DataFrame,
    col: str,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    df_tmp = df[[datetime_col, "day_type", "dayofweek", "month", col]].copy()
    df_tmp[datetime_col] = pd.to_datetime(df_tmp[datetime_col], utc=True)
    df_tmp["date"] = df_tmp[datetime_col].dt.normalize()
    grouped = df_tmp.groupby("date")

    stats = grouped[col].agg(["mean", "count"]).rename(
        columns={"mean": "daily_mean", "count": "n_obs"}
    )
    stats["n_total"] = grouped[col].size()
    stats["coverage"] = stats["n_obs"] / stats["n_total"].where(
        stats["n_total"] > 0, np.nan
    )
    stats["day_type"] = grouped["day_type"].first()
    stats["dayofweek"] = grouped["dayofweek"].first()
    stats["month"] = grouped["month"].first()
    stats.reset_index(inplace=True)
    return stats.sort_values("date").reset_index(drop=True)


def _last_context_daily_mean(
    daily_stats: pd.DataFrame,
    target_date: pd.Timestamp,
    day_type: str,
    min_coverage: float = 0.7,
) -> Optional[float]:
    mask = (
        (daily_stats["date"] < target_date)
        & (daily_stats["day_type"] == day_type)
        & (daily_stats["coverage"] >= min_coverage)
    )
    if not mask.any():
        return None
    return float(daily_stats.loc[mask].iloc[-1]["daily_mean"])


def _choose_donor_day(
    library: Dict[Tuple[str, int, int], List[Dict[str, object]]],
    day_type: str,
    dow: int,
    month: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[pd.Timestamp], Optional[str]]:
    candidates: List[Dict[str, object]] = []

    for (dt, d, m), arrs in library.items():
        if dt == day_type and d == dow and m == month:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type+dow+month"

    candidates = []
    for (dt, d, _m), arrs in library.items():
        if dt == day_type and d == dow:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type+dow"

    candidates = []
    for (dt, _d, _m), arrs in library.items():
        if dt == day_type:
            candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "day_type"

    candidates = []
    for arrs in library.values():
        candidates.extend(arrs)
    if candidates:
        chosen = rng.choice(candidates)
        return chosen["values"], chosen["date"], "any"

    return None, None, None


def _choose_donor_days(
    library: Dict[Tuple[str, int, int], List[Dict[str, object]]],
    day_type: str,
    dow: int,
    month: int,
    max_donors: int = MAX_DONOR_DAYS,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Select up to *max_donors* similar days and return their averaged values.

    Uses the same fallback hierarchy as ``_choose_donor_day`` but averages
    multiple donors (most-recent first) to reduce variance.
    """
    levels = [
        ("day_type+dow+month", lambda dt, d, m: dt == day_type and d == dow and m == month),
        ("day_type+dow", lambda dt, d, m: dt == day_type and d == dow),
        ("day_type", lambda dt, d, m: dt == day_type),
        ("any", lambda dt, d, m: True),
    ]
    for level_name, match_fn in levels:
        candidates: List[Dict[str, object]] = []
        for (dt, d, m), arrs in library.items():
            if match_fn(dt, d, m):
                candidates.extend(arrs)
        if not candidates:
            continue
        candidates.sort(key=lambda c: c["date"], reverse=True)
        selected = candidates[: max_donors]
        min_len = min(len(c["values"]) for c in selected)
        avg_vals = np.mean([c["values"][:min_len] for c in selected], axis=0)
        return avg_vals, level_name

    return None, None


# =====================
# Gap detection / scale
# =====================

def _find_gaps(is_na: np.ndarray) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    n = len(is_na)
    i = 0
    while i < n:
        if is_na[i]:
            start = i
            while i + 1 < n and is_na[i + 1]:
                i += 1
            gaps.append((start, i))
        i += 1
    return gaps


def _estimate_scale_for_gap(
    start: int,
    end: int,
    values: np.ndarray,
    is_na: np.ndarray,
    day_type_arr: np.ndarray,
    month_arr: np.ndarray,
    slot_arr: np.ndarray,
    get_baseline,
    max_context_points: int = MAX_CONTEXT_POINTS,
    scale_min: float = SCALE_MIN,
    scale_max: float = SCALE_MAX,
) -> float:
    n = len(values)
    ratios = []

    i = start - 1
    steps = 0
    while i >= 0 and steps < max_context_points:
        if not is_na[i]:
            base = get_baseline(day_type_arr[i], int(month_arr[i]), int(slot_arr[i]))
            if base > 0:
                ratios.append(values[i] / base)
        i -= 1
        steps += 1

    j = end + 1
    steps = 0
    while j < n and steps < max_context_points:
        if not is_na[j]:
            base = get_baseline(day_type_arr[j], int(month_arr[j]), int(slot_arr[j]))
            if base > 0:
                ratios.append(values[j] / base)
        j += 1
        steps += 1

    scale = float(np.median(ratios)) if ratios else 1.0
    return max(scale_min, min(scale_max, scale))


def _apply_edge_continuity(
    gap_values: np.ndarray,
    prev_value: Optional[float],
    next_value: Optional[float],
    *,
    edge_scale_min: float = 0.1,
    edge_scale_max: float = 10.0,
) -> np.ndarray:
    """Deprecated -- kept for backward compatibility.  Replaced by
    ``_blend_linear_contextual`` which achieves boundary continuity without
    the destructive multiplicative ramp."""
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


def _blend_linear_contextual(
    contextual: np.ndarray,
    prev_value: Optional[float],
    next_value: Optional[float],
) -> np.ndarray:
    """Blend a contextual profile with linear interpolation at gap boundaries.

    Uses fixed-width cosine transitions at each edge (up to 18 points / 3 h)
    so the gap center stays at full contextual strength while boundaries
    match the observed values exactly.

    For one-sided gaps an exponential decay anchors the known boundary.
    """
    gap_len = len(contextual)
    if gap_len == 0:
        return contextual.copy()

    result = contextual.copy()

    if prev_value is not None and next_value is not None:
        linear = np.linspace(prev_value, next_value, gap_len + 2)[1:-1]

        # Global cosine blend: 1 at edges, 0 at center.
        t = np.arange(1, gap_len + 1, dtype=float) / (gap_len + 1)
        w_cos = np.cos(np.pi * t) ** 2
        # Max contextual contribution scales with gap length (tau = 3 days).
        ctx_max = 1.0 - np.exp(-gap_len / 432.0)
        w_linear = 1.0 - ctx_max * (1.0 - w_cos)
        result = w_linear * linear + (1.0 - w_linear) * contextual
    elif prev_value is not None:
        tau = max(gap_len / 4.0, 1.0)
        k = np.arange(gap_len, dtype=float)
        w = np.exp(-k / tau)
        result = w * prev_value + (1.0 - w) * contextual
    elif next_value is not None:
        tau = max(gap_len / 4.0, 1.0)
        k = np.arange(gap_len - 1, -1, -1, dtype=float)
        w = np.exp(-k / tau)
        result = w * next_value + (1.0 - w) * contextual

    result[result < 0] = 0.0
    return result


# ========================
# Naive imputation
# ========================

def naive_impute(series: pd.Series, method: str = "linear"):
    """
    Naive gap filling for baseline comparison.

    Parameters
    ----------
    series : pd.Series
        Consumption series with NaN gaps.
    method : str
        'linear' for linear interpolation, 'zero' for zero-padding.

    Returns
    -------
    (pd.Series, pd.Series)
        Imputed series and quality flags (1 = imputed, 0 = original).
    """
    is_na = series.isna()
    imputed = series.copy()

    if method == "zero":
        imputed = imputed.fillna(0.0)
    else:
        imputed = imputed.interpolate(method="linear")
        imputed = imputed.ffill().bfill()

    quality = pd.Series(
        np.where(is_na, 1, 0),
        index=series.index,
        dtype=int,
    )
    return imputed, quality


# ========================
# Public demo entry point
# ========================

def impute(
    series: pd.Series,
    date_index,
    quality: Optional[pd.Series] = None,
    *,
    large_gap_min: int = LONG_GAP_MIN,
    random_seed: Optional[int] = None,
    **_kwargs,
):
    """
    Impute one consumption series with the reference contextual algorithm.

    Parameters
    ----------
    series : pd.Series
        Consumption series containing missing values.
    date_index : array-like / Series / DatetimeIndex
        Timestamps aligned with ``series``.
    quality : pd.Series, optional
        Optional quality flags. Real points stay at 0. Imputed points are marked
        according to the algorithm branch used.

    Returns
    -------
    (pd.Series, pd.Series)
        Imputed series and integer quality flags.
    """
    values_series = pd.Series(series).copy().reset_index(drop=True)
    dt_series = pd.Series(pd.to_datetime(date_index, utc=True)).reset_index(drop=True)
    if len(values_series) != len(dt_series):
        raise ValueError("series and date_index must have the same length")

    original_index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(values_series))
    values_series.index = original_index

    if quality is None:
        quality_out = pd.Series(
            np.where(values_series.isna(), np.nan, 0),
            index=original_index,
            dtype=float,
        )
    else:
        quality_out = pd.Series(quality, index=original_index, dtype=float).copy()
        quality_out.loc[values_series.notna()] = 0

    # Preserve the original NaN mask for quality-flagging decisions.
    initial_is_na = values_series.isna().to_numpy()

    df = pd.DataFrame({"DateTime": dt_series, "value": values_series.to_numpy()})
    df = _prepare_context(df, datetime_col="DateTime")

    working = df["value"].copy()
    values = working.to_numpy(dtype=float)
    is_na = working.isna().to_numpy()

    if len(working) == 0:
        return values_series.copy(), quality_out.fillna(0).astype(int)

    if np.all(is_na):
        # No information exists to reconstruct from. Keep values as-is and mark
        # them as unresolved if needed; downstream code can decide how to handle it.
        return values_series.copy(), quality_out.fillna(0).astype(int)

    rng = np.random.default_rng(random_seed)
    profiles = _build_profiles(df, "value")
    get_baseline = _make_baseline_getter(profiles)

    day_type_arr = df["day_type"].to_numpy()
    month_arr = df["month"].to_numpy()
    slot_arr = df["slot"].to_numpy()
    dow_arr = df["dayofweek"].to_numpy()
    date_arr = df["DateTime"].dt.normalize().to_numpy()

    library = _build_daily_library(df, "value", datetime_col="DateTime")
    daily_stats = _build_daily_stats(df, "value", datetime_col="DateTime")

    gaps = _find_gaps(is_na)

    for start, end in gaps:
        gap_len = end - start + 1

        prev_idx: Optional[int] = None
        next_idx: Optional[int] = None
        if start > 0 and not is_na[start - 1]:
            prev_idx = start - 1
        if end + 1 < len(values) and not is_na[end + 1]:
            next_idx = end + 1

        prev_value = values[prev_idx] if prev_idx is not None else None
        next_value = values[next_idx] if next_idx is not None else None

        # Scale estimation from surrounding context.
        scale = _estimate_scale_for_gap(
            start=start,
            end=end,
            values=values,
            is_na=is_na,
            day_type_arr=day_type_arr,
            month_arr=month_arr,
            slot_arr=slot_arr,
            get_baseline=get_baseline,
            max_context_points=MAX_CONTEXT_POINTS,
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
        )

        # Build raw contextual profile for the gap.
        contextual = np.zeros(gap_len)

        if gap_len >= large_gap_min and len(library) > 0:
            # --- Long gaps: multi-donor-day reconstruction ---
            gap_dates = np.unique(date_arr[start : end + 1])

            for d in gap_dates:
                idxs_all = np.arange(start, end + 1)
                mask_day = date_arr[start : end + 1] == d
                idxs = idxs_all[mask_day]
                if len(idxs) == 0:
                    continue

                dt_i = day_type_arr[idxs[0]]
                dow_i = int(dow_arr[idxs[0]])
                m_i = int(month_arr[idxs[0]])

                donor_vals, _match_level = _choose_donor_days(
                    library, dt_i, dow_i, m_i
                )

                context_scale = None
                if donor_vals is not None:
                    mean_donor = float(np.mean(donor_vals))
                    mean_last = _last_context_daily_mean(
                        daily_stats,
                        target_date=pd.Timestamp(d),
                        day_type=dt_i,
                        min_coverage=0.7,
                    )
                    if mean_last is not None and mean_donor > 0:
                        context_scale = mean_last / mean_donor
                        context_scale = max(SCALE_MIN, min(SCALE_MAX, context_scale))

                effective_scale = (
                    0.3 * scale + 0.7 * context_scale
                    if context_scale is not None
                    else scale
                )

                for i in idxs:
                    s_i = int(slot_arr[i])
                    if donor_vals is not None and 0 <= s_i < len(donor_vals):
                        base = donor_vals[s_i]
                    else:
                        base = get_baseline(dt_i, m_i, s_i)
                    contextual[i - start] = max(effective_scale * base, 0.0)

            # Faded level correction for donor path: full at 1 day, decays
            # for longer gaps where boundary mean is unreliable.
            if prev_value is not None and next_value is not None:
                ctx_mean = float(np.mean(contextual))
                lin_mean = (prev_value + next_value) / 2.0
                if ctx_mean > 1e-9:
                    level_trust = np.exp(
                        -(gap_len - LONG_GAP_MIN) / 72.0
                    ) if gap_len > LONG_GAP_MIN else 1.0
                    raw_scale = np.clip(lin_mean / ctx_mean, 0.5, 2.0)
                    lsc = 1.0 + level_trust * (raw_scale - 1.0)
                    contextual *= lsc

            flag = 3
        else:
            # --- Medium / short gaps: contextual baseline ---
            for i in range(start, end + 1):
                dt_i = day_type_arr[i]
                m_i = int(month_arr[i])
                s_i = int(slot_arr[i])
                base = get_baseline(dt_i, m_i, s_i)
                contextual[i - start] = max(scale * base, 0.0)

            # Level-correct: match contextual mean to boundary-implied mean.
            if prev_value is not None and next_value is not None:
                ctx_mean = float(np.mean(contextual))
                lin_mean = (prev_value + next_value) / 2.0
                if ctx_mean > 1e-9:
                    lsc = np.clip(lin_mean / ctx_mean, 0.5, 2.0)
                    contextual *= lsc

            flag = 2

        # Blend contextual profile with linear interpolation at boundaries.
        blended = _blend_linear_contextual(contextual, prev_value, next_value)
        values[start : end + 1] = blended
        quality_out.iloc[start : end + 1] = flag

    result = pd.Series(values, index=original_index, name=getattr(series, "name", None))

    # Robust final fallback for edge cases where profiles/global mean were not enough.
    if result.isna().any():
        fill_value = float(values_series.dropna().mean()) if values_series.notna().any() else 0.0
        unresolved = result.isna()
        result = result.ffill().bfill().fillna(fill_value)
        quality_out.loc[unresolved & quality_out.isna()] = 2

    # Keep original non-missing points flagged as real data.
    quality_out.loc[~initial_is_na] = 0
    quality_out = quality_out.fillna(0).astype(int)

    return result, quality_out
