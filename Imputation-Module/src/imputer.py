"""Single-series adapter around ExtendedDeploymentAlgorithm."""
import contextlib
import io
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    AUDIT_LOG_DIR,
    BUILDING_COLUMN,
    HISTORICAL_CSV,
    OUTPUT_DIR,
    RECENT_HA_CSV,
    TIMEZONE,
    WEATHER_CSV,
)
from smart_imputation import ExtendedDeploymentAlgorithm


def _to_window_time(ts: pd.Series) -> pd.Series:
    # Same tz handling as window.extract_window: UTC -> Paris -> naive.
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


_WEATHER: Optional[pd.DataFrame] = None
_HISTORY_CACHE: Dict[str, pd.DataFrame] = {}
_LAST_STRATEGY_LOG: list = []
_LAST_EXTENSION_LEN: int = 0


def get_last_strategy_log() -> list:
    """strategy_log from the most recent impute() call, with indices already
    shifted back to the caller's series."""
    return list(_LAST_STRATEGY_LOG)


def set_history_source(df: Optional[pd.DataFrame], building_column: str) -> None:
    """Let the Cassandra CLI path inject a pre-loaded history. Pass None to reset."""
    if df is None or df.empty:
        _HISTORY_CACHE[building_column] = pd.DataFrame(
            columns=["Timestamp", building_column]
        )
        return
    cleaned = df[["Timestamp", building_column]].copy()
    cleaned["Timestamp"] = _to_window_time(cleaned["Timestamp"])
    cleaned = (
        cleaned.dropna(subset=["Timestamp"])
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    _HISTORY_CACHE[building_column] = cleaned


# 8 weeks: the long-gap ensemble needs this much context before its
# >40%-missing fallback stops kicking in.
_PREPEND_DAYS = 56

# Internal strategy names -> CLI quality flags.
# 1 = linear, 2 = contextual, 3 = donor-day / ML-derived.
_STRATEGY_FLAG_MAP = {
    "LINEAR_MICRO": 1,
    "LINEAR_SHORT": 1,
    "THERMAL_TEMPLATE": 2,
    "ENHANCED_TEMPLATE": 2,
    "WEEKEND_TEMPLATE": 2,
    "WEEKEND_TEMPLATE_SATURDAY": 2,
    "WEEKEND_TEMPLATE_SUNDAY": 2,
    "WEEKEND_TEMPLATE_MIXED": 2,
    "SAFE_LINEAR_MEDIAN": 2,
    "PEER_CORRELATION": 2,
    "HOURLY_MEDIAN_FALLBACK": 2,
    "MULTI_WEEK_TEMPLATE": 3,
    "MICE": 3,
    "KNN_CONTEXT": 3,
    "KALMAN_FILTER": 3,
    "SAFE_MEDIAN": 3,
}


def _flag_for_strategy(strategy: str) -> int:
    return _STRATEGY_FLAG_MAP.get(strategy, 2)


def _load_combined_history(building_column: str) -> pd.DataFrame:
    cached = _HISTORY_CACHE.get(building_column)
    if cached is not None:
        return cached

    # Only Ptot_HA has CSVs; the other buildings need the Cassandra history seed.
    if building_column != "Ptot_HA":
        empty = pd.DataFrame(columns=["Timestamp", building_column])
        _HISTORY_CACHE[building_column] = empty
        return empty

    pieces = []
    for path in (HISTORICAL_CSV, RECENT_HA_CSV):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
        except Exception:
            continue
        if "Ptot_HA" not in df.columns:
            continue
        pieces.append(
            df[["Date", "Ptot_HA"]].rename(columns={"Date": "Timestamp"})
        )
    if not pieces:
        empty = pd.DataFrame(columns=["Timestamp", "Ptot_HA"])
        _HISTORY_CACHE["Ptot_HA"] = empty
        return empty

    combined = pd.concat(pieces, ignore_index=True)
    combined["Timestamp"] = _to_window_time(combined["Timestamp"])
    combined = (
        combined.dropna(subset=["Timestamp"])
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    _HISTORY_CACHE["Ptot_HA"] = combined
    return combined


def _extend_with_history(df_in: pd.DataFrame, building_column: str) -> pd.DataFrame:
    hist = _load_combined_history(building_column)
    if hist.empty:
        return df_in

    window_start = df_in["Timestamp"].iloc[0]
    extension_start = window_start - pd.Timedelta(days=_PREPEND_DAYS)

    mask = (hist["Timestamp"] >= extension_start) & (hist["Timestamp"] < window_start)
    extension = hist.loc[mask, ["Timestamp", building_column]].copy()
    if extension.empty:
        return df_in

    extended = pd.concat([extension, df_in], ignore_index=True, sort=False)
    return (
        extended.drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )


def _get_weather_df() -> pd.DataFrame:
    global _WEATHER
    if _WEATHER is None:
        try:
            w = pd.read_csv(WEATHER_CSV, parse_dates=["Date"])
            w["Date"] = _to_window_time(w["Date"])
            _WEATHER = w.rename(columns={"Date": "Timestamp"})[
                ["Timestamp", "AirTemp"]
            ]
        except Exception:
            _WEATHER = pd.DataFrame(columns=["Timestamp", "AirTemp"])
    return _WEATHER


def impute(
    series: pd.Series,
    date_index,
    quality: Optional[pd.Series] = None,
    *,
    large_gap_min: int = 144,
    random_seed: Optional[int] = None,
    building_column: Optional[str] = None,
    **_kwargs,
):
    """Impute a single series. Routes through ExtendedDeploymentAlgorithm."""
    bcol = building_column or BUILDING_COLUMN
    s = pd.Series(series).reset_index(drop=True)
    if isinstance(date_index, pd.DatetimeIndex):
        dt = pd.Series(date_index)
    else:
        dt = pd.Series(pd.to_datetime(date_index))
    dt = dt.reset_index(drop=True)

    if dt.dt.tz is not None:
        dt_naive = dt.dt.tz_convert(None)
    else:
        dt_naive = dt

    initial_na = s.isna().to_numpy()
    df_window = pd.DataFrame(
        {"Timestamp": dt_naive, bcol: s.to_numpy(dtype=float)}
    )

    df_in = _extend_with_history(df_window, bcol)
    window_start_ts = pd.Timestamp(dt_naive.iloc[0])
    # The algorithm reindexes to a 10-min grid floored on min(Timestamp), so
    # the strategy-log offset has to use the floored value too.
    extension_start_ts = df_in["Timestamp"].min().floor("10min")
    extension_len = int(
        (window_start_ts - extension_start_ts) / pd.Timedelta(minutes=10)
    )

    if random_seed is not None:
        np.random.seed(random_seed)

    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    algo = ExtendedDeploymentAlgorithm(
        site_cols=[bcol],
        use_knn=False,
        use_mice=True,
        use_kalman=True,
        use_multi_week_templates=True,
        use_chunked_recovery=True,
        template_lookback_days=28,
        audit_log_dir=AUDIT_LOG_DIR,
        timezone=TIMEZONE,
    )

    weather = _get_weather_df()
    weather_arg = weather if not weather.empty else None

    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        out_df = algo.impute(df_in, weather_df=weather_arg)
    strategy_log = list(algo.strategy_log)

    if getattr(out_df["Timestamp"].dt, "tz", None) is not None:
        out_df["Timestamp"] = out_df["Timestamp"].dt.tz_localize(None)

    out_df = (
        out_df.set_index("Timestamp")
        .reindex(pd.to_datetime(dt_naive.values))
        .reset_index()
    )
    imputed_vals = out_df[bcol].to_numpy(dtype=float)

    if np.isnan(imputed_vals).any():
        imputed_vals = (
            pd.Series(imputed_vals)
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .to_numpy()
        )

    flags = np.zeros(len(s), dtype=int)
    aligned_log = []
    for entry in strategy_log:
        gs_ext = int(entry["gap_start"])
        ge_ext = int(entry["gap_end"])
        gs = max(0, gs_ext - extension_len)
        ge = max(0, ge_ext - extension_len)
        gs = min(gs, len(flags))
        ge = min(ge, len(flags))
        if ge <= gs:
            # gap fell entirely inside the prepended history
            continue
        flags[gs:ge] = _flag_for_strategy(str(entry["strategy"]))
        aligned_log.append({**entry, "gap_start": gs, "gap_end": ge})

    unlogged = initial_na & (flags == 0)
    flags[unlogged] = 2
    flags[~initial_na] = 0

    global _LAST_STRATEGY_LOG, _LAST_EXTENSION_LEN
    _LAST_STRATEGY_LOG = aligned_log
    _LAST_EXTENSION_LEN = extension_len

    original_index = (
        series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(s))
    )
    out_series = pd.Series(
        imputed_vals, index=original_index, name=getattr(series, "name", None)
    )
    out_quality = pd.Series(flags, index=original_index, dtype=int)
    return out_series, out_quality


def naive_impute(series: pd.Series, method: str = "linear"):
    """Zero-fill baseline used as a comparison point."""
    is_na = series.isna()
    imputed = series.copy().fillna(0.0)
    quality = pd.Series(np.where(is_na, 1, 0), index=series.index, dtype=int)
    return imputed, quality
