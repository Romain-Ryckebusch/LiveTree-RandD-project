"""Adapter wrapping TemperatureAwareHybridEngine for the pilotage module.

Caller passes ``building_column`` through set_history_source and impute;
engine and history caches are keyed by that column name.
"""
import contextlib
import io
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    BUILDING_COLUMN,
    HISTORICAL_CSV,
    LOW_VARIANCE_AUTO_FRACTION,
    LOW_VARIANCE_FLOOR_W,
    OUTPUT_DIR,
    RECENT_HA_CSV,
    TIMEZONE,
    WEATHER_CSV,
)
from hybrid_engine import TemperatureAwareHybridEngine


def _to_window_time(ts: pd.Series) -> pd.Series:
    # Match window.extract_window's tz handling (UTC -> Paris -> naive).
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


_ENGINE: Dict[str, TemperatureAwareHybridEngine] = {}
_WEATHER: Optional[pd.DataFrame] = None
_HISTORY_CACHE: Dict[str, pd.DataFrame] = {}
_LAST_STRATEGY_LOG: list = []
_LAST_EXTENSION_LEN: int = 0


def get_last_strategy_log() -> list:
    """Return strategy_log (with postproc decisions) from the most recent impute() call.

    Indices in the entries are offset to align with the caller's input series
    (history extension already subtracted), and out-of-window gaps are dropped.
    """
    return list(_LAST_STRATEGY_LOG)


def set_history_source(df: Optional[pd.DataFrame], building_column: str) -> None:
    """Inject a pre-loaded history for ``building_column`` instead of reading CSVs.

    Used by the Cassandra CLI path. Pass None or an empty frame to reset.
    """
    if df is None or df.empty:
        _HISTORY_CACHE[building_column] = pd.DataFrame(
            columns=["Timestamp", building_column]
        )
    else:
        cleaned = df[["Timestamp", building_column]].copy()
        cleaned["Timestamp"] = _to_window_time(cleaned["Timestamp"])
        cleaned = (
            cleaned.dropna(subset=["Timestamp"])
            .drop_duplicates(subset=["Timestamp"], keep="last")
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
        _HISTORY_CACHE[building_column] = cleaned
    # Drop cached engine so the new history is picked up.
    _ENGINE.pop(building_column, None)

# 8 weeks: minimum context the long-gap ensemble needs before its >40% fallback kicks in.
_PREPEND_DAYS = 56

# Strategy codes emitted by TemperatureAwareHybridEngine (hybrid_engine.py).
# Quality flags: 1=linear, 2=contextual, 3=donor-day.
_STRATEGY_FLAG_MAP = {
    "LINEAR_MICRO": 1,
    "LINEAR_SHORT": 1,
    "THERMAL_TEMPLATE": 2,
    "ENHANCED_TEMPLATE": 2,
    "WEEKEND_TEMPLATE_SATURDAY": 2,
    "WEEKEND_TEMPLATE_SUNDAY": 2,
    "WEEKEND_TEMPLATE_MIXED": 2,
    "SAFE_LINEAR_MEDIAN": 2,
    "PEER_CORRELATION": 2,
    "MULTI_WEEK_TEMPLATE": 3,
    "SAFE_MEDIAN": 3,
}


def _flag_for_strategy(strategy: str) -> int:
    return _STRATEGY_FLAG_MAP.get(strategy, 2)


def _load_combined_history(building_column: str) -> pd.DataFrame:
    cached = _HISTORY_CACHE.get(building_column)
    if cached is not None:
        return cached

    # CSV fallback only has Ptot_HA. Other buildings must be seeded from Cassandra.
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
    extended = (
        extended.drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return extended


def _auto_low_variance_threshold(hist: pd.DataFrame, building_column: str) -> float:
    if hist.empty or building_column not in hist.columns:
        return 5000.0
    values = pd.to_numeric(hist[building_column], errors="coerce").to_numpy()
    median_abs = float(np.nanmedian(np.abs(values)))
    if not np.isfinite(median_abs) or median_abs <= 0:
        return 5000.0
    return max(LOW_VARIANCE_AUTO_FRACTION * median_abs, LOW_VARIANCE_FLOOR_W)


def _get_engine(building_column: str) -> TemperatureAwareHybridEngine:
    if building_column in _ENGINE:
        return _ENGINE[building_column]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(
        OUTPUT_DIR, f"hybrid_templates_cache_{building_column}.pkl"
    )
    hist = _load_combined_history(building_column)
    low_var = _auto_low_variance_threshold(hist, building_column)
    with contextlib.redirect_stdout(io.StringIO()):
        engine = TemperatureAwareHybridEngine(
            site_cols=[building_column],
            weather_df=None,
            use_historical_data=False,  # skip the missing 2021-2025 path
            template_cache_file=cache_path,
            low_variance_threshold=low_var,
        )
    if not hist.empty:
        engine.use_historical_data = True
        engine.historical_df = hist
    _ENGINE[building_column] = engine
    return engine


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
    """Single-series entry point routed through TemperatureAwareHybridEngine."""
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
    # Engine reindexes to a 10-min grid floored on min(Timestamp).
    # Use the floored start for strategy-log offset alignment.
    extension_start_ts = df_in["Timestamp"].min().floor("10min")
    extension_len = int(
        (window_start_ts - extension_start_ts) / pd.Timedelta(minutes=10)
    )

    engine = _get_engine(bcol)
    weather = _get_weather_df()
    weather_arg = weather if not weather.empty else None

    with contextlib.redirect_stdout(io.StringIO()):
        out_df = engine.impute(df_in, weather_df=weather_arg)
    strategy_log = list(engine.strategy_log)

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
            # gap fell entirely within the prepended history
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
    imputed = series.copy()
    imputed = imputed.fillna(0.0)

    quality = pd.Series(np.where(is_na, 1, 0), index=series.index, dtype=int)
    return imputed, quality
