"""Adapter wrapping TemperatureAwareHybridEngine for the single-building demo."""
from __future__ import annotations

import contextlib
import io
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BUILDING_COLUMN,
    HISTORICAL_CSV,
    OUTPUT_DIR,
    RECENT_HA_CSV,
    TIMEZONE,
    WEATHER_CSV,
)
from hybrid_engine import TemperatureAwareHybridEngine


def _to_window_time(ts: pd.Series) -> pd.Series:
    # Mirror run_demo.extract_window: UTC -> Paris -> tz-naive. Skipping this
    # offsets the historical extension relative to the input window.
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


_ENGINE: Optional[TemperatureAwareHybridEngine] = None
_WEATHER: Optional[pd.DataFrame] = None
_HISTORY_CACHE: Optional[pd.DataFrame] = None

# 8 weeks = the longest lookback the engine's ensemble methods use. Less
# than this and the long-gap ensemble degenerates and the engine collapses
# the gap to a flat constant via its >40% validation fallback.
_PREPEND_DAYS = 56

_STRATEGY_FLAG_MAP = {
    "L1_LINEAR_MICRO": 2,
    "L2_LINEAR_TEMP": 2,
    "L3_SPLINE": 2,
    "L3_PILOTAGE_FALLBACK": 2,
    "L4_ANOMALY_GUARDED": 2,
    "L4_TEMP_AWARE": 2,
    "L5_DAY_SCALE_PILOTAGE": 3,
    "L6_MULTI_DAY_ENSEMBLE": 3,
    "L7_VERY_LONG_ENSEMBLE": 3,
}


def _flag_for_strategy(strategy: str) -> int:
    if "_FALLBACK_MEDIAN_" in strategy:
        return 3
    return _STRATEGY_FLAG_MAP.get(strategy, 2)


def _load_combined_ha_history() -> pd.DataFrame:
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None:
        return _HISTORY_CACHE

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
        _HISTORY_CACHE = pd.DataFrame(columns=["Timestamp", "Ptot_HA"])
        return _HISTORY_CACHE

    combined = pd.concat(pieces, ignore_index=True)
    combined["Timestamp"] = _to_window_time(combined["Timestamp"])
    combined = (
        combined.dropna(subset=["Timestamp"])
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    _HISTORY_CACHE = combined
    return combined


def _extend_with_history(df_in: pd.DataFrame) -> pd.DataFrame:
    # See _PREPEND_DAYS comment above for why we need 56 days of context.
    hist = _load_combined_ha_history()
    if hist.empty:
        return df_in

    window_start = df_in["Timestamp"].iloc[0]
    extension_start = window_start - pd.Timedelta(days=_PREPEND_DAYS)

    mask = (hist["Timestamp"] >= extension_start) & (hist["Timestamp"] < window_start)
    extension = hist.loc[mask, ["Timestamp", "Ptot_HA"]].copy()
    if extension.empty:
        return df_in

    if BUILDING_COLUMN != "Ptot_HA":
        extension = extension.rename(columns={"Ptot_HA": BUILDING_COLUMN})

    extended = pd.concat([extension, df_in], ignore_index=True, sort=False)
    extended = (
        extended.drop_duplicates(subset=["Timestamp"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return extended


def _get_engine() -> TemperatureAwareHybridEngine:
    global _ENGINE
    if _ENGINE is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cache_path = os.path.join(OUTPUT_DIR, "hybrid_templates_cache.pkl")
        with contextlib.redirect_stdout(io.StringIO()):
            _ENGINE = TemperatureAwareHybridEngine(
                site_cols=[BUILDING_COLUMN],
                weather_df=None,
                use_historical_data=False,  # skip the missing 2021-2025 path
                template_cache_file=cache_path,
            )
        hist = _load_combined_ha_history()
        if not hist.empty:
            _ENGINE.use_historical_data = True
            _ENGINE.historical_df = hist
    return _ENGINE


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
    **_kwargs,
):
    """Route the demo single-series API through TemperatureAwareHybridEngine."""
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
        {"Timestamp": dt_naive, BUILDING_COLUMN: s.to_numpy(dtype=float)}
    )

    df_in = _extend_with_history(df_window)
    window_start_ts = pd.Timestamp(dt_naive.iloc[0])
    # Engine reindexes to a 10-min grid floored on min(Timestamp); use the
    # floored extension start (not the row count) to map strategy-log indices
    # back into the window.
    extension_start_ts = df_in["Timestamp"].min().floor("10min")
    extension_len = int(
        (window_start_ts - extension_start_ts) / pd.Timedelta(minutes=10)
    )

    engine = _get_engine()
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
    imputed_vals = out_df[BUILDING_COLUMN].to_numpy(dtype=float)

    if np.isnan(imputed_vals).any():
        imputed_vals = (
            pd.Series(imputed_vals)
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .to_numpy()
        )

    flags = np.zeros(len(s), dtype=int)
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
    unlogged = initial_na & (flags == 0)
    flags[unlogged] = 2
    flags[~initial_na] = 0

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
