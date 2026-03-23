"""
Cascading imputation module (Module de Pilotage -- core logic).

Fills gaps in a consumption time series using progressively heavier methods:
1. Linear interpolation for short gaps (<= 3 points / 30 min)
2. Seasonal-weighted interpolation for medium gaps (4-36 points / 30 min - 6 h)
3. Backup data injection for long gaps (> 36 points / 6 h)
4. Forward-fill / backward-fill for any remaining edge NaNs
"""
import numpy as np
import pandas as pd

from config import POINTS_PER_DAY, FREQ


def _gap_runs(series):
    """
    Identify contiguous runs of NaN in a Series.

    Returns list of (start_idx, length) tuples.
    """
    is_nan = series.isna().values
    runs = []
    i = 0
    while i < len(is_nan):
        if is_nan[i]:
            start = i
            while i < len(is_nan) and is_nan[i]:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


def _build_seasonal_backup(series, date_index):
    """
    Build a day-of-week + time-of-day seasonal profile from non-NaN data.

    Returns a DataFrame indexed by (weekday, minute_of_day) with the median
    consumption value.
    """
    df = pd.DataFrame({"value": series.values, "date": date_index})
    df["weekday"] = df["date"].dt.weekday
    df["minute"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    backup = df.dropna(subset=["value"]).groupby(["weekday", "minute"])["value"].median()
    return backup


def impute(series, date_index, quality=None):
    """
    Apply cascading imputation to a consumption time series.

    Parameters
    ----------
    series : pd.Series
        Consumption values (may contain NaN for gaps).
    date_index : pd.DatetimeIndex
        Corresponding timestamps (tz-aware or naive, must be regular 10-min).
    quality : pd.Series, optional
        Quality flags. Updated in place: 0=real, 1=linear, 2=seasonal, 3=backup.

    Returns
    -------
    pd.Series
        The imputed series (no NaN remaining).
    pd.Series
        Quality flags.
    """
    result = series.copy()

    if quality is None:
        quality = pd.Series(np.where(series.isna(), np.nan, 0), index=series.index)

    # Build seasonal backup from available real data
    backup = _build_seasonal_backup(result, date_index)

    runs = _gap_runs(result)

    for start, length in runs:
        if length <= 3:
            # Short gap: linear interpolation
            # Mark a small window around the gap and interpolate
            lo = max(0, start - 1)
            hi = min(len(result), start + length + 1)
            segment = result.iloc[lo:hi].copy()
            segment = segment.interpolate(method="linear")
            result.iloc[lo:hi] = segment
            quality.iloc[start:start + length] = 1

        elif length <= 36:
            # Medium gap: seasonal-weighted interpolation
            # Blend linear interpolation with seasonal backup
            lo = max(0, start - 1)
            hi = min(len(result), start + length + 1)
            segment = result.iloc[lo:hi].copy()
            linear = segment.interpolate(method="linear")

            for j in range(start, start + length):
                ts = date_index[j]
                key = (ts.weekday(), ts.hour * 60 + ts.minute)
                seasonal_val = backup.get(key, np.nan)
                linear_val = linear.iloc[j - lo]
                if not np.isnan(seasonal_val) and not np.isnan(linear_val):
                    # Weight: more seasonal as gap gets longer
                    weight = length / 36.0  # 0..1
                    result.iloc[j] = (1 - weight) * linear_val + weight * seasonal_val
                elif not np.isnan(seasonal_val):
                    result.iloc[j] = seasonal_val
                elif not np.isnan(linear_val):
                    result.iloc[j] = linear_val
            quality.iloc[start:start + length] = 2

        else:
            # Long gap: backup data injection
            for j in range(start, start + length):
                ts = date_index[j]
                key = (ts.weekday(), ts.hour * 60 + ts.minute)
                val = backup.get(key, np.nan)
                if not np.isnan(val):
                    result.iloc[j] = val
            quality.iloc[start:start + length] = 3

    # Final pass: ffill/bfill for any remaining NaN at edges
    result = result.ffill().bfill()
    quality = quality.fillna(0).astype(int)

    return result, quality
