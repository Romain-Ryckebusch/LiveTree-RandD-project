"""7-day window extraction for the imputer.

Data-source agnostic: the caller passes DataFrames loaded from either CSV
or Cassandra. The function reindexes history onto a complete 10-minute
grid so that sparse Cassandra rows (missing timestamps have no row at
all) become NaN values the imputer can detect, just like CSV data.
"""
import numpy as np
import pandas as pd
import pytz

from config import (
    BUILDING_COLUMN,
    LOOKBACK_POINTS,
    POINTS_PER_DAY,
    TIMEZONE,
)


def extract_window(df, target_date, weather_df):
    """
    Extract the 7-day history window and target-day weather for a given date.

    Returns
    -------
    history_df : DataFrame (1008 rows, NaN where measurements were missing)
    target_timestamps : list of datetime (144, local time for feature engineering)
    weather_temps : array (144)
    actual_target : array (144) or None
    """
    tz = pytz.timezone(TIMEZONE)

    target_start_local = tz.localize(pd.Timestamp(target_date))
    target_end_local = target_start_local + pd.Timedelta(hours=23, minutes=50)
    history_start_local = target_start_local - pd.Timedelta(days=7)

    target_start_utc = target_start_local.astimezone(pytz.utc)
    target_end_utc = target_end_local.astimezone(pytz.utc)
    history_start_utc = history_start_local.astimezone(pytz.utc)

    df = df.copy()
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC")
    else:
        df["Date"] = df["Date"].dt.tz_convert("UTC")

    mask_hist = (df["Date"] >= history_start_utc) & (df["Date"] < target_start_utc)
    history = df.loc[mask_hist].copy()

    # Reindex against a full 10-min grid so any missing timestamps become NaN
    # rows the imputer can detect.
    full_grid = pd.date_range(
        history_start_utc, periods=LOOKBACK_POINTS, freq="10min", tz="UTC"
    )
    history = (
        history
        .set_index("Date")
        .reindex(full_grid)
        .rename_axis("Date")
        .reset_index()
    )

    n_missing = history.iloc[:, 1:].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"Detected {n_missing} missing timestamps in 7-day window (real gaps)")

    target_timestamps = pd.date_range(
        target_start_local, target_end_local, freq="10min"
    ).tolist()

    weather_df = weather_df.copy()
    if weather_df["Date"].dt.tz is None:
        weather_df["Date"] = weather_df["Date"].dt.tz_localize("UTC")
    else:
        weather_df["Date"] = weather_df["Date"].dt.tz_convert("UTC")

    mask_weather = (weather_df["Date"] >= target_start_utc) & (
        weather_df["Date"] <= target_end_utc
    )
    weather_slice = weather_df.loc[mask_weather]

    if len(weather_slice) >= POINTS_PER_DAY:
        weather_temps = weather_slice["AirTemp"].values[:POINTS_PER_DAY]
    else:
        weather_temps = np.full(POINTS_PER_DAY, 15.0)
        if len(weather_slice) > 0:
            weather_temps[: len(weather_slice)] = weather_slice["AirTemp"].values

    mask_actual = (df["Date"] >= target_start_utc) & (df["Date"] <= target_end_utc)
    actual_slice = df.loc[mask_actual]
    actual_target = None
    if len(actual_slice) >= POINTS_PER_DAY:
        actual_target = actual_slice[BUILDING_COLUMN].values[:POINTS_PER_DAY]

    return history, target_timestamps, weather_temps, actual_target
