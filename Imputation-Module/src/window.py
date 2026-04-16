import numpy as np
import pandas as pd
import pytz

from config import (
    BUILDING_COLUMN,
    LOOKBACK_POINTS,
    POINTS_PER_DAY,
    TIMEZONE,
)


def extract_window(df, target_date, weather_df, building_column=None):
    """Pull the 7-day window ending the day before target_date, plus weather
    for target_date itself. Works on both CSV frames and Cassandra pulls."""
    target_column = building_column or BUILDING_COLUMN
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

    hist_mask = (df["Date"] >= history_start_utc) & (df["Date"] < target_start_utc)
    history = df.loc[hist_mask].copy()

    # Cassandra rows are sparse (missing timestamps have no row at all); reindex
    # onto a full 10-min grid so the imputer can actually see the NaNs.
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

    weather_mask = (weather_df["Date"] >= target_start_utc) & (
        weather_df["Date"] <= target_end_utc
    )
    weather_slice = weather_df.loc[weather_mask]

    if len(weather_slice) >= POINTS_PER_DAY:
        weather_temps = weather_slice["AirTemp"].values[:POINTS_PER_DAY]
    else:
        weather_temps = np.full(POINTS_PER_DAY, 15.0)
        if len(weather_slice) > 0:
            weather_temps[: len(weather_slice)] = weather_slice["AirTemp"].values

    actual_mask = (df["Date"] >= target_start_utc) & (df["Date"] <= target_end_utc)
    actual_slice = df.loc[actual_mask]
    actual_target = None
    if len(actual_slice) >= POINTS_PER_DAY:
        actual_target = actual_slice[target_column].values[:POINTS_PER_DAY]

    return history, target_timestamps, weather_temps, actual_target
