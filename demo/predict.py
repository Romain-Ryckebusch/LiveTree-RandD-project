"""
Single-building prediction module, adapted from ConsoFile.py.

Takes a 7-day consumption window (1008 points) + weather data for the target
day and produces 144 predicted values (one per 10-min interval).
"""
import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
import joblib

from config import (
    MODEL_FILE,
    SCALER_X_FILE,
    SCALER_Y_FILE,
    HOLIDAYS_XLSX,
    BUILDING_COLUMN,
    LOOKBACK_POINTS,
    POINTS_PER_DAY,
    TIMEZONE,
)


def _build_feature_matrix(target_timestamps, weather_temps, history_df):
    """
    Build the 15-feature input matrix for one day of predictions.

    Parameters
    ----------
    target_timestamps : list of datetime (tz-aware, Europe/Paris)
        144 timestamps for the target prediction day.
    weather_temps : array-like of float, length 144
        Air temperature for each target timestamp.
    history_df : DataFrame
        Must contain at least 1008 rows with columns [Date, BUILDING_COLUMN],
        ordered chronologically, covering the 7 days before the target day.

    Returns
    -------
    np.ndarray of shape (144, 15)
    """
    dfs = pd.read_excel(HOLIDAYS_XLSX, sheet_name=None)
    n = len(target_timestamps)
    entr = np.zeros((n, 15))

    for i, ts in enumerate(target_timestamps):
        entr[i, 0] = ts.timetuple().tm_yday       # day of year
        entr[i, 1] = ts.minute + ts.hour * 60      # minute of day
        entr[i, 2] = ts.weekday()                   # weekday 0=Mon
        entr[i, 3] = 1 if ts.weekday() >= 5 else 0  # is weekend
        entr[i, 4] = ts.month                        # month

        year_str = str(ts.year)
        date_stamp = pd.Timestamp(ts.strftime("%Y-%m-%d"))

        # Holiday flag
        if (dfs[year_str]["Unnamed: 0"] == date_stamp).any():
            entr[i, 5] = 1

        # Closed flag (overrides holiday)
        if (dfs[year_str]["Unnamed: 2"] == date_stamp).any():
            entr[i, 6] = 1
            entr[i, 5] = 0

    # Weather
    entr[:, 7] = weather_temps

    # Historical consumption lags (from the 1008-point window)
    col = history_df[BUILDING_COLUMN].values
    entr[:, 8] = col[:POINTS_PER_DAY]                               # j-7
    entr[:, 9] = col[6 * POINTS_PER_DAY:]                           # j-1
    entr[:, 10] = col[5 * POINTS_PER_DAY:6 * POINTS_PER_DAY]       # j-2
    entr[:, 11] = col[4 * POINTS_PER_DAY:5 * POINTS_PER_DAY]       # j-3
    entr[:, 12] = col[3 * POINTS_PER_DAY:4 * POINTS_PER_DAY]       # j-4
    entr[:, 13] = col[2 * POINTS_PER_DAY:3 * POINTS_PER_DAY]       # j-5
    entr[:, 14] = col[1 * POINTS_PER_DAY:2 * POINTS_PER_DAY]       # j-6

    return entr


def predict_day(target_timestamps, weather_temps, history_df):
    """
    Predict consumption for one day (144 intervals) for the demo building.

    Parameters
    ----------
    target_timestamps : list of datetime (tz-aware, Europe/Paris), length 144
    weather_temps : array-like of float, length 144
    history_df : DataFrame with 1008 rows and BUILDING_COLUMN

    Returns
    -------
    np.ndarray of shape (144,),predicted power in watts
    """
    entr = _build_feature_matrix(target_timestamps, weather_temps, history_df)

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # Patch scalers saved with sklearn<0.24 (missing 'clip' attr)
    for sc in (scaler_x, scaler_y):
        if not hasattr(sc, "clip"):
            sc.clip = False

    x_scaled = scaler_x.transform(entr)
    # Model saved with input_shape=(None,15) expects 3D; reshape for Keras 3
    x_scaled = np.expand_dims(x_scaled, axis=1)
    y_scaled = model.predict(x_scaled)
    y_scaled = y_scaled.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(y_scaled)

    return predictions.flatten()
