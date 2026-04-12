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
    dfs = pd.read_excel(HOLIDAYS_XLSX, sheet_name=None)
    n = len(target_timestamps)
    entr = np.zeros((n, 15))

    for i, ts in enumerate(target_timestamps):
        entr[i, 0] = ts.timetuple().tm_yday
        entr[i, 1] = ts.minute + ts.hour * 60
        entr[i, 2] = ts.weekday()
        entr[i, 3] = 1 if ts.weekday() >= 5 else 0
        entr[i, 4] = ts.month

        year_str = str(ts.year)
        date_stamp = pd.Timestamp(ts.strftime("%Y-%m-%d"))

        if (dfs[year_str]["Unnamed: 0"] == date_stamp).any():
            entr[i, 5] = 1

        # Closed overrides holiday
        if (dfs[year_str]["Unnamed: 2"] == date_stamp).any():
            entr[i, 6] = 1
            entr[i, 5] = 0

    entr[:, 7] = weather_temps

    col = history_df[BUILDING_COLUMN].values
    entr[:, 8] = col[:POINTS_PER_DAY]
    entr[:, 9] = col[6 * POINTS_PER_DAY:]
    entr[:, 10] = col[5 * POINTS_PER_DAY:6 * POINTS_PER_DAY]
    entr[:, 11] = col[4 * POINTS_PER_DAY:5 * POINTS_PER_DAY]
    entr[:, 12] = col[3 * POINTS_PER_DAY:4 * POINTS_PER_DAY]
    entr[:, 13] = col[2 * POINTS_PER_DAY:3 * POINTS_PER_DAY]
    entr[:, 14] = col[1 * POINTS_PER_DAY:2 * POINTS_PER_DAY]

    return entr


def predict_day(target_timestamps, weather_temps, history_df):
    entr = _build_feature_matrix(target_timestamps, weather_temps, history_df)

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # sklearn<0.24 scalers are missing the clip attr
    for sc in (scaler_x, scaler_y):
        if not hasattr(sc, "clip"):
            sc.clip = False

    x_scaled = scaler_x.transform(entr)
    # Keras 3 wants 3D input for the (None, 15) model
    x_scaled = np.expand_dims(x_scaled, axis=1)
    y_scaled = model.predict(x_scaled)
    y_scaled = y_scaled.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(y_scaled)

    return predictions.flatten()
