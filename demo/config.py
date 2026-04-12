"""
Centralized configuration for the single-building demo pipeline.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get(
    "DEMO_DATA_DIR",
    os.path.join(PROJECT_ROOT, "2nd Phase", "Data"),
)
MODEL_DIR = os.environ.get(
    "DEMO_MODEL_DIR",
    os.path.join(
        PROJECT_ROOT,
        "phase-2",
        "Prediction Model",
        "docker-previsions-conso",
        "build",
        "previsions_conso",
        "code",
    ),
)

HISTORICAL_CSV = os.path.join(DATA_DIR, "Cons_Hotel Academic_2026-03-22_2026-04-10.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "2026 weather data.csv")
HOLIDAYS_XLSX = os.path.join(MODEL_DIR, "Holidays.xlsx")

# --- Demo building ---
# Single building for the demo; multi-building comes later.
BUILDING_COLUMN = "Ptot_HA"
MODEL_SUFFIX = "HA_Puissances_Ptot"

# Model files (version "3" = latest)
MODEL_FILE = os.path.join(MODEL_DIR, f"my_modelCons3{MODEL_SUFFIX}.h5")
SCALER_Y_FILE = os.path.join(MODEL_DIR, f"scalerConso{MODEL_SUFFIX}.save")
SCALER_X_FILE = os.path.join(MODEL_DIR, f"scalerxConso{MODEL_SUFFIX}.save")

# --- Data constants ---
POINTS_PER_DAY = 144  # 10-minute intervals
LOOKBACK_DAYS = 7
LOOKBACK_POINTS = POINTS_PER_DAY * LOOKBACK_DAYS  # 1008
FREQ = "10min"

# --- Timezone ---
TIMEZONE = "Europe/Paris"

# --- Output ---
OUTPUT_DIR = os.environ.get(
    "DEMO_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
