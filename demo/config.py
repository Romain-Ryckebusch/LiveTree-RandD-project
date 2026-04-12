import os

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

# Mar 22 - Apr 10 2026 CSVs at project root, extending HISTORICAL_CSV.
RECENT_HA_CSV = os.path.join(PROJECT_ROOT, "Cons_Hotel Academic_2026-03-22_2026-04-10.csv")
RECENT_SITE_CSV = os.path.join(PROJECT_ROOT, "Cons_Historical Site_2026-03-22_2026-04-10.csv")

BUILDING_COLUMN = "Ptot_HA"
MODEL_SUFFIX = "HA_Puissances_Ptot"

MODEL_FILE = os.path.join(MODEL_DIR, f"my_modelCons3{MODEL_SUFFIX}.h5")
SCALER_Y_FILE = os.path.join(MODEL_DIR, f"scalerConso{MODEL_SUFFIX}.save")
SCALER_X_FILE = os.path.join(MODEL_DIR, f"scalerxConso{MODEL_SUFFIX}.save")

POINTS_PER_DAY = 144
LOOKBACK_DAYS = 7
LOOKBACK_POINTS = POINTS_PER_DAY * LOOKBACK_DAYS
FREQ = "10min"

TIMEZONE = "Europe/Paris"

# --- Cassandra ---
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "127.0.0.1").split(",")
CASSANDRA_USERNAME = os.environ.get("CASSANDRA_USERNAME", "")
CASSANDRA_PASSWORD = os.environ.get("CASSANDRA_PASSWORD", "")
CASSANDRA_KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE", "previsions_data")
CONSO_TABLE = "conso_historiques_clean"
CONSO_PARTITION_KEY = "Conso_Data"
METEO_TABLE = "pv_prev_meteo_clean"
METEO_PARTITION_KEY = "Meteorological_Prevision_Data"

# --- Output ---
OUTPUT_DIR = os.environ.get(
    "DEMO_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
