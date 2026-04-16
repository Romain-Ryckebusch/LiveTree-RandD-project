import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
DATA_DIR = os.environ.get(
    "IMPUTER_DATA_DIR",
    os.path.join(PROJECT_ROOT, "phase-2", "data"),
)

HISTORICAL_CSV = os.path.join(DATA_DIR, "Cons_Hotel Academic_2026-03-22_2026-04-10.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "2026 weather data.csv")

RECENT_HA_CSV = os.environ.get(
    "IMPUTER_RECENT_HA_CSV",
    os.path.join(DATA_DIR, "Cons_Hotel Academic_2026-03-22_2026-04-10.csv"),
)

BUILDING_COLUMN = "Ptot_HA"
BUILDINGS = [
    "Ptot_HA",
    "Ptot_HEI_13RT",
    "Ptot_HEI_5RNS",
    "Ptot_RIZOMM",
    "Ptot_Campus",
]
CAMPUS_COMPONENTS = BUILDINGS[:4]
LOW_VARIANCE_AUTO_FRACTION = 0.05
LOW_VARIANCE_FLOOR_W = 100.0

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
    "IMPUTER_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
