"""
Cassandra data source for the demo pipeline.

Reads consumption and weather data from the Cassandra tables
and returns DataFrames in the same format as the CSV loaders.
"""
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from config import (
    CASSANDRA_HOSTS,
    CASSANDRA_USERNAME,
    CASSANDRA_PASSWORD,
    CASSANDRA_KEYSPACE,
    CONSO_TABLE,
    CONSO_PARTITION_KEY,
    METEO_TABLE,
    METEO_PARTITION_KEY,
)


def _get_session():
    """Create and return a Cassandra session."""
    # Auth is optional — the supervisors' single-node setup has no auth
    if CASSANDRA_USERNAME:
        auth = PlainTextAuthProvider(
            username=CASSANDRA_USERNAME,
            password=CASSANDRA_PASSWORD,
        )
    else:
        auth = None

    cluster = Cluster(CASSANDRA_HOSTS, auth_provider=auth)
    session = cluster.connect(CASSANDRA_KEYSPACE)

    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    session.row_factory = pandas_factory
    session.default_fetch_size = None
    return session, cluster


def load_historical_data_cassandra():
    """
    Load consumption data from Cassandra.

    Returns a DataFrame with columns matching the CSV format:
    Date, Ptot_HA, Ptot_HEI_13RT, Ptot_HEI_5RNS, Ptot_RIZOMM, ...
    """
    session, cluster = _get_session()
    try:
        query = f'SELECT * FROM {CONSO_TABLE} WHERE name=%s'
        result = session.execute(query, (CONSO_PARTITION_KEY,), timeout=None)
        df = result._current_rows

        if df.empty:
            raise RuntimeError(
                f"No data in {CONSO_TABLE} for name='{CONSO_PARTITION_KEY}'"
            )

        df = df.sort_values("Date").reset_index(drop=True)

        # Drop columns the demo pipeline doesn't use
        for col in ("name", "Quality"):
            if col in df.columns:
                df = df.drop(columns=[col])

        print(
            f"[Cassandra] Loaded {len(df)} consumption rows "
            f"({df['Date'].min()} -> {df['Date'].max()})"
        )
        return df
    finally:
        cluster.shutdown()


def load_weather_data_cassandra():
    """
    Load weather data from Cassandra.

    Returns a DataFrame with columns: Date, AirTemp, ...
    """
    session, cluster = _get_session()
    try:
        query = f'SELECT * FROM {METEO_TABLE} WHERE name=%s'
        result = session.execute(query, (METEO_PARTITION_KEY,), timeout=None)
        df = result._current_rows

        if df.empty:
            raise RuntimeError(
                f"No data in {METEO_TABLE} for name='{METEO_PARTITION_KEY}'"
            )

        df = df.sort_values("Date").reset_index(drop=True)

        if "name" in df.columns:
            df = df.drop(columns=["name"])

        print(
            f"[Cassandra] Loaded {len(df)} weather rows "
            f"({df['Date'].min()} -> {df['Date'].max()})"
        )
        return df
    finally:
        cluster.shutdown()
