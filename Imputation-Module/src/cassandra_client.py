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
    RECONSTRUCTED_TABLE,
    RECONSTRUCTED_PARTITION_KEY,
    BUILDING_TO_RECONSTRUCTED_COLUMNS,
)


def _get_session():
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
    """Pull the full consumption table. Returns a DataFrame with Date plus
    one column per building (Ptot_HA, Ptot_HEI_13RT, Ptot_HEI_5RNS, Ptot_RIZOMM)."""
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
    """Pull the weather table. Returns a DataFrame with Date and AirTemp."""
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


def write_reconstructed_window(building, timestamps, values, quality):
    """Upsert one building's reconstructed 7-day window (1008 rows) into
    conso_historiques_reconstructed. Only the building's value and quality
    columns are named, so sibling buildings' columns on the same row are
    left untouched."""
    if building not in BUILDING_TO_RECONSTRUCTED_COLUMNS:
        raise ValueError(f"unknown building {building!r}")
    value_col, quality_col = BUILDING_TO_RECONSTRUCTED_COLUMNS[building]

    session, cluster = _get_session()
    try:
        cql = (
            f'INSERT INTO {RECONSTRUCTED_TABLE} '
            f'(name, "Date", "{value_col}", {quality_col}) '
            f'VALUES (?, ?, ?, ?)'
        )
        prepared = session.prepare(cql)
        partition = RECONSTRUCTED_PARTITION_KEY
        for ts, v, q in zip(timestamps, values, quality):
            session.execute(
                prepared,
                (partition, ts.to_pydatetime(), float(v), int(q)),
            )
        print(
            f"[Cassandra] Wrote {len(values)} rows to {RECONSTRUCTED_TABLE} "
            f"for {building}"
        )
    finally:
        cluster.shutdown()
