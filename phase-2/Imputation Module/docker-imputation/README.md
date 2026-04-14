# docker-imputation

Standalone Docker container for the Phase 2 **module de pilotage**. Wraps
`phase-2/Imputation Module/src/imputer.py` (backed by
`TemperatureAwareHybridEngine`) and feeds it either a CSV of a 7-day
`Ptot_HA` window or a live pull from the production Cassandra cluster.
Returns the imputed series plus per-point quality flags.

## Build

```bash
cd "phase-2/Imputation Module/docker-imputation"
docker compose build
```

The build context is the project root so the Dockerfile can `COPY` from
`phase-2/Imputation Module/src/`. First build is ~3-5 minutes; subsequent
builds reuse the pip layer.

## Run

### CSV mode (offline / reproducible)

Place your input CSV at `./io/input.csv`, then:

```bash
docker compose run --rm imputer --source csv --input /io/input.csv --output /io/output.csv --seed 42
```

On success: `[OK] Imputed N gap point(s) -> /io/output.csv` and exit 0.
On any validation or runtime error: an `ERROR: ...` line on stderr and exit 1.

### Cassandra mode (production pull)

Point the container at a reachable Cassandra cluster and name the target
date. The 7-day window ending the day **before** `--target-date` will be
pulled and imputed:

```bash
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12 \
  docker compose run --rm imputer \
    --source cassandra \
    --target-date 2026-04-10 \
    --output /io/output.csv
```

Credentials and keyspace are optional; see *Configuration via env vars*
below. You can also keep them in a local `.env` next to `docker-compose.yml`
(already allowed via `env_file: required: false`).

## I/O contract

### CSV mode input (`/io/input.csv`)

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | ISO 8601. tz-aware or naive (naive is treated as UTC).               |
| `value`     | float  | `Ptot_HA` consumption (W). Empty / `NaN` for gaps.                   |

Constraints (CLI exits non-zero on violation):
- Exactly **1008 rows** (7 days x 144 points/day at 10-minute spacing).
- Timestamps strictly increasing on a 10-minute grid (1-second tolerance).
- At least one non-NaN value.

### Cassandra mode input

No input file. The container reads from the tables listed in
`phase-2/Imputation Module/src/config.py`:

- `conso_historiques_clean` (partition key `Conso_Data`)
- `pv_prev_meteo_clean` (partition key `Meteorological_Prevision_Data`)

The 7-day window is reindexed onto a complete 10-minute grid so that
missing rows in Cassandra become NaN gaps the imputer can detect.

### Output CSV (`/io/output.csv`)

Same contract in both modes:

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | Echoed from the input (CSV) or ISO 8601 UTC (Cassandra).             |
| `value`     | float  | Imputed series. No NaN remain.                                       |
| `quality`   | int    | Per-point imputation strategy (see legend).                          |

#### Quality flag legend

| Flag | Meaning                                                                  |
|------|--------------------------------------------------------------------------|
| `0`  | Real measurement (input was not NaN).                                    |
| `1`  | Linear interpolation (very short gap).                                   |
| `2`  | Contextual / hybrid strategy (short to medium gap).                      |
| `3`  | Donor-day or ensemble fallback (long gap or anomalous segment).          |

## Mounts

`docker-compose.yml` wires up three volumes:

| Host path                               | Container path  | Mode |
|-----------------------------------------|-----------------|------|
| `phase-2/data/`                         | `/data/`        | ro   |
| `./io/`                                 | `/io/`          | rw   |
| `./cache/`                              | `/app/cache/`   | rw   |

The historical CSVs in `/data/` are needed for **CSV-mode** runs and as a
fallback for the 56-day history prepend in CSV mode. In Cassandra mode the
56-day context is drawn from the same pull as the 7-day window and the
CSVs are only consulted if the Cassandra pull is empty.

`./cache/` persists `hybrid_templates_cache.pkl` so the engine doesn't
rebuild seasonal templates on every run.

## Configuration via env vars

Paths inside the container:
- `IMPUTER_DATA_DIR=/data`
- `IMPUTER_RECENT_HA_CSV=/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv`
- `IMPUTER_OUTPUT_DIR=/app/cache`

Cassandra connection (defaults in parentheses):
- `CASSANDRA_HOSTS` (`127.0.0.1`): comma-separated list of contact points
- `CASSANDRA_USERNAME` (empty, skips auth)
- `CASSANDRA_PASSWORD` (empty)
- `CASSANDRA_KEYSPACE` (`previsions_data`)

Override any of them on the command line (`-e VAR=value`), via the shell
environment (compose picks them up), or through a local `.env` file next
to `docker-compose.yml`.

## Forward path

When the production pipeline is wired to Kafka, add a second entry script
`src/kafka_consumer.py` that:

1. Subscribes to a holed-window topic.
2. Calls `from imputer import impute` (the engine stays warm in the
   long-running process, much faster than one-shot CLI invocations).
3. Publishes the imputed window back to Kafka or writes directly to
   Cassandra via `cassandra_client`.

No changes to `imputer.py`, `hybrid_engine.py`, or this Dockerfile will be
needed. The only addition would be a `kafka-python` (or Confluent) entry in
a separate `requirements-kafka.txt`, layered on top of the existing image.
