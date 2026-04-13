# docker-imputation

Standalone Docker container that wraps the project's imputation module
(`demo/imputer.py`, backed by `TemperatureAwareHybridEngine`). It takes a
7-day window of holed `Ptot_HA` consumption data as a CSV and returns the
imputed series plus per-point quality flags.

This is the first packaging step of the Phase 2 "module de pilotage". The
container's `impute_cli.py` is a thin wrapper; the canonical imputation code
stays in `demo/`. A future Kafka consumer entry point can live in the same
image without touching the imputation logic (see *Forward path* below).

## Build

```bash
cd "phase-2/Imputation Module/docker-imputation"
docker compose build
```

The build context is the project root (so the Dockerfile can `COPY` from
both `demo/` and this directory). First build is ~3-5 minutes; subsequent
builds reuse the pip layer.

## Run

Place your input CSV at `./io/input.csv`, then:

```bash
docker compose run --rm imputer --input /io/input.csv --output /io/output.csv --seed 42
```

On success: `[OK] Imputed N gap point(s) -> /io/output.csv` and exit 0.
On any validation or runtime error: an `ERROR: ...` line on stderr and exit 1.

## I/O contract

### Input CSV (`/io/input.csv`)

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | ISO 8601. tz-aware or naive (naive is treated as UTC).               |
| `value`     | float  | `Ptot_HA` consumption (W). Empty / `NaN` for gaps.                   |

Constraints (CLI exits non-zero on violation):
- Exactly **1008 rows** (7 days x 144 points/day at 10-minute spacing).
- Timestamps strictly increasing on a 10-minute grid (1-second tolerance).
- At least one non-NaN value.

### Output CSV (`/io/output.csv`)

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | Echoed verbatim from the input.                                      |
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

`docker-compose.yml` wires up four volumes:

| Host path                                                       | Container path                                            | Mode |
|-----------------------------------------------------------------|-----------------------------------------------------------|------|
| `phase-2/Data/`                                                 | `/data/`                                                  | ro   |
| `Cons_Hotel Academic_2026-03-22_2026-04-10.csv` (project root)  | `/recent/Cons_Hotel Academic_2026-03-22_2026-04-10.csv`   | ro   |
| `./io/`                                                         | `/io/`                                                    | rw   |
| `./cache/`                                                      | `/app/cache/`                                             | rw   |

(The recent HA CSV mounts into `/recent/` rather than `/data/` because Docker
can't nest a file bind-mount inside a read-only directory mount.)

The historical CSVs in `/data/` are needed because `imputer.py` extends the
input window with 56 days of prior `Ptot_HA` history before calling the
hybrid engine (the long-gap ensemble strategies need that lookback to
calibrate).

`./cache/` persists `hybrid_templates_cache.pkl` so the engine doesn't
rebuild seasonal templates on every run.

## Configuration via env vars

The container sets these so `demo/config.py` resolves paths inside the
container instead of relative to a project root:

- `DEMO_DATA_DIR=/data`
- `DEMO_RECENT_HA_CSV=/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv`
- `DEMO_OUTPUT_DIR=/app/cache`

Override any of them by editing `docker-compose.yml` if your data lives
elsewhere.

## Forward path

When the production pipeline is wired to Kafka/Cassandra, add a second
entry script `src/kafka_consumer.py` that:

1. Subscribes to a holed-window topic.
2. Calls `from imputer import impute` (the engine stays warm in the
   long-running process — much faster than one-shot CLI invocations).
3. Publishes the imputed window back to Kafka or writes directly to
   Cassandra.

No changes to `imputer.py`, `hybrid_engine.py`, or this Dockerfile will be
needed. The only addition would be a `kafka-python` (or Confluent) entry in
a separate `requirements-kafka.txt`, layered on top of the existing image.
