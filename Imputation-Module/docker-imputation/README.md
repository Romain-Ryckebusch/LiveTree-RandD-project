# docker-imputation

Docker image for the imputation module. Wraps `Imputation-Module/src/imputer.py`
(backed by `TemperatureAwareHybridEngine`) and feeds it either a CSV of a 7-day
window or a live Cassandra pull. Returns the imputed series plus quality flags.

## Build

```bash
cd "Imputation-Module/docker-imputation"
docker compose build
```

The compose file's build context is the project root, so the Dockerfile can
`COPY` from `Imputation-Module/src/`. First build is 3-5 minutes; subsequent
builds reuse the pip layer.

## Run

### CSV mode

Put your input at `./io/input.csv`, then:

```bash
docker compose run --rm imputer --source csv --input /io/input.csv --output /io/output.csv --seed 42
```

On success: `[OK] Imputed N gap point(s) -> /io/output.csv` and exit 0.
On validation or runtime errors: `ERROR: ...` on stderr and exit 1.

### Cassandra mode

Point the container at a reachable cluster and name the target date. The
7-day window ending the day *before* `--target-date` gets pulled and imputed.

```bash
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12 \
  docker compose run --rm imputer \
    --source cassandra \
    --target-date 2026-04-10 \
    --output /io/output.csv
```

Credentials and keyspace are optional. Can be kept in a `.env` next to
`docker-compose.yml` (already allowed via `env_file: required: false`).

### Test mode

For evaluating reconstruction quality against known-good data, add one or
more `--test-gap START END` pairs. Those rows are replaced with NaN *in memory*
before imputation; the source (CSV file or Cassandra cluster) is never modified.

```bash
CASSANDRA_HOSTS=10.64.253.10 docker compose run --rm imputer \
    --source cassandra --target-date 2026-04-16 --building Ptot_HEI_13RT \
    --output /io/output.csv \
    --test-gap "2026-04-14 08:00" "2026-04-14 14:00" \
    --test-gap "2026-04-15 18:00" "2026-04-15 22:00" \
    --test-report /io/test_report.csv \
    --plot /io/plot.png
```

- Bounds are naive Europe/Paris times, inclusive on both ends.
- `--test-gap` is repeatable.
- `--test-report PATH` writes one row per masked point with `timestamp`,
  `ground_truth`, `imputed`, `quality`, `abs_error`, then summary metrics
  (`# MAE=...`, `# RMSE=...`, `# max_err=...`) as footer comments. The stdout
  also gets a one-line `[TEST] ...` summary.
- When `--plot` is combined with `--test-gap`, each masked range shows up as
  a translucent grey band on the PNG.
- A range fully outside the 7-day window is a hard error. One partially
  outside gets clipped with a warning. Masking the entire window is only
  allowed in Cassandra mode (where the 56-day history gives context); in CSV
  mode it's a hard error.

## I/O contract

### Input CSV

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | ISO 8601. Tz-aware or naive (naive treated as UTC).                  |
| `value`     | float  | `Ptot_HA` consumption in watts. Empty / NaN for gaps.                |

Rules the CLI enforces:

- Exactly 1008 rows (7 days x 144 points/day at 10 min).
- Timestamps strictly increasing, on a 10-min grid (1-second tolerance).
- At least one non-NaN value.

### Cassandra mode input

No input file. Reads from the tables listed in `Imputation-Module/src/config.py`:

- `conso_historiques_clean` (partition key `Conso_Data`)
- `pv_prev_meteo_clean` (partition key `Meteorological_Prevision_Data`)

The 7-day window is reindexed onto a full 10-min grid, so any rows missing in
Cassandra become NaN gaps the imputer can actually see.

### Output CSV

Same shape either way:

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | Echoed from the input (CSV) or ISO 8601 UTC (Cassandra).             |
| `value`     | float  | Imputed series, no NaN remains.                                      |
| `quality`   | int    | Strategy flag (legend below).                                        |

#### Quality flag legend

| Flag | Meaning                                                                  |
|------|--------------------------------------------------------------------------|
| `0`  | Real measurement (input wasn't NaN).                                     |
| `1`  | Linear interpolation (very short gap).                                   |
| `2`  | Contextual / hybrid strategy (short to medium gap).                      |
| `3`  | Donor-day or ensemble fallback (long gap or anomalous segment).          |

## Mounts

`docker-compose.yml` wires three volumes:

| Host path  | Container path | Mode |
|------------|----------------|------|
| `data/`    | `/data/`       | ro   |
| `./io/`    | `/io/`         | rw   |
| `./cache/` | `/app/cache/`  | rw   |

The CSVs under `/data/` feed the CSV-mode 56-day history prepend. In Cassandra
mode the 56-day context comes from the same pull as the 7-day window, and the
CSVs are only consulted if the Cassandra pull is empty.

`./cache/` persists `hybrid_templates_cache.pkl` so the engine doesn't rebuild
seasonal templates on every run.

## Env vars

Paths inside the container:

- `IMPUTER_DATA_DIR=/data`
- `IMPUTER_RECENT_HA_CSV=/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv`
- `IMPUTER_OUTPUT_DIR=/app/cache`

Cassandra connection (defaults in parens):

- `CASSANDRA_HOSTS` (`127.0.0.1`): comma-separated contact points
- `CASSANDRA_USERNAME` (empty, skips auth)
- `CASSANDRA_PASSWORD` (empty)
- `CASSANDRA_KEYSPACE` (`previsions_data`)

Override any of them on the command line (`-e VAR=value`), via the shell
environment (compose picks them up), or through a `.env` next to
`docker-compose.yml`.
