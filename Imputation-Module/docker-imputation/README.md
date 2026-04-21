# docker-imputation

Docker image for the imputation module. Wraps `Imputation-Module/src/imputer.py`
(backed by `TemperatureAwareHybridEngine`) and feeds it either a CSV of a 7-day
window or a live Cassandra pull. Returns the imputed series plus quality flags.

## Build

```bash
cd "Imputation-Module/docker-imputation"
docker compose build
```

The build context is the project root, so the Dockerfile can `COPY` from
`Imputation-Module/src/`. First build takes 3-5 minutes; later builds reuse
the pip layer.

## Daemon mode (default)

`docker compose up -d` starts a long-running scheduler. At 23:50 Europe/Paris
every day, the scheduler pulls the 7-day window ending at the current timestamp
and runs the imputer once for each entry in `config.BUILDINGS`. The
reconstructed series feeds the next day's forecast (a run on the evening of
2026-04-22 produces the input for the 2026-04-23 prediction).

Outputs land at `./io/reconstructed_<building>_<target_date>.csv`. Previous
`reconstructed_*.csv` files in `/io/` are wiped at the start of each batch so
only the most recent run remains; audit logs under `/io/audit_logs/` are kept.

When one building fails the error is logged to stdout and the batch moves on to
the next. The container itself stays up thanks to `restart: unless-stopped`.
Follow the logs with `docker compose logs -f imputer`.

The fire time can be changed via environment variables:

- `SCHEDULE_HOUR` (default `23`, range `0-23`)
- `SCHEDULE_MINUTE` (default `50`, range `0-59`)

Put them in a `.env` next to `docker-compose.yml`, or set them inline:
`SCHEDULE_HOUR=2 SCHEDULE_MINUTE=0 docker compose up -d`.

## Manual trigger

To run the same batch on demand without restarting the daemon, spin up a
sibling container with `--run-now`:

```bash
  docker compose run --rm imputer scheduler.py --run-now --with-plots
```

`--run-now` executes the batch once and exits; the exit code is non-zero if at
least one building failed. The daemon container launched by
`docker compose up -d` is untouched; `docker compose run` creates a separate
ephemeral container alongside it. The target date is still "tomorrow in
Europe/Paris", same as the nightly run.

Plot flags (all require `--with-plots`, which adds
`--plot /io/reconstructed_<building>_<target>.png` to each invocation):

- `--overlay-prior-week`: dashed "copy last week" baseline.
- `--overlay-actual`: solid pre-imputation measured curve.

Example, with both overlays:

```bash
  docker compose run --rm imputer scheduler.py \
    --run-now --with-plots --overlay-prior-week --overlay-actual
```

For a custom `TARGET_DATE`, `--test-gap`, or `--no-clear`, bypass the scheduler
and use `./run-all-buildings.sh` instead (it calls `impute_cli.py` directly).

## Run

Ad-hoc and test runs override the default command. The container's entrypoint
is plain `python`, so you name the script explicitly.

### CSV mode

With the input at `./io/input.csv`:

```bash
docker compose run --rm imputer impute_cli.py --source csv --input /io/input.csv --output /io/output.csv --seed 42
```

A successful run prints `[OK] Imputed N gap point(s) -> /io/output.csv` and
exits 0. Validation or runtime errors go to stderr as `ERROR: ...` with
exit 1.

### Cassandra mode

Point the container at a reachable cluster and supply the target date. The
7-day window ending the day before `--target-date` is pulled and imputed.

```bash
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12 \
  docker compose run --rm imputer impute_cli.py \
    --source cassandra \
    --target-date 2026-04-10 \
    --output /io/output.csv
```

Credentials and keyspace are optional, and can live in a `.env` next to
`docker-compose.yml` (`env_file: required: false` already allows it).

### Test mode

To measure reconstruction quality against known-good data, pass one or more
`--test-gap START END` pairs. The matching rows are replaced with NaN in
memory before imputation runs; neither the source CSV nor the Cassandra
cluster is modified.

```bash
CASSANDRA_HOSTS=10.64.253.10 docker compose run --rm imputer impute_cli.py \
    --source cassandra --target-date 2026-04-16 --building Ptot_HEI_13RT \
    --output /io/output.csv \
    --test-gap "2026-04-14 08:00" "2026-04-14 14:00" \
    --test-gap "2026-04-15 18:00" "2026-04-15 22:00" \
    --test-report /io/test_report.csv \
    --plot /io/plot.png
```

Bounds are naive Europe/Paris times, inclusive on both ends, and `--test-gap`
is repeatable. `--test-report PATH` writes one row per masked point
(`timestamp`, `ground_truth`, `imputed`, `quality`, `abs_error`) plus summary
metrics (`# MAE=...`, `# RMSE=...`, `# max_err=...`) as footer comments; a
one-line `[TEST] ...` summary also goes to stdout.

Combining `--plot` with `--test-gap` draws each masked range as a translucent
grey band on the PNG.

A range fully outside the 7-day window is a hard error; a partially outside
range is clipped with a warning. Masking the entire window is only permitted
in Cassandra mode, because the 56-day history still provides context; CSV
mode rejects it.

## I/O contract

### Input CSV

| Column      | Type   | Notes                                                                |
|-------------|--------|----------------------------------------------------------------------|
| `timestamp` | string | ISO 8601. Tz-aware or naive (naive treated as UTC).                  |
| `value`     | float  | `Ptot_HA` consumption in watts. Empty / NaN for gaps.                |

The CLI checks three things up front: exactly 1008 rows (7 days x 144 points
at 10-min intervals), strictly increasing timestamps on a 10-min grid with a
1-second tolerance, and at least one non-NaN value.

### Cassandra mode input

No input file. The module reads from the tables in
`Imputation-Module/src/config.py`:

- `conso_historiques_clean` (partition key `Conso_Data`)
- `pv_prev_meteo_clean` (partition key `Meteorological_Prevision_Data`)

The 7-day window is reindexed onto a full 10-min grid, so rows missing from
Cassandra show up as NaN gaps and are handled by the imputer.

### Output CSV

Same shape in either mode:

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

The CSVs under `/data/` provide the 56-day history in CSV mode. In Cassandra
mode that same 56-day context is pulled alongside the 7-day window, and the
CSVs are only read as a fallback when the Cassandra pull is empty.

`./cache/` persists `hybrid_templates_cache.pkl` so seasonal templates don't
need to be rebuilt on every run.

## Env vars

Container paths:

- `IMPUTER_DATA_DIR=/data`
- `IMPUTER_RECENT_HA_CSV=/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv`
- `IMPUTER_OUTPUT_DIR=/app/cache`

Cassandra connection (defaults in parens):

- `CASSANDRA_HOSTS` (`127.0.0.1`): comma-separated contact points
- `CASSANDRA_USERNAME` (empty, skips auth)
- `CASSANDRA_PASSWORD` (empty)
- `CASSANDRA_KEYSPACE` (`previsions_data`)

All of these can be overridden on the command line (`-e VAR=value`), from the
shell environment (compose picks them up), or through a `.env` next to
`docker-compose.yml`.
