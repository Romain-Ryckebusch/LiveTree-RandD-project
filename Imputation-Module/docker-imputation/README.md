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

## Daemon mode (default)

`docker compose up -d` launches a long-running scheduler. Every day at **23:50
Europe/Paris** it imputes all five buildings for the next day's prediction:

- `TARGET_DATE` is computed as tomorrow in Europe/Paris (so a run at 23:50 on
  2026-04-22 feeds the 2026-04-23 forecast).
- The 7-day Cassandra window ending at 23:50 of the current day is pulled and
  reconstructed once per building in `config.BUILDINGS`.
- Before each batch, previous `reconstructed_*.csv` files are wiped from
  `/io/` so only the latest run remains on disk. Audit logs under
  `/io/audit_logs/` are preserved.
- Outputs land at `./io/reconstructed_<building>_<target_date>.csv`.
- A failure on one building is logged to stdout and the batch continues with
  the next. The container keeps running thanks to
  `restart: unless-stopped`.

Logs stream on `docker compose logs -f imputer`.

The fire time is configurable via env vars (defaults shown):

- `SCHEDULE_HOUR=23` (0-23)
- `SCHEDULE_MINUTE=50` (0-59)

Set them in a `.env` next to `docker-compose.yml`, or override on the command
line: `SCHEDULE_HOUR=2 SCHEDULE_MINUTE=0 docker compose up -d`.

## Manual trigger

To run the same batch on demand (instead of waiting for 23:50) without
shutting down the daemon, spin up a fresh ephemeral container with
`--run-now`:

```bash
  docker compose run --rm imputer scheduler.py --run-now --with-plots
```

- `--run-now` executes the full per-building batch once and exits. No cron,
  no daemon loop. Exit code is non-zero if any building failed.
- `--with-plots` adds a `--plot /io/reconstructed_<building>_<target>.png`
  to each call so PNG overlays land next to the CSVs. Omit it for CSV-only.
- `--overlay-prior-week` adds the dashed "copy last week" baseline to each
  plot (forwarded to `impute_cli.py`). Only meaningful with `--with-plots`.
- `--overlay-actual` adds the solid pre-imputation measured curve. Only
  meaningful with `--with-plots`.
- Target date is still "tomorrow in Europe/Paris" — same semantics as the
  nightly cron. Use `run-all-buildings.sh` below if you need a custom date
  or test-gap mode.
- The daemon container started by `docker compose up -d` is untouched;
  `docker compose run` creates a sibling container.

Example with the last-week baseline on every plot:

```bash
  docker compose run --rm imputer scheduler.py \
    --run-now --with-plots --overlay-prior-week --overlay-actual
```

For the debug workflow that needs a custom `TARGET_DATE`, `--test-gap`, or
`--no-clear`, use `./run-all-buildings.sh` instead, which bypasses the
scheduler and calls `impute_cli.py` per building directly.

## Run

For ad-hoc / test runs, override the default command and name the script
explicitly (the entrypoint is plain `python`):

### CSV mode

Put your input at `./io/input.csv`, then:

```bash
docker compose run --rm imputer impute_cli.py --source csv --input /io/input.csv --output /io/output.csv --seed 42
```

On success: `[OK] Imputed N gap point(s) -> /io/output.csv` and exit 0.
On validation or runtime errors: `ERROR: ...` on stderr and exit 1.

### Cassandra mode

Point the container at a reachable cluster and name the target date. The
7-day window ending the day *before* `--target-date` gets pulled and imputed.

```bash
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12 \
  docker compose run --rm imputer impute_cli.py \
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
CASSANDRA_HOSTS=10.64.253.10 docker compose run --rm imputer impute_cli.py \
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
