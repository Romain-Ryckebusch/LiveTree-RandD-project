# 2. Architecture

This chapter is the map of the repository: what lives where, which file owns which responsibility, how data moves through the system on a normal night, and why a handful of non-obvious design decisions were made that way. Read [`01-overview.md`](01-overview.md) first for the business-level picture.

## 2.1 Repository layout

```
RandD-project-public/
|--- README.md                           # Thin landing page; points at docs/
|--- docs/                               # This documentation folder
│   |--- README.md                       # Reading guide / index
│   |--- 01-overview.md
│   |--- 02-architecture.md              # You are here
│   |--- 03-data-model.md
│   |--- 04-usage.md
│   |--- 05-file-reference.md
│   |--- 06-algorithms.md
│   |--- 07-extending.md
│   |--- 08-glossary.md
|--- data/                               # Reference CSVs
│   |--- Cons_Hotel Academic_2026-03-22_2026-04-10.csv
│   |--- 2026 historical data.csv
│   |--- 2026 weather data.csv
│   |--- 2026 forecast data.csv
│   |--- Holidays.xlsx
|--- Imputation-Module/
    |--- src/                            # The actual Python package
    │   |--- config.py                   # Constants, env vars, building list
    │   |--- cassandra_client.py         # All Cassandra I/O
    │   |--- window.py                   # 7-day window slice + grid reindex
    │   |--- smart_imputation.py         # The engine: ExtendedDeploymentAlgorithm
    │   |--- imputer.py                  # Public impute() API + flag mapping
    │   |--- impute_cli.py               # argparse entry point
    │   |--- scheduler.py                # APScheduler nightly daemon
    │   |--- plot_reconstruction.py      # PNG overlay renderer
    |--- cassandra/
    │   |--- schema.cql                  # DDL for the output table
    |--- docker-imputation/
        |--- Dockerfile                  # Python 3.7-slim build
        |--- docker-compose.yml          # rd_imputer service
        |--- requirements.txt            # Pinned dependencies (==)
        |--- run-all-buildings.sh        # Bash wrapper: all five columns in sequence
```

Two folders are gitignored and therefore do not appear in a fresh clone but exist at runtime. `Imputation-Module/docker-imputation/io/` is the mount point for input CSVs, output CSVs, PNGs, and audit logs; the container sees it as `/io/`. `Imputation-Module/docker-imputation/cache/` is the persistent template cache; it is safe to delete, and templates would be rebuilt on the next run if the cache were actively used.

Every file under `Imputation-Module/` is described, one by one, in [`05-file-reference.md`](05-file-reference.md).

## 2.2 Component responsibilities

| Component | Runs where | Reads | Writes | When |
|---|---|---|---|---|
| `cassandra_client.py` | In-container | `conso_historiques_clean`, `pv_prev_meteo_clean` | `conso_historiques_reconstructed` | Every per-building run |
| `window.py` | In-container | Full-history DataFrame + target date | 1008-row window on a 10-min grid | Every per-building run |
| `smart_imputation.ExtendedDeploymentAlgorithm` | In-container | Windowed DataFrame with NaNs | Same shape, NaN-free, plus a strategy log | Every per-building run |
| `imputer.impute()` | In-container | A pandas Series with gaps + timestamps | `(values, quality_flags)` | Per-building |
| `impute_cli.py` | In-container | CLI args | Output CSV + optional PNG + Cassandra upsert | Once per building per run |
| `scheduler.py` | In-container | (none) | Subprocesses `impute_cli.py` | APScheduler cron at 23:50 Europe/Paris |
| `plot_reconstruction.py` | In-container | Output CSV | PNG | Optional, per `--plot` |

Every file under `src/` imports constants from `config.py`. The import graph is a shallow fan-out from there; nothing is deeper than two hops.

## 2.3 End-to-end data flow

One production run, one night, one building:

Step by step:

 1. APScheduler ticks at 23:50 Europe/Paris and invokes `scheduler.run_daily_imputation`.
 2. The scheduler wipes stale `/io/reconstructed_*.csv` and `*.png` files from the previous run.
 3. The scheduler enters a loop, once per building in `config.BUILDINGS`.
 4. Inside the loop, the scheduler spawns `python impute_cli.py --source cassandra --target-date <tomorrow> --building <b> --output <path>` as a subprocess.
 5. The CLI calls `cassandra_client.load_historical_data_cassandra()`, which runs `SELECT * FROM conso_historiques_clean WHERE name='Conso_Data'`.
 6. The CLI calls `cassandra_client.load_weather_data_cassandra()`, which runs `SELECT * FROM pv_prev_meteo_clean WHERE name='Meteorological_Prevision_Data'`.
 7. The CLI calls `window.extract_window(hist, target_date, weather)`, which slices the 7 days ending yesterday, reindexes onto the 10-minute grid, and converts UTC to Europe/Paris.
 8. The CLI calls `imputer.impute(series, timestamps, building_column=...)`, which prepends 56 days of history and delegates to the engine.
 9. `imputer.impute` calls `ExtendedDeploymentAlgorithm.impute(df, weather_df=...)`, which detects gaps, routes each one by length and day-type, fills it, anchors it, and writes the audit log.
10. The algorithm returns the filled DataFrame and the strategy log to `imputer.impute`, which returns `(values, quality_flags)` to the CLI.
11. The CLI writes the output CSV.
12. The CLI calls `cassandra_client.write_reconstructed_window(building, ts, vals, quality)`, which runs an `INSERT (name, Date, Ptot_<b>, quality_<b>)` once per row (1008 rows).
13. If `--plot` was passed, the CLI renders the PNG overlay.
14. The loop continues to the next building.
15. After the loop, the scheduler logs a one-line summary of the form `N/M buildings succeeded`.

A crash in one building is caught by `scheduler.run_daily_imputation` and logged; the other buildings keep running, because each is its own subprocess. `write_reconstructed_window` names only the current building's value column and its quality column in the `INSERT`, so five independent nightly runs on the same `(name, Date)` row never clobber each other. And `scheduler._wipe_previous_outputs` removes `/io/reconstructed_*.csv` and `*.png` at the start of every batch so one night's output cannot be confused with another's. Audit logs under `/io/audit_logs/` are kept.

## 2.4 Key design decisions

These five decisions recur throughout the code.

### 2.4.1 Strict 10-minute grid, exactly 1008 rows

Every function in the module assumes the input has been reindexed onto a complete 10-minute grid, so missing Cassandra rows appear as explicit NaNs rather than as gaps in the index. This unifies two failure modes ("sensor outage" and "no row written") into one shape the gap detector can walk.

The forecaster was trained on exactly 1008 rows in this exact order. `impute_cli.py` validates the row count and grid spacing (1-second tolerance) at ingest and fails fast on a mismatch. The constants live in `config.py`: `POINTS_PER_DAY = 144`, `LOOKBACK_DAYS = 7`, `LOOKBACK_POINTS = 1008`, `FREQ = "10min"`.

### 2.4.2 56-day history prepend

Before imputing, `imputer.impute()` prepends up to 56 days of historical context to the 1008-row window (`_PREPEND_DAYS = 56`). The engine actually reads this *extended* frame; the extension is trimmed off when quality flags are projected back onto the caller's series.

Why 56 days specifically. Weekly and day-of-week templates need enough occurrences of each `(day-of-week, hour-of-day)` cell to build a stable median; the algorithm's default `template_lookback_days=28` means it needs at least four weeks of context, and 56 gives margin. The long-gap donor-day ensemble (above six hours) also has a "more than 40 percent missing" fallback branch that kicks in when there is not enough material, and 56 days of prepend is what keeps that branch from firing on realistic outages.

For `Ptot_HA`, the history comes from reference CSVs in `data/`. For every other building, the CLI seeds the history from the Cassandra full-history pull before calling `impute()` (via `imputer.set_history_source()`).

### 2.4.3 Nightly batch, not streaming

Prediction runs once a day, imputation runs once a day about two hours before. A streaming design was rejected because the forecast granularity is 24 hours and the outages the module exists to handle last minutes to days, not seconds. Batch is simpler, cheaper, easier to audit, and matches the cadence of the consumer.

### 2.4.4 Per-building isolation

Every nightly run spawns `impute_cli.py` once per building as a subprocess. A crash in one building's Python process cannot corrupt another building's run, since they share no memory. Per-building quality columns in the reconstructed table mean siblings' flags are never overwritten by a partial rerun. And rerunning a single building for a single target date is safe and idempotent: the same `(name, Date)` rows get upserted with fresh values.

### 2.4.5 Idempotent reconstruction

`conso_historiques_reconstructed` has primary key `(name, "Date")`. Re-running any `target_date` for any building overwrites the same rows; there is no append-only log. This is intentional: if a late-arriving real measurement shows up in `conso_historiques_clean` after the nightly run, re-running the same night produces a more accurate reconstruction without bookkeeping.

## 2.5 Runtime topology

Components and where they live at runtime:

```
Docker host
|--- rd_imputer container (Python 3.7-slim)
│   |--- scheduler.py                           (APScheduler BlockingScheduler)
│   |--- impute_cli.py subprocesses             (one per building, per run; spawned via subprocess.run)
│
|--- Host-mounted volumes
    |--- ../../data       -> /data        (ro)  reference CSVs, read-only
    |--- ./io             -> /io          (rw)  output CSVs, PNGs, audit logs
    |--- ./cache          -> /app/cache   (rw)  template cache (reserved, unused by shipped code)

External (not managed by this repo)
|--- Cassandra cluster, keyspace previsions_data
```

The `rd_imputer` container talks to the Cassandra cluster over CQL, through the external Docker network `cassandra_default`.

The container joins `cassandra_default` (declared in `docker-compose.yml` as `cassandra_net` with `external: true`). That network is expected to already exist on the host, provisioned by the Cassandra stack. If Cassandra lives elsewhere, drop the `networks:` block and set `CASSANDRA_HOSTS` directly; see [`04-usage.md`](04-usage.md).

Three volumes are mounted: `../../data:/data:ro` holds reference CSVs read-only, so dev fixtures cannot be clobbered by a misbehaving run; `./io:/io` holds inputs, outputs, and audit logs (operators inspect results and drop new input CSVs here); and `./cache:/app/cache` is reserved for precomputed template pickles and is safely discardable.

## 2.6 Configuration surface

All runtime behaviour is driven by environment variables. Defaults are in `config.py`, the Dockerfile pre-sets container paths, and `docker-compose.yml` forwards Cassandra and schedule variables from the host environment or a `.env` file.

| Variable | Default | Consumed by | Effect |
|---|---|---|---|
| `CASSANDRA_HOSTS` | `127.0.0.1` (compose: `cassandra-single-node`) | `config.py` | Comma-separated contact points. Production: `10.64.253.10,10.64.253.11,10.64.253.12`. |
| `CASSANDRA_USERNAME` | *(empty)* | `cassandra_client.py` | If set, switches to `PlainTextAuthProvider`. |
| `CASSANDRA_PASSWORD` | *(empty)* | `cassandra_client.py` | Paired with username. |
| `CASSANDRA_KEYSPACE` | `previsions_data` | `config.py` | Keyspace holding the input and output tables. |
| `IMPUTER_DATA_DIR` | `<repo>/data` (Docker: `/data`) | `config.py` | Reference CSV directory. Used for the `Ptot_HA` history-prepend fallback. |
| `IMPUTER_RECENT_HA_CSV` | `$IMPUTER_DATA_DIR/Cons_Hotel Academic_2026-03-22_2026-04-10.csv` | `config.py` | Override the recent-HA CSV path without moving the file. |
| `IMPUTER_OUTPUT_DIR` | `<src>/output/` (Docker: `/app/cache`) | `config.py` | Template cache location. |
| `IMPUTER_AUDIT_LOG_DIR` | `$IMPUTER_OUTPUT_DIR/audit_logs` (Docker: `/io/audit_logs`) | `config.py` | Per-run audit JSON output. |
| `SCHEDULE_HOUR` | `23` | `scheduler.py` | Cron hour for the nightly daemon tick (0 to 23). |
| `SCHEDULE_MINUTE` | `50` | `scheduler.py` | Cron minute for the nightly tick (0 to 59). |
| `TZ` | `Europe/Paris` (Docker) | Container OS | Should match `config.TIMEZONE`. |

The operational reference (how to set these, how to override at the command line, how to use `.env` files) is in [`04-usage.md`](04-usage.md). The rationale for every non-obvious default is in [`07-extending.md`](07-extending.md).
