# Architecture

This document is the handbook for anyone picking up the project and adapting it to their own use case. It explains what is in this repository, how the parts fit together, and where to touch the code when you need to change something.

It complements `README.md` (which is scoped to operational CLI and Docker usage) and is intended to be read top-to-bottom once, then used as a reference.

## 1. Context and goal

Project N28, *"Amélioration de la résilience des modèles prédictifs par la qualité et la continuité des données"* (JUNIA, LiveTree demonstrator).

Five building consumption channels (`Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, plus a campus-total `Ptot_Campus` computed as their sum) are measured every 10 minutes. A downstream prediction pipeline (out of scope for this repository) consumes 7 days of readings, exactly 1008 points per building (7 × 144), to produce its forecasts.

The acquisition chain (sensors, gateway, Cassandra) is not reliable. Gaps of a few minutes to several hours are routine and multi-hour outages are not rare. The downstream pipeline does not tolerate holes in its 7-day input window, so whenever a gap appears the prediction silently fails to run.

This repository contains the *module de pilotage*: an imputation service that sits between the Cassandra store and the downstream consumer. Given a potentially holed 7-day window, it returns a complete one and tags every point with a quality flag so downstream code can filter on confidence. Anything the downstream consumer does with that window (feature engineering, modelling, publishing) lives in a separate repository and is out of scope here.

The module's goals are to guarantee that a complete 1008-point window is always available, to keep the downstream pipeline running during acquisition outages, to make the backup shapes representative of the actual measurements (seasonality, weekday/weekend, outside temperature), and to surface per-point confidence so consumers can decide what to trust.

## 2. System overview

```
  SENSORS (5 buildings, 10-min interval)
      |
      v
  +---------------------------------------+
  |  Cassandra cluster (external)         |     <-- data lives here
  |  keyspace: previsions_data            |
  |  - conso_historiques_clean    (in)    |
  |  - pv_prev_meteo_clean        (in)    |
  |  - conso_historiques_reconstructed    |     <-- this module's output
  +---------------------------------------+
      ^                       |
      |                       |
      | read 7-day            | write reconstructed window
      | window + weather      |
      |                       v
  +---------------------------------------+
  |  IMPUTATION MODULE      (THIS REPO)   |
  |  Imputation-Module/src/               |
  |    scheduler.py  (APScheduler daemon, |
  |                    nightly 23:50 TZ)  |
  |    impute_cli.py (per-building entry) |
  |    imputer.py    (public API)         |
  |    smart_imputation.py  (strategies)  |
  |    window.py, cassandra_client.py,    |
  |    config.py, plot_reconstruction.py  |
  +---------------------------------------+
                              |
                              v
  +---------------------------------------+
  |  DOWNSTREAM CONSUMER    (OUT OF SCOPE)|
  |  reads conso_historiques_reconstructed|
  +---------------------------------------+
```

The module exposes three integration surfaces, which `README.md` documents in operational detail. The default Docker deployment (`docker compose up`) runs `scheduler.py` as a nightly daemon that spawns one `impute_cli.py` subprocess per building, writes CSVs (and optionally PNGs) to `/io/`, and upserts the reconstructed window back into Cassandra. For manual runs, the test harness, and ad-hoc reconstructions, `impute_cli.py --source {csv|cassandra} ...` can be invoked one-shot. For embedding into another Python process, `from imputer import impute` exposes the core function directly.

## 3. Repository scope

What is inside this repo: the imputation module source (`Imputation-Module/src/`), its Docker packaging (`Imputation-Module/docker-imputation/`), the Cassandra schema for the output table (`Imputation-Module/cassandra/schema.cql`), reference CSVs used by the CSV mode and as a 56-day history seed for `Ptot_HA` (`data/`), and documentation (`README.md` and this `ARCHITECTURE.md`).

What is not in this repo: the downstream prediction pipeline (out of scope); the Cassandra cluster itself, which this module connects to over the Docker network `cassandra_default` (expected to exist outside this project); and phase-1 research artefacts.

## 4. Directory layout

```
.
├── ARCHITECTURE.md                  # this document
├── README.md                        # CLI / Docker operational reference
├── data/                            # reference CSVs (gitignored data files)
│   ├── Cons_Hotel Academic_*.csv
│   ├── 2026 historical data.csv
│   ├── 2026 weather data.csv
│   ├── 2026 forecast data.csv
│   └── Holidays.xlsx
└── Imputation-Module/
    ├── src/                         # the module's source
    │   ├── config.py
    │   ├── imputer.py
    │   ├── smart_imputation.py
    │   ├── window.py
    │   ├── cassandra_client.py
    │   ├── impute_cli.py
    │   ├── scheduler.py
    │   └── plot_reconstruction.py
    ├── cassandra/
    │   └── schema.cql               # conso_historiques_reconstructed
    └── docker-imputation/
        ├── Dockerfile
        ├── docker-compose.yml
        ├── requirements.txt         # pinned
        ├── run-all-buildings.sh     # multi-building CLI loop
        ├── README.md                # Docker-specific usage
        ├── io/                      # mount point for inputs/outputs (gitignored)
        └── cache/                   # hybrid-template cache (gitignored)
```

## 5. Core concepts

All time series in this system are sampled every 10 minutes. One day is exactly 144 points; the 7-day sliding window is exactly 1008 points. These numbers are hard-coded as `POINTS_PER_DAY`, `LOOKBACK_DAYS`, and `LOOKBACK_POINTS` in `Imputation-Module/src/config.py`.

The five building identifiers are defined in `config.py:BUILDINGS`:

| ID              | Meaning                                       |
|-----------------|-----------------------------------------------|
| `Ptot_HA`       | Hôtel Académique                              |
| `Ptot_HEI_13RT` | 13 Rue de Toul                                |
| `Ptot_HEI_5RNS` | 5 Rue Nicolas Souriau                         |
| `Ptot_RIZOMM`   | RIZOMM                                        |
| `Ptot_Campus`   | Sum of the four above, computed internally    |

Every output row carries an integer quality flag describing how its value was produced. Flags are assigned in `imputer.py` via `_STRATEGY_FLAG_MAP`:

| Flag | Meaning                                                                                                                                                           | Typical gap length |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `0`  | Real measurement, no imputation applied                                                                                                                           | n/a                |
| `1`  | Linear interpolation (`LINEAR_MICRO`, `LINEAR_SHORT`)                                                                                                             | <= 30 min          |
| `2`  | Contextual / template / peer (`THERMAL_TEMPLATE`, `ENHANCED_TEMPLATE`, `WEEKEND_TEMPLATE_*`, `SAFE_LINEAR_MEDIAN`, `PEER_CORRELATION`, `HOURLY_MEDIAN_FALLBACK`)   | 30 min to 6 h      |
| `3`  | Donor-day ensemble or ML fallback (`MULTI_WEEK_TEMPLATE`, `MICE`, `KNN_CONTEXT`, `KALMAN_FILTER`, `SAFE_MEDIAN`)                                                   | > 6 h              |

Downstream consumers typically keep `quality <= 1` for high confidence and `quality <= 2` when templates are trusted. A `quality = 3` point describes a plausible shape, not a measurement.

The current demo validates the full pipeline on a single building (`Ptot_HA`). The other four building columns exist and are wired through, but the CSV-based history seed only exists for `Ptot_HA`; for the others, the 56-day history must be injected via `set_history_source()` from Cassandra. Multi-building support is intentionally deferred.

## 6. Data model

### 6.1 Cassandra, keyspace `previsions_data`

Two input tables are read by this module, one output table is written.

`conso_historiques_clean` (input, consumption):

```
PRIMARY KEY (name, "Date")

name           text        -- always 'Conso_Data' for the partition used here
"Date"         timestamp   -- UTC, 10-minute grid
Ptot_HA        double
Ptot_HEI       double      -- legacy / duplicate, not consumed by this module
Ptot_HEI_13RT  double
Ptot_HEI_5RNS  double
Ptot_RIZOMM    double
Quality        int         -- source-system quality flag (0 = real); dropped on read
```

`pv_prev_meteo_clean` (input, weather forecast; only `AirTemp` is consumed):

```
PRIMARY KEY (name, "Date")

name            text       -- 'Meteorological_Prevision_Data'
"Date"          timestamp  -- UTC
AirTemp         double
CloudOpacity    double     -- ignored
Dni10, Dni90, DniMoy, Ghi10, Ghi90, GhiMoy   double   -- ignored
```

`conso_historiques_reconstructed` (output, this module writes here). Schema is in `Imputation-Module/cassandra/schema.cql`:

```
PRIMARY KEY (name, "Date")

name               text       -- 'Conso_Data' (same as input partition)
"Date"             timestamp
Ptot_HA            double
Ptot_HEI_13RT      double
Ptot_HEI_5RNS      double
Ptot_RIZOMM        double
Ptot_Campus        double
quality_ha         int        -- per-building quality flags so the 5 nightly
quality_hei_13rt   int        -- per-building runs don't overwrite each
quality_hei_5rns   int        -- other's flags
quality_rizomm     int
quality_campus     int
```

Apply `schema.cql` once per cluster before enabling `--source=cassandra` writes. Each per-building run upserts only its own value column and quality column, leaving siblings untouched.

### 6.2 CSV format (for CSV mode)

Input CSV, exactly 1008 rows:

| column      | type                | notes                                                                         |
|-------------|---------------------|-------------------------------------------------------------------------------|
| `timestamp` | ISO-8601 string     | Strictly increasing, 10-minute grid (+/- 1 s). Naive or tz-aware both accepted. |
| `value`     | float (watts)       | Empty / `NaN` marks a gap.                                                    |

Output CSV, same row count as input:

| column      | type   | notes                                                 |
|-------------|--------|-------------------------------------------------------|
| `timestamp` | string | Copied from input.                                    |
| `value`     | float  | Imputed, guaranteed non-NaN on success.               |
| `quality`   | int    | Per-point quality flag (see section 5).               |

Detailed flag semantics and the optional test-harness report format are in `README.md`.

## 7. Pipeline end-to-end

One production run, one night, one building:

1. Tick. `scheduler.py` wakes at `SCHEDULE_HOUR:SCHEDULE_MINUTE` (defaults `23:50` Europe/Paris) using APScheduler's `CronTrigger`.
2. Previous artefacts are wiped. Stale `/io/reconstructed_*.csv` and `*.png` files are removed so one night's output cannot be confused with another's.
3. Fan-out. For each entry in `config.BUILDINGS`, the scheduler spawns `python impute_cli.py --source cassandra --target-date <tomorrow> --building <b> --output /io/reconstructed_<b>_<date>.csv` as a subprocess. Subprocess isolation means a crash on one building does not abort the batch.
4. Load. `cassandra_client.load_historical_data_cassandra()` pulls the full consumption partition, and `load_weather_data_cassandra()` pulls the weather partition. Both queries are `SELECT * FROM <table> WHERE name=%s`; the partition keys are `'Conso_Data'` and `'Meteorological_Prevision_Data'` respectively (see `config.CONSO_PARTITION_KEY`, `METEO_PARTITION_KEY`).
5. Extract window. `window.extract_window()` slices the 7 days ending the day before `--target-date`, converts UTC to Europe/Paris, and reindexes onto a full 10-min grid so that rows missing from Cassandra show up as explicit NaNs (this is how the imputer sees gaps).
6. Extend with history. Inside `imputer.impute()`, the window is prepended with up to `_PREPEND_DAYS = 56` days of history loaded either from a Cassandra-side seed (via `set_history_source()`, used by the CLI's Cassandra path) or from the reference CSVs (for `Ptot_HA` only). The long-gap donor-day ensemble requires this much context to stop falling back to its ">40% missing" branch.
7. Impute. `smart_imputation.ExtendedDeploymentAlgorithm.impute()` walks the gaps and, for each one, dispatches to a strategy based on length, day-type (weekend vs. weekday), and peer availability. See section 8.
8. Align and flag. Results are reindexed back to the caller's 1008-row frame; strategy-log entries are translated into per-point quality flags via `_STRATEGY_FLAG_MAP`; any real (non-NaN on input) point is force-flagged `0`.
9. Write. The CLI writes the output CSV to `/io/...`. In Cassandra mode, `cassandra_client.write_reconstructed_window()` prepares an `INSERT INTO conso_historiques_reconstructed` and upserts the 1008 rows, naming only the building's value column and its quality column so siblings are not clobbered.
10. Downstream (out of scope). The reconstructed 7-day window is now available in `conso_historiques_reconstructed` for whatever consumer reads from that table.

## 8. Imputation strategies

The engine is `ExtendedDeploymentAlgorithm` in `Imputation-Module/src/smart_imputation.py`. It owns the entire gap-walking loop and dispatches to per-strategy fill methods (`_fill_linear`, `_fill_with_thermal_template`, `_fill_enhanced_template`, `_fill_weekend_template`, `_fill_with_multi_week_template`, `_fill_via_peer_correlation`, `_fill_with_mice`, `_fill_with_kalman_filter`, `_fill_with_knn_context`, `_fill_safe_median_template`, `_fill_safe_linear_median`).

Gap-length cascade. Gaps of 30 minutes or less are filled by linear interpolation (`LINEAR_MICRO`, `LINEAR_SHORT`); the shape is already well approximated by a straight line over that window. Gaps from 30 min to 6 h use contextual templates: the engine picks between weekday/weekend templates (`WEEKEND_TEMPLATE_*`, `ENHANCED_TEMPLATE`), thermal-regime templates conditioned on outside air temperature (`THERMAL_TEMPLATE`), peer-building correlation when a sibling building is intact over the same window (`PEER_CORRELATION`), and median fallbacks (`SAFE_LINEAR_MEDIAN`, `HOURLY_MEDIAN_FALLBACK`). Gaps longer than 6 h use a donor-day ensemble (`MULTI_WEEK_TEMPLATE`) over the 56-day prepended history, with ML fallbacks (`MICE`, `KNN_CONTEXT`, `KALMAN_FILTER`) and a safety net (`SAFE_MEDIAN`).

Anchoring. After any fill, `_anchor_to_boundaries()` blends the filled segment into the real values at both ends so junctions do not introduce a discontinuity.

Audit logging. Each gap and the strategy chosen are recorded to a per-run JSON audit log under `IMPUTER_AUDIT_LOG_DIR` (default `/io/audit_logs` in Docker). `get_strategy_log()`, `get_gap_log()`, `get_zero_fill_report()`, and `get_raw_anomaly_report()` on the algorithm expose this for inspection.

Zero-fill detection. Sensor chains sometimes emit stretches of literal zeros when they are actually disconnected. `_detect_zero_fills()` uses an IsolationForest over the 56-day context to flag such stretches as anomalies rather than accepting them as measurements.

DST handling. `_detect_dst_events()` tags spring-forward and fall-back days so the weekly template builder does not misalign by one hour on a DST day.

## 9. Component reference

Files under `Imputation-Module/src/`:

- `config.py`: single source of truth for constants and environment-driven settings. Holds the `BUILDINGS` list, the 10-min grid constants (`POINTS_PER_DAY`, `LOOKBACK_DAYS`, `LOOKBACK_POINTS`, `FREQ`), Cassandra connection parameters (`CASSANDRA_HOSTS`, `CASSANDRA_KEYSPACE`, etc.), table and partition-key names (`CONSO_TABLE`, `METEO_TABLE`, `RECONSTRUCTED_TABLE`), the `BUILDING_TO_RECONSTRUCTED_COLUMNS` map, and I/O directories (`DATA_DIR`, `OUTPUT_DIR`, `AUDIT_LOG_DIR`). Every other module imports from here.
- `imputer.py`: public API. Top-level function `impute(series, date_index, quality=None, *, large_gap_min=144, random_seed=None, building_column=None) -> (values, quality_flags)` is what you call when embedding the module in another Python process. Internally it prepends `_PREPEND_DAYS = 56` days of history (via `_load_combined_history()` for `Ptot_HA` or `set_history_source()` for Cassandra-seeded buildings), feeds the extended frame to `ExtendedDeploymentAlgorithm`, then reprojects the result back to the caller's 1008-row series and translates the algorithm's `strategy_log` into per-point quality flags. Also exposes `naive_impute()` as a zero-fill baseline.
- `smart_imputation.py`: the engine. One ~2000-line class, `ExtendedDeploymentAlgorithm`, owning gap detection, routing (`_intelligent_router`), the strategy implementations listed in section 8, boundary anchoring, weekly/thermal/seasonal template building, DST handling, IsolationForest anomaly detection, and audit logging.
- `window.py`: `extract_window(df, target_date, weather_df, building_column)` slices the 7 days ending the day before `target_date`, converts UTC to Europe/Paris, and reindexes onto a full 10-min grid so missing Cassandra rows become explicit NaNs. Returns `(history_df, target_timestamps, weather_temps, actual_target)`. Used by `impute_cli.py` in the Cassandra path.
- `cassandra_client.py`: three functions that own all Cassandra I/O: `load_historical_data_cassandra()`, `load_weather_data_cassandra()`, and `write_reconstructed_window(building, timestamps, values, quality)`. Session creation (`_get_session()`) honours `CASSANDRA_USERNAME`/`CASSANDRA_PASSWORD` if set, anonymous otherwise. Swap this one file to change the data backend (see section 11).
- `impute_cli.py`: the per-building command-line entry. Parses `--source {csv|cassandra}`, `--input`, `--target-date`, `--building`, `--output`, `--plot`, `--test-gap` / `--test-report`, `--overlay-prior-week`, `--overlay-actual`, `--seed`. Validates input (the CSV must be exactly 1008 rows on a 10-min grid), optionally applies synthetic gap masks for the test harness, calls `imputer.impute()`, writes the output CSV, and in Cassandra mode also calls `write_reconstructed_window()`. Full flag tables are in `README.md`.
- `scheduler.py`: APScheduler daemon. On tick (default `23:50 Europe/Paris`, configurable via `SCHEDULE_HOUR` / `SCHEDULE_MINUTE`), wipes stale `/io/` artefacts, then subprocess-calls `impute_cli.py` once per building in `config.BUILDINGS`. `--run-now` runs one batch immediately and exits (return code non-zero if any building failed). `--with-plots`, `--overlay-prior-week`, `--overlay-actual`, `--test-gap` are forwarded to each per-building call.
- `plot_reconstruction.py`: standalone renderer invoked when `--plot` is passed. Produces a matplotlib PNG overlay with per-quality colour coding (`0` real, `1` naive-linear, `2` linear/contextual, `3` donor-day), and optional overlays for the prior-week "copy last week" baseline and the raw measured values.

## 10. Configuration

All runtime configuration is environment-driven. Defaults live in `Imputation-Module/src/config.py`; the Dockerfile pre-sets a production-ready set; `docker-compose.yml` forwards the Cassandra ones from the host environment.

| Variable                  | Default                                          | Purpose                                                                 |
|---------------------------|--------------------------------------------------|-------------------------------------------------------------------------|
| `CASSANDRA_HOSTS`         | `127.0.0.1` (compose: `cassandra-single-node`)   | Comma-separated contact points. Production: `10.64.253.10,.11,.12`.     |
| `CASSANDRA_USERNAME`      | empty                                            | Anonymous when unset.                                                   |
| `CASSANDRA_PASSWORD`      | empty                                            | Paired with username.                                                   |
| `CASSANDRA_KEYSPACE`      | `previsions_data`                                | Keyspace holding input tables and the output reconstructed table.       |
| `IMPUTER_DATA_DIR`        | repo's `data/` (Docker: `/data`)                 | Reference CSVs for the 56-day `Ptot_HA` history seed.                   |
| `IMPUTER_RECENT_HA_CSV`   | `$IMPUTER_DATA_DIR/Cons_Hotel Academic_*.csv`    | Override the recent-HA reference CSV path.                              |
| `IMPUTER_OUTPUT_DIR`      | `src/output/` (Docker: `/app/cache`)             | Template cache (`hybrid_templates_cache_<building>.pkl`).               |
| `IMPUTER_AUDIT_LOG_DIR`   | `$IMPUTER_OUTPUT_DIR/audit_logs` (Docker: `/io/audit_logs`) | Per-run audit JSON.                                          |
| `SCHEDULE_HOUR`           | `23`                                             | Cron hour for the nightly daemon tick.                                  |
| `SCHEDULE_MINUTE`         | `50`                                             | Cron minute for the nightly daemon tick.                                |
| `TZ`                      | `Europe/Paris` (Docker)                          | Container timezone; should match `config.TIMEZONE`.                     |

Docker network. `docker-compose.yml` joins the external network `cassandra_default` (declared as `cassandra_net` with `external: true`). The Cassandra stack is expected to run outside this project and to have created that network. If your Cassandra runs elsewhere, adjust the `networks:` block or remove it and rely on `CASSANDRA_HOSTS` directly.

Volumes. `../../data:/data:ro`, `./io:/io` (CSV inputs/outputs, audit logs), `./cache:/app/cache` (template cache, safely discardable).

Full operational guide (build, run modes, flag tables, test harness): `Imputation-Module/docker-imputation/README.md`.

## 11. Adapting the project to your use case

The table below points at the single file (or two) you need to touch for the most common adaptations. Every path is relative to the repo root.

| Want to change                                              | Touch                                                                                                  |
|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Data source (Postgres, InfluxDB, Parquet, anything other than Cassandra) | `Imputation-Module/src/cassandra_client.py`. Keep the three function signatures (`load_historical_data_*`, `load_weather_data_*`, `write_reconstructed_window`) and swap the implementations. `impute_cli.py` only calls these three functions for data access. |
| The list of buildings (add, remove, rename)                 | `Imputation-Module/src/config.py`: `BUILDINGS`, `CAMPUS_COMPONENTS`, `BUILDING_TO_RECONSTRUCTED_COLUMNS`. Also update `Imputation-Module/cassandra/schema.cql` to add the matching value and quality columns. |
| Window size (not 7 days / 1008 points)                      | `Imputation-Module/src/config.py`: `LOOKBACK_DAYS`. Then re-check `window.extract_window()`'s assumptions, `EXPECTED_ROWS` in `impute_cli.py`, and the `_PREPEND_DAYS = 56` constant in `imputer.py` (donor-day needs enough history; scale accordingly). |
| Sampling frequency (not 10 min)                             | `config.py`: `POINTS_PER_DAY`, `FREQ`. Then search the codebase for `"10min"` and `600` (seconds) to catch hard-coded spots (`impute_cli.py:EXPECTED_FREQ_SECONDS`, grid-building calls in `window.py` and `smart_imputation.py`). |
| Gap-length thresholds (30 min, 6 h)                         | `Imputation-Module/src/smart_imputation.py`: the routing logic in `_intelligent_router()` and the per-strategy gate conditions. |
| Which strategies are enabled                                | `Imputation-Module/src/imputer.py`: the `ExtendedDeploymentAlgorithm(...)` constructor call (`use_knn`, `use_mice`, `use_kalman`, `use_multi_week_templates`, `use_chunked_recovery`, `template_lookback_days`). |
| Quality-flag mapping                                        | `Imputation-Module/src/imputer.py`: `_STRATEGY_FLAG_MAP`. Align with downstream consumers: the reconstructed-table schema writes one int per building. |
| Scheduling (not APScheduler, not nightly, cron/Airflow/systemd instead) | `Imputation-Module/src/scheduler.py`. Or drop it entirely and have your external scheduler invoke `impute_cli.py` directly; the CLI is the real entry point, the daemon is just a loop around it. |
| Output destination (somewhere other than the reconstructed Cassandra table) | `Imputation-Module/src/cassandra_client.py:write_reconstructed_window()`, plus `impute_cli.py`'s `--output` handling. CSV output is always produced as a fallback. |
| Cassandra schema (add columns, change names)                | `Imputation-Module/cassandra/schema.cql`, and keep `BUILDING_TO_RECONSTRUCTED_COLUMNS` in `config.py` in lockstep. |
| Test-harness gap injection                                  | `Imputation-Module/src/impute_cli.py`: `apply_test_gaps()` and the `--test-gap` / `--test-report` flags. |
| Downstream consumer / publisher                             | Out of scope for this repo. This module writes to `conso_historiques_reconstructed`; whatever reads that table and does further processing is a separate concern. |

Things to keep in lockstep when you adapt: `config.BUILDINGS` stays in sync with `BUILDING_TO_RECONSTRUCTED_COLUMNS` and `schema.cql`; `config.POINTS_PER_DAY * config.LOOKBACK_DAYS` must equal `config.LOOKBACK_POINTS` and `impute_cli.EXPECTED_ROWS`; `config.FREQ` should match `impute_cli.EXPECTED_FREQ_SECONDS` and the hard-coded `"10min"` in `window.py` and `smart_imputation.py`; and `config.TIMEZONE` should match Docker's `TZ` environment variable.

## 12. Conventions and constants

Constants, all in `Imputation-Module/src/config.py`:

| Constant                  | Value                  | Meaning                                      |
|---------------------------|------------------------|----------------------------------------------|
| `POINTS_PER_DAY`          | `144`                  | 10-minute samples per day                    |
| `LOOKBACK_DAYS`           | `7`                    | Sliding-window length                        |
| `LOOKBACK_POINTS`         | `1008`                 | `POINTS_PER_DAY * LOOKBACK_DAYS`             |
| `FREQ`                    | `"10min"`              | pandas offset alias used for reindexing      |
| `TIMEZONE`                | `"Europe/Paris"`       | Local timezone for feature engineering       |
| `_PREPEND_DAYS`           | `56` (in `imputer.py`) | Donor-day history length                     |

Timestamp handling. Cassandra stores timestamps in UTC; all reads convert to Europe/Paris for window extraction and feature engineering. Output timestamps written back to Cassandra are UTC, and the output CSV preserves the input's representation. DST transitions require special care: `smart_imputation._detect_dst_events()` tags spring-forward and fall-back days so weekly templates do not misalign by one hour. If you move this module to a non-DST timezone you can safely strip that logic out; if you move to a different DST-observing zone, you need to update the timezone rules and re-verify.

Column name mapping. Phase-1 research scripts (not in this repo) used different column names. The canonical names are the Cassandra ones used throughout this module:

| Historical (phase-1) | Current (Cassandra and this module) | Building                |
|----------------------|-------------------------------------|-------------------------|
| `HA`                 | `Ptot_HA`                           | Hôtel Académique        |
| `HEI1`               | `Ptot_HEI_13RT`                     | 13 Rue de Toul          |
| `HEI2`               | `Ptot_HEI_5RNS`                     | 5 Rue Nicolas Souriau   |
| `RIZOMM`             | `Ptot_RIZOMM`                       | RIZOMM                  |
| `Campus`             | `Ptot_Campus`                       | Sum of the four         |
| `DateTime`           | `Date`                              | Timestamp column        |

The Cassandra `conso_historiques_clean` table also contains a `Ptot_HEI` column; it is a legacy duplicate of one of the HEI buildings and is not consumed by this module.

Dependency pinning. Every `requirements.txt` in this repo uses exact `==` version pins (no `>=`, no `~=`, no unpinned names). This is a non-negotiable reproducibility rule for the whole project; when you add a dependency, pin it and pin its transitive closure (use `pip freeze` or `pip-compile`).

## 13. Glossary

- Module de pilotage: the French name for the imputation module; literally "steering module". Referenced in project docs and historical context.
- Donor day: a past day whose consumption shape is used as a template to fill a long gap. Selected from the 56-day history based on similarity (weekday, weather, occupancy).
- Donor-day ensemble: an aggregate (weighted mean) over multiple donor days rather than a single one, reducing sensitivity to any one day's anomalies.
- 56-day history: the rolling window of history prepended to the 7-day target window before imputation, so that long-gap strategies have enough material. `_PREPEND_DAYS` in `imputer.py`.
- Sliding window: the 7-day (1008-point) window the downstream consumer requires.
- Quality flag: the per-point integer (0 to 3) tagging how a value was obtained. See section 5.
- Demo scope: the current deliberately-narrow validation, one building (`Ptot_HA`), CSV-based test harness, no multi-building evaluation yet.
- `HA`, `HEI_13RT`, `HEI_5RNS`, `RIZOMM`, `Campus`: short building identifiers. `Campus` is the sum of the four physical buildings.

## 14. Further reading

- `README.md`: day-to-day operational reference: CLI flag tables, CSV I/O contracts, Docker quick-start.
- `Imputation-Module/docker-imputation/README.md`: Docker-specific deployment, volume and network layout, `run-all-buildings.sh` usage.
- `Imputation-Module/cassandra/schema.cql`: exact DDL for the output table.
- `Imputation-Module/src/config.py`: single source of truth for constants and defaults; skim it when diagnosing configuration surprises.
