# 5. File reference

A walk-through of every file under `Imputation-Module/`, in the order a reader tracing an imputation run would touch them. For each file: what it does, what it exposes, where the interesting bits live, and when you would edit it.

This is a structural map, not an API reference. For the algorithm itself (how the strategies actually work), see [`06-algorithms.md`](06-algorithms.md).

## 5.1 `src/config.py`

72 lines. The single source of truth for constants and environment-driven settings. Every other module under `src/` imports from here.

The file defines paths (`PROJECT_ROOT`, `DATA_DIR`, `HISTORICAL_CSV`, `WEATHER_CSV`, `RECENT_HA_CSV`, `OUTPUT_DIR`, `AUDIT_LOG_DIR`), all overridable via env vars (`IMPUTER_DATA_DIR`, `IMPUTER_RECENT_HA_CSV`, `IMPUTER_OUTPUT_DIR`, `IMPUTER_AUDIT_LOG_DIR`).

It holds the building list: `BUILDINGS` (five columns), `BUILDING_COLUMN` (the default, `Ptot_HA`), and `CAMPUS_COMPONENTS` (the first four, summed to make `Ptot_Campus`).

Grid constants live here: `POINTS_PER_DAY = 144`, `LOOKBACK_DAYS = 7`, `LOOKBACK_POINTS = 1008`, `FREQ = "10min"`, `TIMEZONE = "Europe/Paris"`.

The Cassandra connection settings are `CASSANDRA_HOSTS`, `CASSANDRA_USERNAME`, `CASSANDRA_PASSWORD`, `CASSANDRA_KEYSPACE` (all env-driven), plus the table and partition-key names: `CONSO_TABLE`, `CONSO_PARTITION_KEY`, `METEO_TABLE`, `METEO_PARTITION_KEY`, `RECONSTRUCTED_TABLE`, `RECONSTRUCTED_PARTITION_KEY`.

`BUILDING_TO_RECONSTRUCTED_COLUMNS` is the `{building_id: (value_column, quality_column)}` dict used by `cassandra_client.write_reconstructed_window` to build its per-building `INSERT`. It must stay in lockstep with `cassandra/schema.cql`.

The low-variance thresholds `LOW_VARIANCE_AUTO_FRACTION = 0.05` and `LOW_VARIANCE_FLOOR_W = 100.0` are used by the anomaly detector in `smart_imputation`.

**When to edit.** Any change that adds, removes, or renames a building; changes the Cassandra cluster or keyspace; or moves data directories. Most of these changes need parallel edits in `schema.cql` and downstream consumers; see [`07-extending.md`](07-extending.md).

## 5.2 `src/cassandra_client.py`

125 lines. Owns all Cassandra I/O through three top-level functions, plus a private session factory.

`_get_session()` creates a `cassandra.cluster.Cluster`, connects to `CASSANDRA_KEYSPACE`, and attaches a pandas `row_factory` so queries return DataFrames directly. Anonymous if `CASSANDRA_USERNAME` is empty, `PlainTextAuthProvider` otherwise. Returns `(session, cluster)` so the caller can shut down the cluster in `finally`.

`load_historical_data_cassandra()` runs `SELECT * FROM conso_historiques_clean WHERE name = 'Conso_Data'`, drops the legacy `Quality` column and the `name` partition-key column, and raises `RuntimeError` if the partition is empty.

`load_weather_data_cassandra()` runs `SELECT * FROM pv_prev_meteo_clean WHERE name = 'Meteorological_Prevision_Data'`, drops `name`, and raises on an empty partition.

`write_reconstructed_window(building, timestamps, values, quality)` prepares an `INSERT` that names only the current building's value column and quality column (from `BUILDING_TO_RECONSTRUCTED_COLUMNS`), then executes it once per row (1008 rows per call).

This file is the entire data-source seam. To swap Cassandra for another store (Postgres, Parquet, InfluxDB), keep the three function signatures and rewrite the bodies; nothing else in the codebase needs to change. See [`07-extending.md`](07-extending.md#73-swap-the-data-source).

**When to edit.** Switching data sources; adding credentials, TLS, or auth mechanisms; changing the queries (for example, pushing the partition slice into CQL instead of slicing in Python).

## 5.3 `src/window.py`

82 lines. One public function.

`extract_window(df, target_date, weather_df, building_column=None)` slices the 7 days ending the day before `target_date` out of a full-history DataFrame, converts UTC to Europe/Paris, and reindexes onto a complete 10-minute grid so missing Cassandra rows appear as explicit NaNs. It also extracts the weather window for the target day and, if present in the history, the "actual" target values (useful for overlay plots). Returns `(history, target_timestamps, weather_temps, actual_target)`.

The function accepts both naive (assumed UTC) and tz-aware `Date` columns. It fills in missing weather rows with a constant 15 °C if the weather pull is short or empty, so the imputer does not crash downstream. And it prints `Detected N missing timestamps in 7-day window (real gaps)` when the reindex introduces NaNs, which is the smoke signal that the window has gaps.

**When to edit.** Changing the window size or the sampling frequency; changing the timezone conversion rules; changing the fallback weather temperature.

## 5.4 `src/smart_imputation.py`

1 964 lines. The engine. One big class (`ExtendedDeploymentAlgorithm`) plus a handful of module-level helpers.

### Module-level

`SITE_COLS`, `STEPS_PER_DAY`, and `SITE_TZ` are module constants. Note that `SITE_COLS` lists six identifiers (and includes legacy `Ptot_HEI` and `Ptot_Ilot`); the live five-building list is in `config.BUILDINGS`.

`METER_HIERARCHY` is a dict mapping each site to its parent, children, and tier (`entry`, `sub`, or `main`). Used by the peer-correlation strategy.

`_load_all_holidays()` reads `Consumption_<year>_Holiday.csv`, `_Close.csv`, and `_Special.csv` from `IMPUTER_DATA_DIR` for years 2021 through 2026, plus a hard-coded 2026 holiday list. Returns three sets of `date` objects. Called once during engine construction.

`_ts(dt)` and `_json_safe(obj)` are audit-log helpers (timestamp formatting, numpy-to-JSON coercion).

### `class ExtendedDeploymentAlgorithm`

Main public surface:

| Method | Purpose |
|---|---|
| `__init__(site_cols=..., use_knn=..., use_mice=..., use_kalman=..., use_multi_week_templates=..., use_chunked_recovery=..., template_lookback_days=..., audit_log_dir=..., timezone=...)` | Construct the engine. Feature flags enable or disable individual strategies. |
| `impute(df, weather_df=None)` | The entry point. Walks every gap in `df`, routes to a strategy, fills, anchors, smooths, and returns a NaN-free DataFrame. Populates `strategy_log` as a side effect. |
| `save_templates(filepath)` / `load_templates(filepath)` | Pickle and unpickle the precomputed weekly, thermal, and seasonal templates. Present on the class but not called anywhere in the shipped code: nothing currently persists or reloads the cache across runs. Kept as a hook for callers that want to manage this themselves. |
| `get_audit_log()`, `get_run_summary()`, `get_gap_log()`, `get_zero_fill_report()`, `get_raw_anomaly_report()`, `get_low_confidence_report()`, `get_strategy_log()` | Inspect what happened during the last `impute()` call. Used by the test harness and by `imputer._flag_for_strategy`. |

Internal structure, roughly top-to-bottom in the file:

Timezone and DST handling sit in `_localise_timestamps` and `_detect_dst_events` (line 272). Audit logging is `_init_audit_log`, `_record_gap`, `_finalise_audit_log` (lines 321, 373, 476). Anomaly detection is `_detect_raw_anomalies` (line 539) and `_detect_zero_fills` (line 736, IsolationForest-based).

The main loop is `impute` (line 860): it detects gaps, calls `_intelligent_router` for each, dispatches to a strategy, and post-processes via `_anchor_to_boundaries`, `_smooth_junctions`, `_validate_and_clip`, and `_nan_guard_final_pass`.

Routing is `_intelligent_router` (line 1379): it picks a strategy from gap length, day type, peer availability, and DST status.

Strategies live in roughly lines 1474 to 1616: `_fill_linear`, `_fill_weekend_template`, `_fill_with_thermal_template`, `_fill_enhanced_template`, `_fill_safe_linear_median`, `_fill_safe_median_template`, `_fill_via_peer_correlation`, `_fill_with_mice`, `_fill_with_kalman_filter`, `_fill_with_knn_context`. The long-gap donor-day strategy is `_fill_with_multi_week_template` (line 1182). `_fill_chunked_gap` and `_get_smart_chunks` (lines 1022 and 1059) split very long gaps for recovery.

Post-processing: `_anchor_to_boundaries` (line 1616) blends the filled segment into the real values at both ends; `_smooth_junctions` (line 1805) runs Savitzky-Golay smoothing at junctions; `_validate_and_clip` (line 1785) does range checks.

Feature engineering: `_add_occupancy_features`, `_add_external_features`, `_add_datetime_features`, `_fill_airtemp_forward`, `_merge_weather`, `_classify_thermal_regimes`.

Template builders: `_build_weekly_templates` (line 1110), `_build_day_specific_templates`, `_build_seasonal_templates`, `_build_peer_ratios`, `_build_multi_site_correlations`.

Gap utilities: `_find_gap_groups`, `_is_weekend_gap`, `_is_pure_weekend_gap`, `_get_weekend_day_type`.

Confidence: `_build_uncertainty_bounds`, `_calculate_confidence_with_uncertainty`, `_flag_low_confidence`.

**When to edit.** Changing routing thresholds (the 30-minute and 6-hour bands); adding a new strategy; changing the 56-day template lookback; changing anchoring behaviour. All of these are covered with gotchas in [`07-extending.md`](07-extending.md), and the algorithmic context is in [`06-algorithms.md`](06-algorithms.md).

## 5.5 `src/imputer.py`

283 lines. A single-series adapter around `ExtendedDeploymentAlgorithm`. This is the public Python API: both the CLI and external callers go through here, not directly through `smart_imputation`.

`_STRATEGY_FLAG_MAP` (line 65) is the authoritative mapping from internal strategy names to quality flag values (1, 2, or 3). Unknown strategies default to 2 via `_flag_for_strategy`. This is the table reproduced in [`03-data-model.md`](03-data-model.md#314-quality-flag-encoding).

`_PREPEND_DAYS = 56` is the history-prepend length. Changing it changes how much context the long-gap donor-day ensemble has access to.

`_HISTORY_CACHE` is the per-building history DataFrame cache, populated either by `_load_combined_history` (from reference CSVs, only for `Ptot_HA`) or by `set_history_source` (called by the CLI in Cassandra mode). `_WEATHER` is the weather DataFrame loaded from `WEATHER_CSV` on first call. `_LAST_STRATEGY_LOG` and `_LAST_EXTENSION_LEN` are module globals holding the most recent call's state for the `get_last_strategy_log()` accessor used by the test harness.

Public API:

`impute(series, date_index, quality=None, *, large_gap_min=144, random_seed=None, building_column=None, **_kwargs) -> (values, quality_flags)` is the main function. It prepends up to 56 days of history, calls `ExtendedDeploymentAlgorithm.impute()`, reprojects the result back onto the caller's 1008-row series, translates the strategy log into per-point quality flags, and force-overwrites flags for originally-non-NaN rows to 0.

`set_history_source(df, building_column)` injects a pre-loaded history. Used by `impute_cli.load_cassandra_window`.

`get_last_strategy_log()` returns the strategy log (with indices shifted back to the caller's series) from the most recent `impute()` call.

`naive_impute(series, method="linear")` is the zero-fill baseline: returns `(series.fillna(0.0), quality_flags)`. Useful as a comparison point in evaluations. The `method` argument is accepted but currently ignored; zero-fill is the baseline.

**When to edit.** Changing the 56-day prepend; changing quality-flag values or strategy-to-flag associations; changing how the history cache is populated; adding a new public entry point.

## 5.6 `src/impute_cli.py`

548 lines. The command-line entry point. Every per-building run goes through `main()`.

Top-level constants are `EXPECTED_ROWS = 1008`, `EXPECTED_FREQ_SECONDS = 600`, and `FREQ_TOLERANCE_SECONDS = 1`. Those are the CSV validation constants.

Key functions:

`main()` sets up argparse, dispatches to the CSV or Cassandra loader, optionally applies test-gap masks, calls `impute()`, writes the output CSV, writes to Cassandra (Cassandra mode only), renders the plot (if `--plot`), and writes the test report (if `--test-report`).

`load_input(path)` is the CSV-mode input loader and validator. It enforces `EXPECTED_ROWS`, the 10-minute grid, strict increasing timestamps, and at least one non-NaN value.

`load_cassandra_window(target_date, building_column, include_prior_week=False)` is the Cassandra-mode input loader. It pulls the full history and weather partitions, materialises `Ptot_Campus` from the four components with `skipna=False` if requested, calls `window.extract_window`, seeds `imputer.set_history_source`, and optionally fetches the prior-week window for `--overlay-prior-week`.

`parse_test_gap(raw_start, raw_end)` parses a single `--test-gap` pair.

`apply_test_gaps(df, gaps, value_col="value", source="csv")` masks rows in each gap range with NaN in memory. It handles naive vs tz-aware timestamps, rejects masks fully outside the window, clips partially-outside masks with a warning, and rejects full-window masks in CSV mode.

`write_test_report(path, timestamp_strings, ground_truth, imputed, quality, mask)` emits the per-point comparison CSV plus the MAE, RMSE, and max footer and the `# strategy=...` / `# postproc.*` header lines harvested from the last strategy log.

`_strategy_lines_for_mask(mask)` is a helper that walks `imputer.get_last_strategy_log()` and emits one `# strategy=...` line per gap that overlaps the test mask.

`fail(msg)` is the uniform error printer that exits 1.

Argparse flags are all documented in [`04-usage.md`](04-usage.md) with required/optional semantics and examples.

**When to edit.** Adding a new CLI flag; changing CSV validation rules; changing the test-harness report format; adding a new input source.

## 5.7 `src/scheduler.py`

250 lines. The long-running daemon wrapper around `impute_cli.py`.

Top-level constants: `IO_DIR = "/io"` is the artefact directory; `RECONSTRUCTION_GLOBS` are the file patterns wiped at the start of each run (stale CSVs and PNGs); `CLI_SCRIPT` is the absolute path to `impute_cli.py`. `SCHEDULE_HOUR` and `SCHEDULE_MINUTE` are parsed from env vars via the `_env_int` helper, which validates range and type at startup.

Key functions:

`main()` handles argparse for one-shot (`--run-now`) or daemon mode.

`run_daily_imputation(with_plots=False, overlay_prior_week=False, overlay_actual=False, test_gaps=None)` is the batch runner. It computes `_tomorrow_iso()`, wipes stale outputs, and subprocesses `impute_cli.py` once per building in `config.BUILDINGS`. It captures and logs each subprocess's stdout and stderr, and returns the list of buildings that failed.

`_tomorrow_iso()` returns tomorrow's date in Europe/Paris as an ISO string, passed as `--target-date`.

`_wipe_previous_outputs()` deletes `/io/reconstructed_*.csv` and `*.png` from the previous run.

`_env_int(name, default, lo, hi)` parses an env var into an int with range validation. Exits with a clear message on bad input.

The daemon loop is an APScheduler `BlockingScheduler` with a single `CronTrigger` job (`misfire_grace_time=3600`, `coalesce=True`) so one missed tick does not double-fire the next run.

**When to edit.** Changing the schedule time's semantics; adding a new flag to propagate to per-building CLI calls; replacing APScheduler with an external scheduler (in which case, delete most of this file; see [`07-extending.md`](07-extending.md)).

## 5.8 `src/plot_reconstruction.py`

116 lines. Matplotlib PNG renderer for `--plot`. Headless (`matplotlib.use("Agg")`).

`QUALITY_LABELS` is `{flag: (label, color)}`, the colour legend: blue for real (0), green for linear (1), orange for contextual (2), red for donor-day (3).

`render(csv_path, png_path, building_column, masked_ranges=None, prior_week_values=None, actual_values=None)` reads the output CSV and builds the overlay in five layers, back to front. First, translucent grey `axvspan` bands for any `masked_ranges` (from `--test-gap`). Then a grey line of the reconstructed series (quality-blind shape). Then a dashed purple line of the prior-week values (if `--overlay-prior-week`). Then a solid black line of the actual pre-imputation values (if `--overlay-actual`). Finally, a per-flag coloured scatter of all points.

`_to_paris_naive(ts)` is a timestamp normalisation helper for axvspan bounds.

**When to edit.** Changing plot colours or styling; adding a new overlay; changing the title format; changing DPI or size.

## 5.9 `cassandra/schema.cql`

22 lines. DDL for the output table `conso_historiques_reconstructed`. Apply once per cluster before enabling `--source cassandra` writes.

The schema is documented in detail in [`03-data-model.md`](03-data-model.md#313-conso_historiques_reconstructed-output). Short version: one row per `(name, "Date")`, one `Ptot_*` value column and one `quality_*` column per building, so per-building runs can upsert their own columns without touching siblings'.

**When to edit.** Any change to `config.BUILDINGS` or `BUILDING_TO_RECONSTRUCTED_COLUMNS`. These three must stay in lockstep: `BUILDINGS` and `BUILDING_TO_RECONSTRUCTED_COLUMNS` keys and `schema.cql` column names.

## 5.10 `docker-imputation/Dockerfile`

25 lines. `FROM python:3.7-slim-bookworm`.

The image installs `gcc` (required to build pinned old wheels of `numpy==1.19.0`, `scipy==1.5.4`, and so on), then runs `pip install --no-cache-dir -r requirements.txt` and copies `Imputation-Module/src/` into `/app/pkg/`. It pre-creates `/app/cache`, `/io`, `/io/audit_logs`, and `/data`, sets `PYTHONPATH=/app/pkg`, and sets the imputer env vars (`IMPUTER_DATA_DIR=/data`, `IMPUTER_RECENT_HA_CSV`, `IMPUTER_OUTPUT_DIR=/app/cache`, `IMPUTER_AUDIT_LOG_DIR=/io/audit_logs`).

The entry point is `python` and the default command is `scheduler.py`. That is why `docker compose up` runs the daemon while `docker compose run imputer impute_cli.py ...` runs one-shot CLI calls.

**When to edit.** Upgrading Python (everything is tested on 3.7; moving up needs dependency rebumps); changing container paths; pre-setting additional env vars.

## 5.11 `docker-imputation/docker-compose.yml`

31 lines. Defines a single service `imputer`.

`build.context: ../..` sets the build context to the repo root so the Dockerfile can `COPY` from both `Imputation-Module/src/` and `Imputation-Module/docker-imputation/requirements.txt`. `image: rd-imputer:latest` plus `container_name: rd_imputer` sets the container name used in logs and `docker ps`.

Three volume mounts are declared: `../../data:/data:ro`, `./io:/io`, `./cache:/app/cache`.

`environment:` forwards `CASSANDRA_*`, `TZ=Europe/Paris`, `SCHEDULE_HOUR`, and `SCHEDULE_MINUTE` from the host environment with sensible defaults. `env_file: path: .env, required: false` pulls a `.env` next to `docker-compose.yml` if present and does not fail if it is absent.

`networks: cassandra_net: external: true, name: cassandra_default` joins an external network. `restart: unless-stopped` means the daemon auto-restarts on crash but stops when a human stops it.

**When to edit.** Changing volume mounts; changing the attached network; adding new env vars; changing the base image tag.

## 5.12 `docker-imputation/requirements.txt`

11 lines. Pinned dependencies, all with `==` (no `>=`, no `~=`):

```
numpy==1.19.0
pandas==1.0.5
pytz==2020.1
scipy==1.5.4
scikit-learn==0.23.1
joblib==0.16.0
pykalman==0.9.5
cassandra-driver==3.24.0
matplotlib==3.3.4
APScheduler==3.6.3
```

Strict pinning is a reproducibility rule for the whole project; see [`07-extending.md`](07-extending.md#710-dependency-pinning). When adding a dependency, pin it and its transitive closure.

**When to edit.** Adding a new library; bumping a pinned version (which requires a rebuild and a regression pass against the test harness).

## 5.13 `docker-imputation/run-all-buildings.sh`

125 lines. Bash wrapper that loops `impute_cli.py --source cassandra` over all five buildings for a given `TARGET_DATE` (default: today UTC).

The positional `TARGET_DATE` argument (first non-flag argument) defaults to `date -u +%Y-%m-%d`. The flags are `--test-gap START END` (repeatable), `--overlay-prior-week`, `--overlay-actual`, and `--no-clear`.

When `--test-gap` is set, the output filenames are suffixed `_test` so clean reconstructions are not overwritten, and `--test-report /io/test_report_<b>_<date>.csv` is added per-building.

`set -euo pipefail` at the top means the script aborts on the first failing building. This is different from `scheduler.py --run-now`, which logs failures and continues.

`--no-clear` keeps the previous run's artefacts in `./io/`; without it, `reconstructed*.csv`, `reconstructed*.png`, and `test_report_*.csv` are wiped at the start.

**When to edit.** Changing which buildings are iterated (hard-coded `BUILDINGS` array near the top); changing the per-building filename pattern; changing wipe behaviour.

## 5.14 Runtime-only directories (not in git)

`docker-imputation/io/` is the `/io/` mount point, holding inputs, outputs, PNGs, and per-run audit JSON.

`docker-imputation/cache/` is the `/app/cache/` mount point, reserved for persistent template caches. The shipped code does not currently write here (the engine has `save_templates` and `load_templates` methods but nothing calls them). The directory is kept because the volume is wired in `docker-compose.yml` and `IMPUTER_OUTPUT_DIR` points at it.

`data/` at the repo root holds reference CSVs; it is mounted into the container as `/data:ro`. The folder exists in fresh clones but the CSV files themselves are gitignored, to avoid committing measurement data.
