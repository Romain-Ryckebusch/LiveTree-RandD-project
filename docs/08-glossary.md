# 8. Glossary

Terms, acronyms, and internal labels used across the project. Each entry points at where it actually matters.

## Project and domain terms

**Project N°28.** The JUNIA research project this module belongs to: *"Amélioration de la résilience des modèles prédictifs par la qualité et la continuité des données"* ("Improving the resilience of predictive models through data quality and continuity"). See [`01-overview.md`](01-overview.md).

**LiveTree demonstrator.** The physical setup at JUNIA campus in Lille: four instrumented buildings, a Cassandra cluster, and the day-ahead forecasting service this module feeds into.

**JUNIA.** The French engineering school hosting the demonstrator (Lille, France).

**Module de pilotage.** French label for this project; literally "steering module". The imputation service steers the forecaster's input-data quality. Appears in historical documents and occasionally in code comments.

**Demo scope.** Phrase used in earlier docs to describe the fact that the current test harness validates only `Ptot_HA` in CSV mode. Cassandra mode seeds history for every building; CSV mode assumes the input is a `Ptot_HA`-shaped series.

## Pipeline terms

**Sliding window.** The 7-day (1008-point) window the downstream forecaster requires. Every reconstruction produces exactly one such window per building per run. See [`03-data-model.md`](03-data-model.md#33-cli-csv-io-contract---source-csv).

**56-day history.** The rolling window of past data prepended to the 7-day target window before imputation, so the long-gap strategies have enough context. Controlled by `_PREPEND_DAYS = 56` in `imputer.py`. See [`02-architecture.md`](02-architecture.md#242-56-day-history-prepend).

**Gap.** A consecutive run of NaN values in the target series. Detected by `_find_gap_groups` in `smart_imputation.py`.

**Donor day.** A past day whose consumption shape is used as a template to fill a long gap. Selected from the 56-day history by similarity in `(day-of-week, thermal regime, occupancy)`.

**Donor-day ensemble.** A weighted mean over multiple donor days rather than a single one, reducing sensitivity to any one day's anomalies. Emitted by `_fill_with_multi_week_template`. See [`06-algorithms.md`](06-algorithms.md#6311-multi-week-template-_fill_with_multi_week_template).

**Template.** A median profile over historical values, indexed by some combination of `(day-of-week, hour-of-day, thermal regime, occupancy)`. Built in `_build_day_specific_templates`, `_build_seasonal_templates`, and `_build_weekly_templates`.

**Anchoring.** The post-fill step that blends a filled segment into the real values at its endpoints so no discontinuity appears. `_anchor_to_boundaries` in `smart_imputation.py`. See [`06-algorithms.md`](06-algorithms.md#65-boundary-anchoring-_anchor_to_boundaries).

**Routing trace.** The ordered list of decisions (`BRANCH: ...`, `TRY: ...`, `FAIL: ...`, `SELECTED: ...`) recorded per gap in the audit log. Makes it possible to replay why a given strategy was chosen. See [`06-algorithms.md`](06-algorithms.md#68-audit-logging).

**Audit log.** Per-run JSON dump of every gap, strategy, confidence, routing trace, and post-processing parameter. Written to `/io/audit_logs/<timestamp>.json`. See [`06-algorithms.md`](06-algorithms.md#68-audit-logging).

## Quality flags

**Quality flag.** The per-point integer (0 to 3) tagging how a value was obtained. Authoritative mapping in `imputer._STRATEGY_FLAG_MAP`; reproduced in [`03-data-model.md`](03-data-model.md#314-quality-flag-encoding). The values are `0` for a real measurement (not imputed), `1` for linear interpolation (`LINEAR_MICRO`, `LINEAR_SHORT`), `2` for contextual, template, or peer fills, and `3` for donor-day ensemble or ML fallback.

**ML cascade.** The sequence `MICE` then `Kalman` then `KNN` tried first for long gaps on non-entry meters. All three emit flag 3. See [`06-algorithms.md`](06-algorithms.md#62-routing-decision-tree).

## Building identifiers

The five columns that go through the pipeline:

| Current (canonical) | Short (phase-1) | Meaning | Tier |
|---|---|---|---|
| `Ptot_HA` | `HA` | Hôtel Académique | entry |
| `Ptot_HEI_13RT` | `HEI1` | 13 Rue de Toul (HEI annex) | sub (parent `Ptot_HEI`) |
| `Ptot_HEI_5RNS` | `HEI2` | 5 Rue Nicolas Souriau (HEI annex) | sub (parent `Ptot_HEI`) |
| `Ptot_RIZOMM` | `RIZOMM` | RIZOMM building | entry |
| `Ptot_Campus` | `Campus` | Virtual sum of the four above | (virtual) |

And two legacy aggregates that appear in the data but are not consumed.

**`Ptot_HEI`.** Legacy aggregate of `Ptot_HEI_13RT` plus `Ptot_HEI_5RNS`. Present as a column in `conso_historiques_clean`; referenced only as the `parent` in `METER_HIERARCHY`; never read as a data source by this module.

**`Ptot_Ilot`.** Legacy name for `Ptot_Campus` in the downstream forecaster's output schema (outside this repo). *Îlot* is French for "block", as in "city block".

Phase-1 research scripts (not in this repo) used the short names (`HA`, `HEI1`, and so on) and `DateTime` as the timestamp column. The current canonical names are the `Ptot_*` columns and `"Date"`.

## Timing and deployment terms

**Nightly tick.** The scheduled run at 23:50 Europe/Paris (default `SCHEDULE_HOUR=23`, `SCHEDULE_MINUTE=50`) when the daemon reconstructs the next day's input window.

**Target date.** The date for which the forecaster will produce a prediction. The imputer's 7-day window ends the day before `target_date`.

**Partition key.** In Cassandra, the constant string used to group rows into a single storage partition. This module uses `"Conso_Data"` for consumption tables and `"Meteorological_Prevision_Data"` for weather.

## Time zones

**UTC.** What Cassandra stores. All `"Date"` columns in `conso_historiques_clean`, `pv_prev_meteo_clean`, and `conso_historiques_reconstructed` are UTC timestamps.

**Europe/Paris.** What the module reasons in: window extraction, template indexing, DST handling, APScheduler cron. The conversion boundary is `window.extract_window` / `imputer._to_window_time`.

**DST.** Daylight Saving Time. Europe/Paris shifts twice a year, and `_detect_dst_events` in `smart_imputation.py` records the transitions. See [`06-algorithms.md`](06-algorithms.md#67-dst-handling-_detect_dst_events-and-_localise_timestamps).

## Algorithm acronyms

**MICE.** Multiple Imputation by Chained Equations. `_fill_with_mice`.

**KNN.** K-Nearest Neighbours. `_fill_with_knn_context` uses `sklearn.neighbors.NearestNeighbors`.

**Kalman.** Kalman filter for state-space smoothing. `_fill_with_kalman_filter` uses `pykalman`.

**IsolationForest.** Tree-based anomaly detector. Used pre-fill (`_detect_raw_anomalies`) and post-fill (`_detect_zero_fills`) to flag implausible values. `sklearn.ensemble.IsolationForest`.

**Savitzky-Golay.** Polynomial-smoothing filter. Used by `_smooth_junctions` to remove second-derivative discontinuities at fill-segment boundaries. `scipy.signal.savgol_filter`.

**MAE and RMSE.** Mean Absolute Error and Root Mean Squared Error. Summary metrics in the test-harness report.

## Infrastructure acronyms

**TGBT.** *"Tableau Général Basse Tension"*, the main low-voltage distribution board. Appears in legacy file names like `Rizomm_TGBT_Puissances_Ptot`, the physical point at which `Ptot_RIZOMM` is metered.

**APScheduler.** Python scheduling library (`APScheduler==3.6.3` pinned). Wraps cron-style triggers around the nightly batch.

**CQL.** Cassandra Query Language. SQL-like dialect used in `schema.cql` and the queries in `cassandra_client.py`.

**DDL.** Data Definition Language. The `CREATE TABLE` and `ALTER TABLE` statements in `schema.cql`.

## File-path suffix conventions

`reconstructed_<building>_<target_date>.csv` is the output series for one building, one run. `reconstructed_<building>_<target_date>.png` is the optional overlay plot. `reconstructed_<building>_<target_date>_test_report.csv` is the test-harness per-point comparison (produced when `--test-gap` is set). Files named `<timestamp>.json` under `/io/audit_logs/` are per-run audit logs.

## What is not in this project

Listed so nobody wastes time looking for them.

The forecasting model itself is not here. It lives in a different repository and runs as a separate container; this module only produces its input.

There is no Kafka producer here. References to a `CONSO_Prevision_Data` topic in earlier docs belonged to the downstream forecaster, not to this module.

There is no `Prediction-Model/` folder at the repo root. It was referenced in the previous documentation but does not exist.

There is no REST or HTTP API. Integration is via CSV files, direct Cassandra reads and writes, or Python import; see [`04-usage.md`](04-usage.md#45-python-import).

There is no alerting path for sensor failures. The module logs anomalies to the audit JSON but does not page anyone.

There is no automated retraining on reconstructed data. This is intentional: the forecaster is trained on raw measurements only, to avoid a feedback loop.
