# 3. Data model

The module reads and writes three kinds of storage: Cassandra tables, reference CSVs on disk, and the output CSV produced by each run. This chapter documents the schema and semantics of each.

All timestamps in storage are UTC. The module converts to Europe/Paris for wall-clock reasoning (window slicing, DST handling, template day-of-week indexing) and back to UTC before writing. The conversion boundary is explicit and centralised in `window.extract_window` and `imputer._to_window_time`.

## 3.1 Cassandra, keyspace `previsions_data`

Three tables matter to this module: two inputs (read-only) and one output (write-only). The authoritative DDL for the output table is in [`../Imputation-Module/cassandra/schema.cql`](../Imputation-Module/cassandra/schema.cql). The input tables are provisioned upstream by the acquisition system.

### 3.1.1 `conso_historiques_clean` (input)

The source of truth for real measurements. Written by the acquisition chain, read by this module.

| Column | Type | Notes |
|---|---|---|
| `name` | `text` | Partition key. Always the literal `"Conso_Data"` for rows the imputer reads. |
| `"Date"` | `timestamp` | Clustering key. UTC, 10-minute grid. |
| `Ptot_HA` | `double` | Hôtel Académique, watts. |
| `Ptot_HEI_13RT` | `double` | HEI 13 Rue de Toul, watts. |
| `Ptot_HEI_5RNS` | `double` | HEI 5 Rue Nicolas Souriau, watts. |
| `Ptot_RIZOMM` | `double` | RIZOMM, watts. |
| `Ptot_HEI` | `double` | Legacy aggregate column; not consumed by this module. |
| `Quality` | any | Upstream quality flag. Dropped on read by `cassandra_client.load_historical_data_cassandra`. |

Query used (in `cassandra_client.load_historical_data_cassandra`):

```cql
SELECT * FROM conso_historiques_clean WHERE name = 'Conso_Data';
```

The module pulls the full partition on every run and then slices the seven-day window in Python (see `window.extract_window`). Sparse rows (timestamps with no measurement) simply do not exist in the table; the `reindex` inside `window.extract_window` turns those absences into explicit NaNs on the full 10-minute grid.

The partition-key constant is defined in `config.py` as `CONSO_PARTITION_KEY = "Conso_Data"`.

### 3.1.2 `pv_prev_meteo_clean` (input)

Meteorological history and short-term forecasts, written by a sibling photovoltaic-forecasting project.

| Column | Type | Notes |
|---|---|---|
| `name` | `text` | Partition key. `"Meteorological_Prevision_Data"` for rows the imputer reads. |
| `"Date"` | `timestamp` | Clustering key. UTC, 10-minute grid. |
| `AirTemp` | `double` | Outside air temperature in °C at the reference station. The only field the imputer uses. |
| `CloudOpacity`, `Dni*`, `Ghi*` | `double` | Radiometric fields, ignored by this module. |

Query used (in `cassandra_client.load_weather_data_cassandra`):

```cql
SELECT * FROM pv_prev_meteo_clean WHERE name = 'Meteorological_Prevision_Data';
```

The partition-key constant is defined in `config.py` as `METEO_PARTITION_KEY = "Meteorological_Prevision_Data"`.

`AirTemp` drives the thermal-regime templates (see [`06-algorithms.md`](06-algorithms.md)). When the weather pull is empty or short, `window.extract_window` falls back to a constant 15 °C so imputation continues.

### 3.1.3 `conso_historiques_reconstructed` (output)

Written by this module only. One row per `(name, "Date")` in UTC, containing the reconstructed value and the quality flag for whichever building the current run is reconstructing.

DDL ([`schema.cql`](../Imputation-Module/cassandra/schema.cql)):

```cql
CREATE TABLE IF NOT EXISTS previsions_data.conso_historiques_reconstructed (
    name              text,
    "Date"            timestamp,
    "Ptot_HA"         double,
    "Ptot_HEI_13RT"   double,
    "Ptot_HEI_5RNS"   double,
    "Ptot_RIZOMM"     double,
    "Ptot_Campus"     double,
    quality_ha        int,
    quality_hei_13rt  int,
    quality_hei_5rns  int,
    quality_rizomm    int,
    quality_campus    int,
    PRIMARY KEY (name, "Date")
);
```

Apply this DDL once per cluster before enabling `--source cassandra` writes (`cqlsh -f schema.cql`).

The partition key is the same string (`"Conso_Data"`) as in `conso_historiques_clean`, so reconstructed rows sit alongside the raw ones in the logical partition. There is one `quality_*` column per building rather than a single global `Quality` column, which is what makes the five independent per-building subprocesses safe to run on the same `(name, Date)` row: each one upserts only its own value column and its own quality column, and siblings are never touched.

The INSERT statement built by `cassandra_client.write_reconstructed_window` looks like:

```cql
INSERT INTO conso_historiques_reconstructed
    (name, "Date", "Ptot_HA", quality_ha)
VALUES (?, ?, ?, ?);
```

The `BUILDING_TO_RECONSTRUCTED_COLUMNS` dict in `config.py` is the authoritative mapping from CLI `--building` argument to `(value_column, quality_column)` pair.

### 3.1.4 Quality-flag encoding

Every `quality_*` column holds an integer 0 to 3. The authoritative mapping from internal strategy names to flag values is `_STRATEGY_FLAG_MAP` in `Imputation-Module/src/imputer.py`:

| Flag | Meaning | Strategies |
|---|---|---|
| `0` | Real measurement (not imputed) | (set by force-overwriting flags on rows whose input was not NaN) |
| `1` | Linear interpolation | `LINEAR_MICRO`, `LINEAR_SHORT` |
| `2` | Contextual / template / peer fill | `THERMAL_TEMPLATE`, `ENHANCED_TEMPLATE`, `WEEKEND_TEMPLATE`, `WEEKEND_TEMPLATE_SATURDAY`, `WEEKEND_TEMPLATE_SUNDAY`, `WEEKEND_TEMPLATE_MIXED`, `SAFE_LINEAR_MEDIAN`, `PEER_CORRELATION`, `HOURLY_MEDIAN_FALLBACK` |
| `3` | Donor-day ensemble or ML fallback | `MULTI_WEEK_TEMPLATE`, `MICE`, `KNN_CONTEXT`, `KALMAN_FILTER`, `SAFE_MEDIAN` |

A strategy name seen at runtime that is not in the map defaults to flag `2` (the safe contextual bucket) via `imputer._flag_for_strategy`. This matters when adding a new strategy; see [`07-extending.md`](07-extending.md).

Which strategy a given gap gets routed to depends on the gap's length, day type, the meter's tier (entry vs sub-meter), and peer availability. The full routing decision tree is documented in [`06-algorithms.md`](06-algorithms.md#62-routing-decision-tree). A rough mnemonic: short gaps (up to about 3 h) get linear, middle gaps (up to about 1 day) get templates, long gaps (above about 8 h on non-entry meters) enter the ML cascade, and pure-weekend gaps always get the weekend template regardless of length.

Downstream consumers typically filter `quality <= 1` for high-confidence analytics, `quality <= 2` when they trust the contextual templates, and treat `quality == 3` as a shape estimate rather than a measurement.

### 3.1.5 `Ptot_Campus`, the virtual channel

`Ptot_Campus` is not stored in `conso_historiques_clean`. The imputer materialises it on the fly inside `impute_cli.load_cassandra_window`:

```python
hist_df["Ptot_Campus"] = hist_df[CAMPUS_COMPONENTS].sum(axis=1, skipna=False)
```

`CAMPUS_COMPONENTS` is `["Ptot_HA", "Ptot_HEI_13RT", "Ptot_HEI_5RNS", "Ptot_RIZOMM"]` (the first four entries of `config.BUILDINGS`).

`skipna=False` is deliberate: if any component is NaN at a given timestamp, the campus total is also NaN for that timestamp and will be routed through the imputer like any other gap. This prevents the aggregate from silently under-counting one component while the others are complete.

`Ptot_Campus` does get written to `conso_historiques_reconstructed` (the output table has a `"Ptot_Campus"` column and a `quality_campus` column) so downstream consumers can read the materialised total directly.

## 3.2 Reference CSVs on disk

Files under `data/` at the repo root (mounted as `/data:ro` inside the container). These are fixtures and history seeds, not the source of truth in production.

| File | Purpose | Consumed by |
|---|---|---|
| `Cons_Hotel Academic_2026-03-22_2026-04-10.csv` | Recent `Ptot_HA` history used as the 56-day prepend when Cassandra is not available. | `imputer._load_combined_history` |
| `2026 weather data.csv` | `Date`, `AirTemp` series for 2026. Fallback when Cassandra weather is empty. | `imputer._get_weather_df` |
| `2026 historical data.csv` | Full multi-building consumption reference, with a legacy `Quality` column. | Dev fixtures only |
| `2026 forecast data.csv` | Expected output format for downstream predictions. | Reference only |
| `Holidays.xlsx`, `Consumption_2021.xlsx`, `Consumption_2022.xlsx`, `2023 -2025.xlsx` | Historical references, holiday calendars. | Dev only |

The CSVs that get read follow two conventions. Consumption CSVs require a `Date` column (ISO 8601 string, naive or tz-aware both accepted; naive is treated as UTC) plus one numeric column per building (`Ptot_HA`, etc.); the loader renames `Date` to `Timestamp` internally. Weather CSVs require `Date` and `AirTemp`.

CSV mode in the CLI (`--source csv`) uses a different, simpler schema for the single-window input and output. See section 3.3.

## 3.3 CLI CSV I/O contract (`--source csv`)

`impute_cli.py --source csv` is the mode for offline testing or any environment without Cassandra access. Input and output are both CSVs with a simple schema, exactly 1008 rows per file.

### Input CSV

| Column | Type | Notes |
|---|---|---|
| `timestamp` | ISO 8601 string | Strictly increasing, on a 10-minute grid (±1 s tolerance). Naive or tz-aware both accepted; naive is interpreted as UTC. |
| `value` | float | Power in watts. Empty cells (`NaN`) mark gaps. |

Validation checks performed up-front by `impute_cli.load_input`: exactly `EXPECTED_ROWS = 1008` rows (7 days × 144 points/day); no duplicated or decreasing timestamps; every inter-row delta within `FREQ_TOLERANCE_SECONDS = 1` of `EXPECTED_FREQ_SECONDS = 600`; and at least one non-NaN `value` (otherwise there is nothing to impute from).

A failure in any of these produces `ERROR: ...` on stderr and exit code 1.

### Output CSV

Same shape regardless of input mode (CSV or Cassandra):

| Column | Type | Notes |
|---|---|---|
| `timestamp` | string | Echoed through verbatim from the input (CSV mode) or ISO 8601 with explicit UTC offset (Cassandra mode). |
| `value` | float | Imputed, guaranteed non-NaN on a successful run. |
| `quality` | int | Per-point flag 0 to 3, see section 3.1.4. |

Row count matches the input one-to-one.

## 3.4 Test-harness report CSV (`--test-report`)

When `--test-gap` is passed, `impute_cli.py` hides the matching rows in memory, runs the imputer, and (if `--test-report PATH` is also given) writes a per-point comparison CSV with columns `timestamp, ground_truth, imputed, quality, abs_error`.

Footer lines (comments starting with `#`):

```
# strategy=<name> confidence=<f> gap_size=<n>
# postproc.norm mode=... std_ratio=... mean_ratio=... scale=...
# postproc.align mode=... cap=... start_offset_raw=... applied=... end_offset_raw=... applied=... slope_weight=...
# MAE=<f>
# RMSE=<f>
# max_err=<f>
# n_points=<n>
# n_ground_truth=<n>
```

Only gaps that overlap the masked region emit `# strategy=…` and `# postproc.*` headers. The `MAE`, `RMSE`, and `max_err` footers are always present. See [`04-usage.md`](04-usage.md) for the full `--test-gap` / `--test-report` workflow.

## 3.5 Timezone conventions

| Context | Zone |
|---|---|
| Cassandra `"Date"` columns (all tables) | UTC |
| Weather CSV `Date` (on disk) | UTC; tz-naive in the file, localised to UTC on read |
| Internal window, templates, routing | Europe/Paris, tz-aware then tz-naive after stripping the offset |
| APScheduler cron spec | Europe/Paris (`config.TIMEZONE`) |
| Output CSV `timestamp` | Passes through verbatim from the input (CSV mode) or renders as ISO 8601 with explicit `+0000` offset (Cassandra mode) |

The single conversion boundary is `imputer._to_window_time` (mirrored in `window.extract_window`): naive to UTC to Europe/Paris, then strip the offset. Writing back, `cassandra_client.write_reconstructed_window` calls `ts.to_pydatetime()` on the pandas Timestamps, which preserves whatever timezone the caller attached.

DST handling is delegated to `smart_imputation._detect_dst_events`, which tags spring-forward and fall-back days so the weekly template builder does not misalign by one hour. See [`06-algorithms.md`](06-algorithms.md) for the walk-through.

## 3.6 Naming quirks

A few historical inconsistencies in the naming, flagged here so they do not surprise anyone.

`name` is a partition-key constant, not a building code. All rows in both `conso_historiques_clean` and `conso_historiques_reconstructed` use `name = "Conso_Data"`. All rows in `pv_prev_meteo_clean` use `name = "Meteorological_Prevision_Data"`. There is no "one partition per building".

`Ptot_HEI` exists in the clean table but is ignored by this module. It is a legacy aggregate from an earlier phase. The module consumes the individual `Ptot_HEI_13RT` and `Ptot_HEI_5RNS` columns instead.

`Ptot_Campus` is computed, not stored in `_clean`. It appears only in the reconstructed output table and as an in-memory column during processing.

The CSV mode uses a generic `value` column, not a building-named column. The `--building` flag is accepted but ignored in CSV mode: the input file has one data column named `value` and the imputer treats it as `Ptot_HA` internally.

Phase-1 research scripts used short names (`HA`, `HEI1`, `HEI2`, `RIZOMM`, `Campus`, `DateTime`). The current canonical names are the `Ptot_*` columns and `"Date"`. See [`08-glossary.md`](08-glossary.md) for the full mapping.
