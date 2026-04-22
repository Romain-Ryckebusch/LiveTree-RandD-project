# 3. Data Model

Everything the pipeline reads or writes lives in one of three places: Cassandra tables, reference CSVs on disk, or the Kafka `CONSO_Prevision_Data` topic. This file documents the schema of each.

All timestamps in storage are **UTC**. The pipeline converts to **Europe/Paris** for wall-clock reasoning (daily grid, DST detection, scheduling).

## 3.1 Cassandra — keyspace `previsions_data`

Three tables are relevant to N°28. Their schemas are maintained in [`Imputation-Module/cassandra/schema.cql`](../Imputation-Module/cassandra/schema.cql) and in the legacy predictor deployment scripts.

### 3.1.1 `conso_historiques_clean` (input, read-only for N°28)

Source of truth for **real measurements**, written by upstream acquisition services.

| Column | Type | Notes |
|--------|------|-------|
| `name` | `text` | Partition key. Always the literal `"Conso_Data"` for consumption rows. |
| `Date` | `timestamp` | Clustering key. 10-minute grid in UTC. |
| `Ptot_HA` | `double` | Hôtel Académique, watts. |
| `Ptot_HEI_13RT` | `double` | HEI 13 Rue de Toul, watts. |
| `Ptot_HEI_5RNS` | `double` | HEI 5 Rue Nicolas Souriau, watts. |
| `Ptot_RIZOMM` | `double` | RIZOMM building, watts. |
| `Quality` | `text` | Upstream quality flag (dropped by `load_historical_data_cassandra`). |

Queries used:

```cql
-- imputer reads the full history
SELECT * FROM conso_historiques_clean WHERE name = 'Conso_Data';

-- predictor reads a specific 7-day window
SELECT * FROM conso_historiques_clean
 WHERE name = 'Conso_Data' AND "Date" >= ? AND "Date" < ?;
```

Sparse rows (no data for a given timestamp) simply do not exist. Downstream code reindexes onto a complete 10-minute grid and treats the missing rows as NaN.

### 3.1.2 `pv_prev_meteo_clean` (input, read-only for N°28)

Meteorological history and forecasts, written by the PV-forecasting sibling project.

| Column | Type | Notes |
|--------|------|-------|
| `name` | `text` | Partition key. `"Meteorological_Prevision_Data"` for records the imputer uses. |
| `Date` | `timestamp` | Clustering key. 10-minute grid, UTC. |
| `AirTemp` | `double` | °C at the demonstrator's reference station. |
| (other meteo columns) | — | Cloud opacity, irradiance, etc. Not consumed by the imputer. |

Queries used:

```cql
-- imputer pulls the same window as consumption
SELECT * FROM pv_prev_meteo_clean WHERE name = 'Meteorological_Prevision_Data';

-- predictor pulls tomorrow's forecast for feature Airtemp
SELECT "AirTemp" FROM pv_prev_meteo_clean
 WHERE name = 'Meteorological_Prevision_Data'
   AND "Date" >= ? AND "Date" <= ?;
```

### 3.1.3 `conso_historiques_reconstructed` (output, written by the imputer)

DDL:

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

- Partition key `name` is `"Conso_Data"` (same partition key as the clean table, so the rebuilt rows sit next to the real ones logically).
- One `quality_*` column **per building**, not a single `Quality` column. This is deliberate: each of the five nightly per-building runs upserts only its own `(value, quality)` pair, so sibling buildings on the same `Date` keep their existing flags. See the `INSERT` construction in [`cassandra_client.write_reconstructed_window`](../Imputation-Module/src/cassandra_client.py).
- Values are written as raw watts, quality as the integer flag (0–3) documented below.

Quality flag encoding (authoritative mapping in `imputer._STRATEGY_FLAG_MAP`):

| Flag | Meaning | Strategies that emit it |
|------|---------|-------------------------|
| `0` | Real measurement, not imputed | — (set by flagging originally-non-NaN rows) |
| `1` | Linear interpolation | `LINEAR_MICRO`, `LINEAR_SHORT` |
| `2` | Contextual fill (template, peer, safe-median blend) | `THERMAL_TEMPLATE`, `ENHANCED_TEMPLATE`, `WEEKEND_TEMPLATE*`, `PEER_CORRELATION`, `SAFE_LINEAR_MEDIAN`, `HOURLY_MEDIAN_FALLBACK` |
| `3` | ML-derived or donor-day ensemble | `MULTI_WEEK_TEMPLATE`, `MICE`, `KNN_CONTEXT`, `KALMAN_FILTER`, `SAFE_MEDIAN` |

A strategy seen at runtime that is **not** in the mapping (e.g. a new variant added in a future patch) defaults to flag `2` — the safe, contextual bucket.

### 3.1.4 `Ptot_Campus` — a virtual channel

`Ptot_Campus` is **not** stored in `conso_historiques_clean`. The imputer materialises it on-the-fly:

```python
hist_df["Ptot_Campus"] = hist_df[CAMPUS_COMPONENTS].sum(axis=1, skipna=False)
```

`skipna=False` is deliberate: if any of `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM` is NaN for a given timestamp, the campus total is also NaN and gets routed through the imputer just like any other gap. This guarantees the campus reconstruction never silently under-counts one building.

## 3.2 Reference CSVs on disk

These live in `data/` at the repository root (or in `/data` inside the container). They are dev fixtures and holiday calendars — **not** the source of truth in production.

| File | Purpose | Consumed by |
|------|---------|-------------|
| `Cons_Hotel Academic_2026-03-22_2026-04-10.csv` | Recent Ptot_HA history used as a prepend fallback when Cassandra is unavailable | `imputer._load_combined_history` |
| `2026 historical data.csv` | Full 2026 consumption history per building, with an existing quality column | Dev fixtures; not used at runtime in production |
| `2026 weather data.csv` | `Date`, `AirTemp` series for 2026 | `imputer._get_weather_df` (fallback) |
| `2026 forecast data.csv` | Expected output format for predictions | Reference only |
| `Holidays.xlsx` | Multi-year holiday/close calendar used by the predictor | `ConsoFile.py` (training features 5 and 6) |
| `Consumption_<year>_Holiday.csv` / `_Close.csv` / `_Special.csv` | Per-year holiday/close/special-day lists loaded by the imputer | `smart_imputation._load_all_holidays` |
| `Consumption_2021.xlsx`, `Consumption_2022.xlsx`, `2023 -2025.xlsx` | Historical reference data | Dev only |

CSV column conventions consumed by the imputer:

- **Consumption CSVs** — `Date` column (ISO 8601, UTC), one column per building (`Ptot_HA`, etc.).
- **Weather CSV** — `Date`, `AirTemp`. The imputer renames `Date` → `Timestamp` internally and ffill/bfill missing temperatures before building templates.

## 3.3 Kafka — topic `CONSO_Prevision_Data`

The predictor writes one message per 10-minute slot (144 per day). Messages are Avro-encoded and registered with Schema Registry.

### 3.3.1 Key schema

```json
{
  "namespace": "my.test",
  "name": "key",
  "type": "record",
  "fields": [
    {"name": "serie", "type": "string"}
  ]
}
```

The `serie` field is always the literal `"CONSO_Prevision_Data"`. It exists so consumers can demultiplex if the topic ever carries multiple streams.

### 3.3.2 Value schema

```json
{
  "namespace": "my.test",
  "name": "value",
  "type": "record",
  "fields": [
    {"name": "Ptot_HA_Forecast",       "type": ["null", "float"], "default": null},
    {"name": "Ptot_HEI_13RT_Forecast", "type": ["null", "float"], "default": null},
    {"name": "Ptot_HEI_5RNS_Forecast", "type": ["null", "float"], "default": null},
    {"name": "Ptot_Ilot_Forecast",     "type": ["null", "float"], "default": null},
    {"name": "Ptot_RIZOMM_Forecast",   "type": ["null", "float"], "default": null},
    {"name": "Date",                   "type": "string"}
  ]
}
```

- All five forecast fields are nullable — a null indicates "model could not produce a value" (e.g. history was too sparse). In practice, non-null values are always emitted because `MakePredConso` interpolates the history before inference and fails the whole run if the interpolation cannot complete.
- `Date` is the ISO 8601 timestamp of the forecast point in UTC, with explicit offset (e.g. `2026-04-23T02:10:00+0000`).
- `Ptot_Ilot_Forecast` is the legacy name for the campus total (see Section 3.4).

### 3.3.3 Delivery semantics

`AvroProducer.produce()` is called with `flush()` after every record. Delivery is reported via a callback (`delivery_report`) that logs successes and errors. A failure to deliver a single record does not abort the remaining 143 records for that day.

## 3.4 Naming quirks (read me carefully)

A few historical artefacts make the naming inside this project less than perfectly consistent. None of them are bugs; all are documented so nothing is surprising to a new reader.

- **`Ptot_Ilot` vs `Ptot_Campus`.** The Kafka schema says `Ptot_Ilot_Forecast`; the imputer's virtual channel is `Ptot_Campus`. They represent the same physical aggregate: the sum of the four entry/sub-meters. "Îlot" is the legacy label from the original deployment.
- **`Rizomm_TGBT_Puissances_Ptot` in model filenames.** The predictor writes models named `my_modelConsRizomm_TGBT_Puissances_Ptot.h5`. `TGBT` ("Tableau Général Basse Tension") refers to the main low-voltage distribution board; that is the physical point where RIZOMM is metered.
- **`HA_Puissances_Ptot`.** Same pattern: `HA` is the site code, `Puissances_Ptot` is the measurement type. The feature column in the dataframes is `Ptot_HA`.
- **`name` = `"Conso_Data"`.** The Cassandra partition key is a constant string, not a building code. All buildings live in the same partition.

## 3.5 Timezone conventions

| Context | Zone |
|---------|------|
| Cassandra columns | UTC |
| Weather CSV `Date` | UTC (tz-naive in the file, localised on read) |
| Holiday dates | Date-only, interpreted in `Europe/Paris` |
| Window extraction, template building, routing | `Europe/Paris` (tz-aware) |
| APScheduler cron | `Europe/Paris` for the imputer, `Europe/Paris` for the predictor |
| Kafka `Date` field | UTC, ISO 8601 with explicit offset |
| Scheduler cron spec for the predictor | The predictor schedules `02:10` in `Europe/Paris` — this becomes `00:10` or `01:10` UTC depending on DST |

The imputer's `_localise_timestamps` is the single place where naive timestamps are promoted to tz-aware. Spring-forward gaps are filled (`nonexistent='shift_forward'`) and fall-back duplicates are resolved by keeping the first occurrence (`ambiguous=False`). See [`imputation/03-algorithm-deep-dive.md`](imputation/03-algorithm-deep-dive.md#322-dst-handling) for the DST walk-through.
