# 6. Algorithms

How the module actually fills gaps. The engine is `ExtendedDeploymentAlgorithm` in `Imputation-Module/src/smart_imputation.py`. This chapter walks through the run lifecycle, the routing decision tree, every strategy, anomaly detection, post-processing, and the audit log. For a structural map of where things live in the file, see [`05-file-reference.md`](05-file-reference.md).

---

## 6.1 Run lifecycle (`ExtendedDeploymentAlgorithm.impute`)

One call to `impute(df, weather_df=None)` executes these steps in order:

1. **Input** — a DataFrame with a `Timestamp` column and one column per site (`Ptot_HA`, `Ptot_HEI`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, `Ptot_Ilot`), possibly containing NaNs.
2. **Localise timestamps** — convert to `Europe/Paris` (CET/CEST). Naive timestamps are localised directly; UTC-aware timestamps are converted. DST spring-forward and fall-back transitions are detected and logged. See [§6.7](#67-dst-handling).
3. **Merge weather** — if a weather DataFrame was supplied, merge `AirTemp` onto the grid via a left join on `Timestamp`, then forward-fill. Otherwise skip.
4. **Initialise audit log** — stamp the run ID, capture the pre-fill NaN counts per site, record the active configuration, and pre-allocate all log sections. See [§6.8](#68-audit-logging).
5. **Reindex to complete 10-minute grid** — rows missing from the input become explicit NaNs. The grid is tz-aware, anchored at `floor('10min')` of the earliest timestamp.
6. **Feature engineering** — add `Hour`, `DayOfWeek`, `Date`, `ThermalRegime`, `OccupancyType`, `IsOccupied`, `IsHoliday`, `IsSpecialDay`, `IsHolidayClose`, `IsEventDay`, `WeatherSpike`, and `Season`. Occupancy classification uses France 2026 public holidays by default; a `calendar_data` override is accepted at construction time.
7. **Detect raw anomalies (pre-fill)** — `_detect_raw_anomalies` scans for stuck-sensor runs, near-zero readings during active hours, and IsolationForest outliers. Flagged points are masked to NaN so they enter the normal fill pipeline rather than contaminating templates. See [§6.4.1](#641-pre-fill-raw-anomalies-_detect_raw_anomalies).
8. **Build lookups** — peer ratios (`_build_peer_ratios`), day-specific and weekend templates (`_build_day_specific_templates`), seasonal templates (`_build_seasonal_templates`), 28-day adaptive weekly templates (`_build_weekly_templates`), uncertainty bounds (`_build_uncertainty_bounds`), and multi-site correlations (`_build_multi_site_correlations`). Templates are built after anomaly masking so bad values do not skew medians.
9. **Gap loop** — for each site, `_find_gap_groups` returns the list of contiguous NaN runs as `(gap_start, gap_end)` index pairs.
10. **Dispatch each gap** — if `use_chunked_recovery` is on and the gap exceeds `gap_chunk_size`, call `_fill_chunked_gap`, which splits the gap into smart chunks and routes each one. Otherwise call `_intelligent_router` directly. Both paths record a full audit entry per gap/chunk via `_record_gap`.
11. **NaN guard** — `_nan_guard_final_pass` catches anything the main loop left unfilled and applies `_fill_safe_median_template` as a last resort.
12. **Post-fill zero-fill scan** — `_detect_zero_fills` re-runs IsolationForest over all imputed regions and re-fills any stretches that still look like disconnected-sensor zeros. See [§6.4.2](#642-post-fill-zero-fill-detection-_detect_zero_fills).
13. **Smooth junctions** — `_smooth_junctions` applies a Savitzky-Golay filter over the boundary between filled and real segments to remove residual discontinuities.
14. **Finalise and save audit log** — `_finalise_audit_log` writes the run summary and inputs snapshot into the audit structure, then serialises the whole thing to `audit_logs/imputation_audit_{run_id}.json`. See [§6.8](#68-audit-logging).
15. **Return** — the completed DataFrame, column-filtered to `Timestamp` + site columns + `AirTemp` (if present).

---

## 6.2 Routing decision tree

`_intelligent_router(df, site, gap_start, gap_end)` picks one strategy per gap. The cascade runs in order; the first branch that succeeds owns that gap.

```
1. ML cascade — fires if gap_size > 50 AND meter tier is not "entry":
     try MICE         (if use_mice=True)
     try KALMAN       (if use_kalman=True)
     try KNN_CONTEXT  (if use_knn=True)
   First success wins. If all fail or all disabled, fall through.

2. Peer correlation — fires if meter tier == "sub"
   AND parent column exists in the frame
   AND parent has at least one non-NaN value inside [gap_start, gap_end):
     PEER_CORRELATION

3. Rule-based by size:
     gap_size <=   3  ->  LINEAR_MICRO             (<= 30 min)
     gap_size <=  18  ->  LINEAR_SHORT             (<= 3 h)
     pure weekend gap ->  WEEKEND_TEMPLATE_<day>   (SAT / SUN / MIXED)
     non-entry, size <= 144  ->  THERMAL_TEMPLATE  (<= 1 day)
     non-entry, size  > 144  ->  ENHANCED_TEMPLATE (> 1 day)
     entry meter, fallback   ->  SAFE_LINEAR_MEDIAN
```

The ML cascade fires before the size-based rules for non-entry meters, so a 500-point gap (~3.5 days) on `Ptot_HEI_13RT` goes to MICE first, not to `ENHANCED_TEMPLATE`.

**Pure weekend** means every row in `[gap_start, gap_end)` falls on a Saturday or Sunday in `Europe/Paris` with no weekday rows in between. `_get_weekend_day_type` returns `SATURDAY`, `SUNDAY`, or `MIXED` (spans Saturday into Sunday), and the strategy name gets that suffix appended.

**Meter tiers** come from `METER_HIERARCHY` at the top of `smart_imputation.py`:

| Site | Tier | Parent |
|---|---|---|
| `Ptot_HEI` | `main` | — |
| `Ptot_HEI_13RT` | `sub` | `Ptot_HEI` |
| `Ptot_HEI_5RNS` | `sub` | `Ptot_HEI` |
| `Ptot_HA` | `entry` | — |
| `Ptot_RIZOMM` | `entry` | — |

Entry meters skip the ML cascade and peer correlation entirely; they receive `SAFE_LINEAR_MEDIAN` as the terminal fallback.

**`MULTI_WEEK_TEMPLATE`** is not called from `_intelligent_router` directly. It is the per-chunk strategy inside `_fill_chunked_gap`: when a gap exceeds `gap_chunk_size` and `use_chunked_recovery` is on, the gap is split into smart chunks and each chunk is filled with the 28-day adaptive weekly template. This is the primary path for very long gaps.

**`SAFE_MEDIAN` and `HOURLY_MEDIAN_FALLBACK`** are ultimate safety nets, invoked inside specific fill methods when a preferred strategy has insufficient material (too few template samples, an empty parent meter, and so on). They are not part of the main cascade.

Every routing decision appends to a `routing_trace` list that is saved verbatim in the audit log, so any gap's strategy selection can be reconstructed after the fact.

---

## 6.3 Fill strategies

Every strategy mutates `df.loc[gap_start:gap_end - 1, site]` in place. All assume the 10-minute grid is complete and the NaN mask is correct. Filled values are clipped to `[0, site_max × 1.5]` by `_validate_and_clip` before being written.

### 6.3.1 Linear interpolation — `_fill_linear`

Emits `LINEAR_MICRO` (≤ 3 steps) or `LINEAR_SHORT` (≤ 18 steps).

Draws a straight line between `df.loc[gap_start - 1]` and `df.loc[gap_end]`. If either boundary is itself NaN, it walks outward to the nearest real value. Building consumption over a 30-minute to 3-hour window is well-approximated by a segment, so heavier machinery is not warranted.

### 6.3.2 Weekend template — `_fill_weekend_template`

Emits `WEEKEND_TEMPLATE_SATURDAY`, `WEEKEND_TEMPLATE_SUNDAY`, or `WEEKEND_TEMPLATE_MIXED`.

Looks up the median `(day-of-week, hour-of-day)` profile built by `_build_day_specific_templates` and applies it directly. Weekend patterns differ sharply from weekdays (offices closed, labs idle), so a dedicated template avoids contaminating weekday medians.

### 6.3.3 Thermal template — `_fill_with_thermal_template`

Emits `THERMAL_TEMPLATE`.

Uses thermal-regime-conditioned hourly medians. The regime (`Cold`, `Mild`, `Hot`) at gap time is read from `ThermalRegime`; the matching `(regime, day-of-week, hour)` cell is used. Captures heating and cooling load shape without needing explicit occupancy data. Applied to sub-day, non-entry gaps when peer correlation is unavailable.

### 6.3.4 Enhanced template — `_fill_enhanced_template`

Emits `ENHANCED_TEMPLATE`.

Blends the `(day-of-week, hour)` median with the thermal template over a multi-day window. Used for gaps longer than one day on non-entry meters when the ML cascade was not applicable or failed.

### 6.3.5 Safe linear median — `_fill_safe_linear_median`

Emits `SAFE_LINEAR_MEDIAN`.

Blends linear interpolation with an hourly-median correction drawn from recent history (weight 90% linear, 10% median). The terminal fallback for entry meters, which have no peer.

### 6.3.6 Safe median template — `_fill_safe_median_template`

Emits `SAFE_MEDIAN`.

Pure hourly-median fill from the available history. Invoked as an ultimate safety net inside other methods when no template or peer can produce a better shape, and by `_nan_guard_final_pass` for any values that remain NaN after the main loop.

### 6.3.7 Peer correlation — `_fill_via_peer_correlation`

Emits `PEER_CORRELATION`.

Applies `child = parent × ratio`, where `ratio` is the median `child / parent` ratio learned from the most recent `STEPS_PER_DAY` (144) paired observations by `_build_peer_ratios`. For sub-meters whose parent is intact in the gap window this usually gives the best shape, because both meters see the same load profile.

Only fires for `Ptot_HEI_13RT` and `Ptot_HEI_5RNS` when `Ptot_HEI` has at least one non-NaN value within the gap.

### 6.3.8 MICE — `_fill_with_mice`

Emits `MICE`.

Multiple Imputation by Chained Equations via `sklearn.impute.IterativeImputer`. Iteratively predicts missing values from the other site columns and recent history; runs for 3 iterations by default. First choice in the ML cascade for long gaps on non-entry meters.

### 6.3.9 Kalman filter — `_fill_with_kalman_filter`

Emits `KALMAN_FILTER`.

State-space smoothing via `pykalman.KalmanFilter`. Falls back to `pandas.Series.interpolate(method='linear')` if `pykalman` is not installed. Second choice in the ML cascade.

### 6.3.10 KNN context — `_fill_with_knn_context`

Emits `KNN_CONTEXT`.

`sklearn.neighbors.NearestNeighbors` (default `k=5`) over the other site columns as a feature vector. Disabled by default (`use_knn=False`) because MICE and Kalman cover most cases; kept as a third-line ML fallback.

### 6.3.11 Multi-week template — `_fill_with_multi_week_template`

Emits `MULTI_WEEK_TEMPLATE`.

The 28-day adaptive weighted template. For each `(day-of-week, hour)` cell in the gap, looks up the blended median from `_build_weekly_templates`, which weights recent observations at `adaptive_template_bias` (default 0.7) and historical observations at `1 - adaptive_template_bias`. Falls back to `HOURLY_MEDIAN_FALLBACK` for cells with no template data.

Primarily invoked by `_fill_chunked_gap` for very long gaps split into chunks. The chunked approach means each chunk is filled fresh from the template without accumulating error from previous chunks.

---

## 6.4 Anomaly detection

Two IsolationForest-based scans bracket the fill: one before (to clean the input) and one after (to catch bad fills).

### 6.4.1 Pre-fill: raw anomalies — `_detect_raw_anomalies`

Runs after feature engineering, before template building, so bad values do not skew medians.

Three detectors run per site in sequence:

**Stuck-sensor** — a run of 18 or more consecutive rows where `|val[i] − val[i−1]| < 0.01 % × site_median`. Physically, a meter frozen at the same reading for three or more hours signals a communication fault.

**Near-zero** — a non-NaN value below 1 % of the site median during an `OccupancyType` of `work_hours` or `evening`. A meter reading near zero during a busy Tuesday afternoon is implausible. Not applied to `weekend` or `holiday` rows.

**IsolationForest** — trained on the clean (non-stuck, non-zero) portion of the site column using a two-feature vector `[value, hour_of_day]` so the model understands that a high reading at 14:00 is normal but the same value at 03:00 is not. Contamination fraction is `zero_fill_contamination` (default 0.05). Falls back to a 3-σ z-score check if `sklearn` is unavailable.

All three detectors contribute to a unified anomaly mask. Contiguous flagged runs are converted to NaN and logged as individual records in `audit_log['raw_anomaly_corrections']`. The report is accessible via `get_raw_anomaly_report()`.

### 6.4.2 Post-fill: zero-fill detection — `_detect_zero_fills`

After the main fill loop and NaN guard, `_detect_zero_fills` re-runs IsolationForest over every imputed region, scoring each filled value against a model trained on the pre-fill valid data. Values flagged as anomalous *and* below 1 % of the site median are suspected zero fills — for example, a linear interpolation between two near-zero boundaries that produces a plausible-looking but actually flat segment.

For each flagged stretch: the original filled values are logged, the region is reset to NaN, and `_intelligent_router` is called again with `zero_fill_correction=True`. The before and after values are both saved in `audit_log['zero_fill_corrections']`. The report is accessible via `get_zero_fill_report()`.

---

## 6.5 Chunked gap recovery — `_fill_chunked_gap`

When `use_chunked_recovery` is on and a gap exceeds `gap_chunk_size`, the gap is split into chunks rather than filled as a single unit. This prevents error from accumulating across a long, heterogeneous span.

With `use_smart_chunking` on (default), `_get_smart_chunks` adapts chunk boundaries to natural break points — midnight transitions and high-variance days — rather than cutting at a fixed number of steps. High-variance days are those whose within-day standard deviation exceeds 1.3× the median across all days. Chunks are capped at `gap_chunk_size × 1.5` steps.

Recommended chunk sizes by gap length:

| Gap length | `gap_chunk_size` |
|---|---|
| 1–4 days | 144 steps (1 full day) |
| 5–7 days | 96 steps (~6.7 h) |
| 8+ days | 72 steps (~5 h) |

Each chunk is filled with `_fill_with_multi_week_template` when `use_multi_week_templates` is on, otherwise with `_intelligent_router`. Each chunk produces its own audit record, with `is_chunk=True` and `chunk_index` set.

---

## 6.6 Confidence scoring

After every fill, `_calculate_confidence_with_uncertainty` produces a score in `[0.0, 1.0]` as the product of four factors:

| Factor | What it captures |
|---|---|
| Strategy factor | Quality ceiling of the method used (MULTI_WEEK_TEMPLATE = 0.90, MICE = 0.85, down to SAFE_MEDIAN = 0.30) |
| Gap factor | Degrades with gap length (1.0 for < 10 steps, 0.3 for ≥ 7 days) |
| Occupancy factor | Work hours = 1.0, evening = 0.8, weekend = 0.7, holiday = 0.5 |
| Holiday factor | 0.7 if the gap falls on a public holiday, 1.0 otherwise |

Gaps scoring below 0.50 are appended to `_low_confidence_flags` and accessible via `get_low_confidence_report()`. Gaps scoring above 0.85 are considered high quality and require no manual review.

---

## 6.7 DST handling — `_localise_timestamps` and `_detect_dst_events`

`Europe/Paris` switches twice a year. Both transitions are handled explicitly so they do not corrupt templates or produce phantom gaps.

**Spring-forward** (last Sunday of March, 02:00 → 03:00) — one hour of wall-clock time vanishes, producing six missing 10-minute slots. `_localise_timestamps` uses `tz_localize(..., nonexistent='shift_forward')` to map those absent slots forward rather than raising. `_detect_dst_events` detects a ~70-minute jump between consecutive timestamps and logs the event as `spring_forward` in `detection_summary.dst_events`. The six slots appear as a structural gap and are filled normally by the main loop.

**Fall-back** (last Sunday of October, 03:00 → 02:00) — one hour of wall-clock time repeats. Without handling, the same `(day, hour)` template cell would be populated from two different UTC instants. `_localise_timestamps` uses `ambiguous=False`, which keeps the first occurrence of the duplicated hour; the reindex step deduplicates the second. `_detect_dst_events` detects a negative or near-zero step and logs the event as `fall_back`.

If you move the module to a non-DST timezone, `_detect_dst_events` is safe to leave in place — it will simply find nothing. For a different DST-observing zone, the 70-minute and −50-minute detection thresholds still hold for one-hour transitions on a 10-minute grid, but verify the zone's actual offset changes.

---

## 6.8 Audit logging

Every `impute()` call writes a structured JSON file to `audit_logs/` (configurable via `audit_log_dir` at construction time). The filename is `imputation_audit_{run_id}.json`. The log is always written — there is no opt-out flag.

### Structure

**Run metadata** (`run_meta`)
- `run_id`, `run_started_at`, `run_completed_at`
- `timezone`, `data_range` (start / end timestamps), `total_rows`
- `nan_counts_before` and `nan_counts_after` per site
- `sites_with_gaps`
- `template_lookback_days`, `adaptive_template_bias`, `gap_chunk_size`
- `methods_enabled` — boolean flags for MICE, Kalman, KNN, multi-week templates, chunked recovery, smart chunking

**Inputs snapshot** (`inputs_snapshot`) — filled after lookups are built, before any gap is touched:
- `peer_ratios` — learned `child / parent` ratio per sub-meter
- `uncertainty_bounds` — `(lower, upper)` per site
- `peer_correlations` — pairwise Pearson correlations across all site pairs

**Detection summary** (`detection_summary`):
- `gaps_found_per_site` — count of gap groups per site
- `total_missing_steps` — aggregate NaN count before filling
- `detection_method` — always `reindex_to_10min_grid_then_isna_scan`
- `dst_events` — list of spring-forward / fall-back events detected

**Per-gap records** (`gaps`) — one entry per gap (or chunk, when chunked recovery is active):
- `site`, `gap_start_index`, `gap_end_index`, `gap_size_steps`, `gap_size_minutes`
- `first_missing_timestamp`, `last_missing_timestamp`, `missing_timestamps` (full list)
- `missing_dates`, `missing_day_names`, `unique_days_in_gap`, `missing_hours`
- `occupancy_types`, `is_holiday_flags` — per-step arrays
- `method` — strategy name
- `confidence` — score in `[0.0, 1.0]`
- `is_chunk`, `chunk_index`
- `zero_fill_corrected` — `true` if this record was produced by a zero-fill re-impute pass
- `routing_trace` — ordered list of every branch decision, e.g. `START: size=288, tier=sub`, `BRANCH: ML_CASCADE`, `TRY: MICE`, `FAIL: MICE`, `SELECTED: KALMAN_FILTER (confidence=0.612)`
- `inputs_used` — template values (`median`, `std`, `recent_count`, `historical_count`) per `(day, hour)` cell, peer ratio if applicable, and uncertainty bounds

**Anomaly corrections**:
- `raw_anomaly_corrections` — one record per contiguous pre-fill anomalous run: site, timestamps, original values, detection reason (`stuck_sensor`, `near_zero`, `isolation_forest_outlier`)
- `zero_fill_corrections` — one record per post-fill correction: site, timestamps, original filled values, new filled values, detection method

**Run summary** (`run_summary`):
- `total_gaps_processed`, `total_sites_affected`
- `methods_used` — ordered unique list of strategies applied
- `mean_confidence`, `min_confidence`
- `low_confidence_gap_count` — gaps scoring below 0.50
- `raw_anomaly_corrections_count`, `zero_fill_corrections_count`, `dst_events_count`
- `nan_counts_after` — post-fill NaN count per site (should be all zeros on a clean run)

### Accessors

| Method | Returns |
|---|---|
| `get_audit_log()` | Full audit dict from the most recent run |
| `get_run_summary()` | Flat run-level summary dict |
| `get_gap_log()` | DataFrame, one row per imputed gap |
| `get_strategy_log()` | DataFrame of `(site, gap_start, gap_end, gap_size, strategy, confidence, day_type, occupancy_type, is_holiday)` |
| `get_low_confidence_report()` | DataFrame of gaps scoring below 0.50 |
| `get_zero_fill_report()` | DataFrame of post-fill zero-fill corrections |
| `get_raw_anomaly_report()` | DataFrame of pre-fill anomaly runs |

### Replaying an audit

Load the JSON, take the `gaps` list, and filter to the gap of interest. The `routing_trace` is exhaustive — it records every branch considered and the reason the final strategy was selected. To re-run the fill for a specific gap, reconstruct the inputs from `inputs_used` and replay the method named in `method`.

### Template persistence

Templates can be saved and reloaded independently of an `impute()` call:

```python
algo.save_templates('templates_2026Q2.pkl')   # saves after a run
algo.load_templates('templates_2026Q2.pkl')   # restores before the next run
```

Saved artefacts: `_weekly_templates`, `_day_variance`, `_seasonal_templates`, `_peer_ratios`, `_multi_site_correlations`, `_uncertainty_bounds`.
