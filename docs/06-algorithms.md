# 6. Algorithms

How the module actually fills gaps. The engine is `ExtendedDeploymentAlgorithm` in `Imputation-Module/src/smart_imputation.py` (1 964 lines). This chapter walks through the run lifecycle, the routing decision tree, every strategy, and the post-processing steps. For a structural map of where things live in the file, see [`05-file-reference.md`](05-file-reference.md#54-srcsmart_imputationpy).

## 6.1 Run lifecycle (`ExtendedDeploymentAlgorithm.impute`)

One call to `impute(df, weather_df=None)` executes these steps in order:

 1. Input: a DataFrame with a `Timestamp` column and one column per site, possibly containing NaNs.
 2. Localise timestamps: convert UTC to Europe/Paris.
 3. If a weather DataFrame was supplied, merge `AirTemp` onto the grid. Otherwise skip.
 4. Initialise the audit log.
 5. Reindex to a complete 10-minute grid; rows that were missing from the input become explicit NaNs.
 6. Feature engineering: add datetime features, occupancy, thermal regime, and external features.
 7. Detect raw anomalies (pre-fill): scan for zero-fill and stuck-sensor patterns and mark them NaN.
 8. Build lookups: peer ratios, day-specific, seasonal, and weekly templates, uncertainty bounds, and multi-site correlations.
 9. For each site, find gap groups (runs of NaNs).
10. For each gap, dispatch. If the gap size exceeds `chunk_size` and `use_chunked_recovery` is on, call `_fill_chunked_gap`, which splits the gap into smart chunks and routes each chunk. Otherwise call `_intelligent_router`, which picks and runs one strategy.
11. Run a final NaN guard pass to catch anything missed.
12. Post-fill zero-fill scan (`_detect_zero_fills`): re-run IsolationForest and re-fill suspicious stretches.
13. Smooth junctions: Savitzky-Golay filter over fill/real boundaries.
14. Finalise the audit log and write it to JSON on disk.
15. Return the completed DataFrame.

Key file locations: the `impute` method is at line 860; the routing function at line 1379; the template builders are around 1110; the DST detector is at line 272.

## 6.2 Routing decision tree

`_intelligent_router(df, site, gap_start, gap_end)` picks one strategy per gap. The cascade, in order:

```
1. ML cascade: if gap_size > 50 AND meter is not an "entry" tier:
     try MICE        (if enabled)   -> flag 3
     try KALMAN      (if enabled)   -> flag 3
     try KNN_CONTEXT (if enabled)   -> flag 3
   First success wins. If all fail or all disabled, fall through.

2. Peer correlation: if meter tier == "sub"
   AND parent column exists in the frame
   AND parent has at least one non-NaN value inside [gap_start, gap_end):
     PEER_CORRELATION -> flag 2

3. Rule-based by size:
     gap_size <=  3          -> LINEAR_MICRO           -> flag 1    (<= 30 min)
     gap_size <= 18          -> LINEAR_SHORT           -> flag 1    (<= 3 h)
     pure weekend gap        -> WEEKEND_TEMPLATE_<day> -> flag 2    (SAT / SUN / MIXED)
     non-entry, size <= 144  -> THERMAL_TEMPLATE       -> flag 2    (<= 1 day)
     non-entry, size  > 144  -> ENHANCED_TEMPLATE      -> flag 2    (> 1 day)
     entry meter, fallback   -> SAFE_LINEAR_MEDIAN     -> flag 2    (anything left on an entry meter)
```

The ML cascade fires before the size-based rules on non-entry meters, so a 500-point gap (about 3.5 days) on `Ptot_HEI_13RT` goes to MICE first, not to `ENHANCED_TEMPLATE`. That is why `ENHANCED_TEMPLATE` is seen less often than the flag-2 tier might suggest.

"Pure weekend" means `gap_start` and `gap_end - 1` are both Saturday or Sunday in Europe/Paris and no weekday rows sit in between; `_get_weekend_day_type` returns `SATURDAY`, `SUNDAY`, or `MIXED` (spans Sat into Sun), and the strategy name gets that suffix.

The entry-meter tier comes from `METER_HIERARCHY` at the top of `smart_imputation.py`: `Ptot_HA` and `Ptot_RIZOMM` are entries, `Ptot_HEI_13RT` and `Ptot_HEI_5RNS` are sub-meters (parent `Ptot_HEI`, which is never consumed as data but only referenced for the hierarchy); `Ptot_HEI` itself is `main` and `Ptot_Campus` is not in the hierarchy.

Every routing decision records a `routing_trace` list into the audit log, so the audit JSON can be replayed to explain why a specific gap got a specific strategy.

`MULTI_WEEK_TEMPLATE` is not called directly from `_intelligent_router`. It runs inside `_fill_chunked_gap` when `use_chunked_recovery` is on and the gap is too long to route as a single unit: each chunk gets routed individually, and the multi-week template is the long-chunk fallback. This is the primary way flag `3` appears under the `MULTI_WEEK_TEMPLATE` name, as opposed to the ML-cascade strategies.

`SAFE_MEDIAN` and `HOURLY_MEDIAN_FALLBACK` are ultimate safety nets invoked when a preferred strategy has no material to work with (too few template samples, the parent meter empty, and so on). They are not in the main cascade; they are fallbacks inside specific fill methods.

## 6.3 Fill strategies

Every strategy mutates `df.loc[gap_start:gap_end - 1, site]` in place. They all assume the 10-minute grid is complete and the NaN mask is correct.

### 6.3.1 Linear interpolation: `_fill_linear`

Emits `LINEAR_MICRO` (gap of 3 steps or fewer) or `LINEAR_SHORT` (gap of 18 steps or fewer). Flag 1.

Draws a straight line between `df.loc[gap_start - 1]` and `df.loc[gap_end]`. Falls back gracefully if either boundary is itself NaN by walking outward to find the nearest real value.

Used by the size-based branch for gaps up to 3 hours. The shape of a building's consumption over a 3-hour window is well approximated by a segment, so there is no point bringing heavier machinery in.

### 6.3.2 Weekend template: `_fill_weekend_template`

Emits `WEEKEND_TEMPLATE_SATURDAY`, `WEEKEND_TEMPLATE_SUNDAY`, or `WEEKEND_TEMPLATE_MIXED`. Flag 2.

Looks up the median `(day-of-week, hour-of-day)` profile from `_build_day_specific_templates` and uses it directly. Weekend patterns are strongly different from weekdays (offices closed, labs idle), so building a separate median avoids contaminating weekday templates with weekend data.

Triggered when the entire gap falls within a Saturday, a Sunday, or both. Mixed weekends (the gap spans Sat into Sun) get the `_MIXED` suffix.

### 6.3.3 Thermal template: `_fill_with_thermal_template`

Emits `THERMAL_TEMPLATE`. Flag 2.

Uses thermal-regime-conditioned templates built in `_classify_thermal_regimes` and `_build_day_specific_templates`. The regime at gap time is determined from the `AirTemp` series, and the template matching that `(regime, day-of-week, hour)` cell gets pulled.

Used for sub-day, non-entry gaps when a peer-correlation fill was not available. Captures the heating and cooling-load shape without needing to know the occupancy explicitly.

### 6.3.4 Enhanced template: `_fill_enhanced_template`

Emits `ENHANCED_TEMPLATE`. Flag 2.

A multi-day template fill that blends the `(day-of-week, hour)` median with the thermal template over a multi-day window. Used when the gap exceeds 1 day but the ML cascade is not applicable (an entry meter with ML disabled, or a non-entry where ML failed and the size-based rule fell through).

### 6.3.5 Safe linear median: `_fill_safe_linear_median`

Emits `SAFE_LINEAR_MEDIAN`. Flag 2.

A blend of linear interpolation with an hourly-median correction from the recent weeks. Used as the final fallback for entry meters, which do not have a peer.

### 6.3.6 Safe median template: `_fill_safe_median_template`

Emits `SAFE_MEDIAN_TEMPLATE`. Flag 2 by default (the `_STRATEGY_FLAG_MAP` currently has no entry for this literal name, so `_flag_for_strategy`'s fallback kicks in).

A pure hourly-median fill from the 56-day prepend. Safety net for cases where no template or peer can produce a better shape.

### 6.3.7 Peer correlation: `_fill_via_peer_correlation`

Emits `PEER_CORRELATION`. Flag 2.

Applies `child = parent × ratio`, where `ratio` was learned in `_build_peer_ratios` from historical `(child, parent)` pairs. For sub-meters whose parent is intact in the gap window, this is usually the best shape, because both meters see the same load.

Only triggered for sub-meter tiers (`Ptot_HEI_13RT`, `Ptot_HEI_5RNS`) when the parent column (`Ptot_HEI`) has at least one non-NaN value across the gap.

### 6.3.8 MICE: `_fill_with_mice`

Emits `MICE`. Flag 3.

Multiple Imputation by Chained Equations. Iteratively predicts the missing values from the other site columns and recent history. Runs for a fixed 3 iterations by default.

First choice in the ML cascade for long gaps on non-entry meters.

### 6.3.9 Kalman filter: `_fill_with_kalman_filter`

Emits `KALMAN_FILTER`. Flag 3.

State-space smoothing via `pykalman`. Used as the second ML-cascade fallback when MICE fails.

### 6.3.10 KNN context: `_fill_with_knn_context`

Emits `KNN_CONTEXT`. Flag 3.

`sklearn.neighbors.NearestNeighbors` over a feature-engineered context window (day-of-week, hour, occupancy, temperature). Default `k=5`.

Disabled by default in `imputer.py` (`use_knn=False`) because MICE and Kalman cover most cases; KNN is kept as a third-line fallback.

### 6.3.11 Multi-week template: `_fill_with_multi_week_template`

Emits `MULTI_WEEK_TEMPLATE`. Flag 3.

The donor-day ensemble. Walks the 56-day (`_PREPEND_DAYS`) history, finds days similar to the gap day by `(day-of-week, thermal regime, occupancy)`, takes a weighted mean over the donor days' shapes at the corresponding hours, and fills.

Has a "more than 40 percent missing" fallback branch: if the donor-day window itself is too sparse, the method falls back to `SAFE_MEDIAN`. The 56-day prepend is what keeps this branch from kicking in on realistic outages.

Primarily invoked by `_fill_chunked_gap` for very long gaps that are split into chunks.

## 6.4 Anomaly detection

Two IsolationForest-based scans sandwich the fill.

### 6.4.1 Pre-fill: raw anomalies (`_detect_raw_anomalies`)

Before any gap is filled, `_detect_raw_anomalies` scans the raw input for three kinds of suspect rows. Stuck sensors are long runs of exactly-identical values that are not credible physically. Near-zero stretches are periods where a meter pegs at zero while neighbours show normal load. IsolationForest outliers are points that the detector, trained on the 56-day context, flags as implausibly far from the local distribution.

Flagged points are masked to NaN so they get imputed rather than propagated into templates. The report is accessible via `get_raw_anomaly_report()`.

### 6.4.2 Post-fill: zero-fill detection (`_detect_zero_fills`)

After the main fill, `_detect_zero_fills` re-runs IsolationForest over the reconstructed series, looking for stretches that still look like disconnected-sensor zeros. Sometimes a linear fill between two near-zero boundaries produces a plausible-looking but actually-flat segment; those get re-filled via `_fill_chunked_gap` with the zero-fill correction flag, which logs them distinctly in the audit.

The report is accessible via `get_zero_fill_report()`.

## 6.5 Boundary anchoring (`_anchor_to_boundaries`)

After a template-based or ML-based strategy fills a gap, the filled segment usually has a different mean and sometimes a different shape from the surrounding real values. Without correction, a filled segment can show a visible discontinuity at its endpoints, which then propagates into the forecaster's features as a synthetic transient.

`_anchor_to_boundaries(df, site, gap_start, gap_end, filled_vals, tpl_fn=None)` runs after every template or ML fill. It blends the filled segment into the real values at both ends using a slope-aware offset. It computes `start_offset = real_left - filled_left` and `end_offset = real_right - filled_right`, applies a weighted, ramped blend from `start_offset` to `end_offset` across the segment so both endpoints match the neighbours exactly, and records the blend parameters (`mode`, `cap`, offsets, slope weight) into `postproc.align` in the audit log.

The `--test-report` CSV's `# postproc.align mode=... cap=... start_offset_raw=... applied=... end_offset_raw=...` footer comes from this step.

## 6.6 Smoothing (`_smooth_junctions`)

After all gaps are filled and anchored, `_smooth_junctions` runs a Savitzky-Golay filter over the junctions between filled and real segments to remove any residual second-derivative discontinuity. This is a cosmetic pass: it does not change the mean behaviour, but it prevents the forecaster's gradient-based features from seeing a notch.

The window length and polynomial order are hard-coded for 10-minute sampling and 7-day windows; if you change the frequency, revisit this function.

## 6.7 DST handling (`_detect_dst_events` and `_localise_timestamps`)

Europe/Paris switches twice a year. The imputer handles both transitions explicitly so they do not corrupt templates.

### Spring-forward (last Sunday of March, 02:00 to 03:00)

One hour of wall-clock time vanishes. On a 10-minute grid, six consecutive slots are absent. `_localise_timestamps` uses pandas `tz_localize(..., nonexistent="shift_forward")` so those vanished slots are shifted into the next valid slot rather than raising. `_detect_dst_events` scans the localised timestamps for a 70-minute jump between consecutive rows and logs it as `spring_forward` in the audit. The affected six slots show up as a structural gap and get filled normally.

### Fall-back (last Sunday of October, 03:00 to 02:00)

One hour of wall-clock time repeats. Without care, the same `(day, hour)` cell in a weekly template would be populated from two different UTC instants. `_localise_timestamps` uses `tz_localize(..., ambiguous=False)` which keeps the first occurrence of the duplicated hour. The duplicate is then simply an extra six rows at the same local wall-clock time, which the reindex step deduplicates. `_detect_dst_events` logs the transition as `fall_back` in the audit.

If you move this module to a non-DST timezone, `_detect_dst_events` is safe to leave in; it simply finds nothing. If you move to a different DST-observing zone with different transition rules, the 70-minute and −50-minute thresholds in the detector still hold (one-hour shifts, 10-minute grid), but double-check the zone's actual offset changes.

## 6.8 Audit logging

Every `impute()` call writes a structured JSON audit log to `AUDIT_LOG_DIR` (default `/io/audit_logs/` in Docker). The log has three top-level sections.

Run metadata covers the run ID, start and end timestamps, input timespan, total rows, NaN counts per site, and DST events detected.

The detection summary lists gaps found per site, sites affected, stuck-sensor detections, and IsolationForest outliers.

Per-gap records, one per gap, carry the site, the gap start and end indices, the gap size in steps, a `routing_trace` (the ordered list of routing-branch decisions like `START: size=..., tier=...`, `BRANCH: ML_CASCADE`, `TRY: MICE`, `FAIL: MICE`, ..., `SELECTED: <strategy>`), the strategy chosen, the confidence, the occupancy type, the holiday flag, the day type, `postproc.norm` (normalisation step parameters: mode, std ratio, mean ratio, scale applied), `postproc.align` (anchoring parameters: mode, cap, start and end offsets raw and applied, slope weight), and when applicable the template inputs used (template values, peer ratios, uncertainty bounds).

The run summary records total gaps, methods used, mean confidence, and zero-fill corrections applied.

Accessors on the engine object cover every slice of this log. `get_audit_log()` returns the full dict, `get_run_summary()` returns just the summary section, `get_gap_log()` returns a pandas DataFrame of all per-gap records, and `get_strategy_log()` returns a DataFrame of `(site, gap_start, gap_end, gap_size, strategy, confidence, day_type, occupancy_type, is_holiday)`. The last is the table reprojected by `imputer.py` to produce the per-point quality flags. `get_zero_fill_report()`, `get_raw_anomaly_report()`, and `get_low_confidence_report()` are focused views for specific inspections.

For replaying an audit from a past run, load the JSON, take the `per_gap_records` list, and filter to the gap of interest. The `routing_trace` is complete: it lists every branch considered and tells you why the final one was picked.
