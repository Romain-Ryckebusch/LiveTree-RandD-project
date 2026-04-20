"""
Extended Deployment Gap Recovery Algorithm — v2
Adds to the fixed v1:
  ✓ Full audit log (instance + auto-saved JSON per run)
  ✓ Per-gap record: missing timestamps, day, occupancy, method, confidence,
    inputs used (template values / peer ratios / bounds), routing trace
  ✓ Run-level summary: gaps found, sites affected, methods used, mean confidence
  ✓ Detection log: how each gap was found
  ✓ Zero-fill detection via IsolationForest → re-impute flagged regions
  ✓ Timezone awareness: Europe/Paris (CET/CEST), DST spring-forward + fall-back
"""

import json
import logging
import os
import pickle
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors

try:
    import pytz
    _PYTZ_AVAILABLE = True
except ImportError:
    _PYTZ_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    _IF_AVAILABLE = True
except ImportError:
    IsolationForest = None
    _IF_AVAILABLE = False

try:
    from autoencoder_imputer import ProfileImputer
except ImportError:
    ProfileImputer = None

log = logging.getLogger(__name__)

SITE_COLS = ['Ptot_HA', 'Ptot_HEI', 'Ptot_HEI_13RT', 'Ptot_HEI_5RNS', 'Ptot_RIZOMM', 'Ptot_Ilot']
STEPS_PER_DAY = 144  # 10-min intervals × 6/hr × 24 hr
SITE_TZ = 'Europe/Paris'

METER_HIERARCHY = {
    'Ptot_HEI':      {'parent': None,       'children': ['Ptot_HEI_13RT', 'Ptot_HEI_5RNS'], 'tier': 'main'},
    'Ptot_HEI_13RT': {'parent': 'Ptot_HEI', 'children': [],                                  'tier': 'sub'},
    'Ptot_HEI_5RNS': {'parent': 'Ptot_HEI', 'children': [],                                  'tier': 'sub'},
    'Ptot_HA':       {'parent': None,        'children': [],                                  'tier': 'entry'},
    'Ptot_RIZOMM':   {'parent': None,        'children': [],                                  'tier': 'entry'},
}


# ─────────────────────────────────────────────────────────────────────────────
# Holiday loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_all_holidays():
    holidays: set = set()
    close_days: set = set()
    special_days: set = set()
    data_dir = os.environ.get('IMPUTER_DATA_DIR', 'data')
    for year in range(2021, 2027):
        for ftype, target in [('Holiday', holidays), ('Close', close_days), ('Special', special_days)]:
            fname = os.path.join(data_dir, f'Consumption_{year}_{ftype}.csv')
            try:
                df = pd.read_csv(fname, header=0, names=['date'], parse_dates=['date'])
                target.update(df['date'].dt.date.unique())
            except FileNotFoundError:
                pass
    for m, d in [(1,1),(4,5),(4,6),(5,1),(5,8),(5,14),(5,25),(7,14),(8,15),(11,1),(11,11),(12,25)]:
        holidays.add(date(2026, m, d))
    return holidays, close_days, special_days


FRANCE_HOLIDAYS_2026, FRANCE_CLOSE_DAYS_2026, FRANCE_SPECIAL_DAYS_2026 = _load_all_holidays()


# ─────────────────────────────────────────────────────────────────────────────
# Audit helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(dt) -> str:
    """Convert a pandas Timestamp or datetime to an ISO string for JSON serialisation."""
    if pd.isna(dt):
        return None
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)


def _json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return _ts(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ExtendedDeploymentAlgorithm:
    """
    Gap-recovery algorithm with full audit logging, zero-fill detection via
    IsolationForest, and Europe/Paris DST-aware timestamp handling.
    """

    def __init__(
        self,
        site_cols=None,
        use_mice: bool = True,
        use_knn: bool = True,
        use_kalman: bool = True,
        use_deep_learning: bool = True,
        use_multi_week_templates: bool = True,
        use_chunked_recovery: bool = True,
        gap_chunk_size: int = 96,
        occupancy_data: Optional[pd.DataFrame] = None,
        calendar_data: Optional[pd.DataFrame] = None,
        template_lookback_days: int = 28,
        use_smart_chunking: bool = True,
        adaptive_template_bias: float = 0.7,
        # ── NEW v2 params ──────────────────────────────────────────────────
        audit_log_dir: str = 'audit_logs',
        timezone: str = SITE_TZ,
        zero_fill_contamination: float = 0.05,
    ):
        self.site_cols = site_cols or SITE_COLS
        self.use_mice = use_mice
        self.use_knn = use_knn
        self.use_kalman = use_kalman
        self.use_deep_learning = use_deep_learning
        self.use_multi_week_templates = use_multi_week_templates
        self.use_chunked_recovery = use_chunked_recovery
        self.gap_chunk_size = gap_chunk_size
        self.occupancy_data = occupancy_data
        self.calendar_data = calendar_data
        self.template_lookback_days = template_lookback_days
        self.use_smart_chunking = use_smart_chunking
        self.adaptive_template_bias = adaptive_template_bias

        # v2 config
        self.audit_log_dir = audit_log_dir
        self.timezone = timezone
        self.zero_fill_contamination = zero_fill_contamination

        # Internal state
        self.strategy_log: List[Dict] = []
        self.weather_df = None
        self._peer_ratios: Dict[str, float] = {}
        self.templates: Dict = {}
        self._kalman_states: Dict = {}
        self._kalman_covariance: Dict = {}
        self._knn_models: Dict = {}
        self._fitted_knn: bool = False
        self._reconstruction_confidence: Dict[str, float] = {}
        self._low_confidence_flags: List[Dict] = []
        self._seasonal_templates: Dict = {}
        self._multi_site_correlations: Dict = {}
        self._weekly_templates: Dict = {}
        self._uncertainty_bounds: Dict = {}
        self._occupancy_patterns: Dict = {}
        self._day_variance: Dict = {}
        self._external_event_flags: Dict = {}
        self._weather_variance: Dict = {}

        # ── v2: audit state ───────────────────────────────────────────────
        # Cleared at the start of each impute() call.
        self._audit_log: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_templates(self, filepath: str):
        """Save templates and statistics to a pickle file for later reuse."""
        data = {
            'weekly_templates': self._weekly_templates,
            'day_variance': self._day_variance,
            'seasonal_templates': self._seasonal_templates,
            'peer_ratios': self._peer_ratios,
            'multi_site_correlations': self._multi_site_correlations,
            'uncertainty_bounds': self._uncertainty_bounds,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        log.info(f"[PERSIST] Saved templates to {filepath}")

    def load_templates(self, filepath: str):
        """Load templates and statistics from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self._weekly_templates = data.get('weekly_templates', {})
        self._day_variance = data.get('day_variance', {})
        self._seasonal_templates = data.get('seasonal_templates', {})
        self._peer_ratios = data.get('peer_ratios', {})
        self._multi_site_correlations = data.get('multi_site_correlations', {})
        self._uncertainty_bounds = data.get('uncertainty_bounds', {})
        log.info(f"[PERSIST] Loaded templates from {filepath}")

    # ─────────────────────────────────────────────────────────────────────────
    # NEW v2: Timezone handling
    # ─────────────────────────────────────────────────────────────────────────

    def _localise_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all Timestamps are timezone-aware in self.timezone (Europe/Paris).

        Handles three input cases:
          1. Naive timestamps (no tz info)  → localise directly, ambiguous=False (keep first),
             nonexistent='shift_forward' (spring-forward gap filled).
          2. UTC-aware timestamps           → convert to local tz.
          3. Already localised              → convert to self.timezone if different.

        Returns df with tz-aware Timestamps.  DST events are recorded in the
        audit log under 'detection_summary.dst_events'.
        """
        tz_str = self.timezone
        try:
            if _PYTZ_AVAILABLE:
                local_tz = pytz.timezone(tz_str)
            else:
                local_tz = tz_str  # pandas will accept a string

            ts = df['Timestamp']
            dst_events = []

            if ts.dt.tz is None:
                # Naive → localise.  'ambiguous=False' keeps the first (pre-fallback) reading
                # for the duplicated fall-back hour. 'nonexistent=shift_forward' maps the
                # missing spring-forward hour to the post-transition time.
                df['Timestamp'] = pd.to_datetime(ts).dt.tz_localize(
                    tz_str,
                    ambiguous=False,
                    nonexistent='shift_forward',
                )
                log.info(f"[TZ] Localised naive timestamps to {tz_str}")
            elif str(ts.dt.tz) != tz_str:
                df['Timestamp'] = ts.dt.tz_convert(tz_str)
                log.info(f"[TZ] Converted timestamps from {ts.dt.tz} to {tz_str}")
            else:
                log.debug(f"[TZ] Timestamps already in {tz_str}, no conversion needed")

            dst_events = self._detect_dst_events(df)
            self._audit_log.setdefault('detection_summary', {})['dst_events'] = dst_events

        except Exception as e:
            log.warning(f"[TZ] Timezone localisation failed: {e}. Proceeding with original timestamps.")

        return df

    def _detect_dst_events(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scan for DST transition artefacts in the (now tz-aware) timestamp series.

        Spring-forward: a 10-min step that is actually 70 min (one slot vanishes).
        Fall-back:      a 10-min step that is actually −50 min (duplicate hour).

        Returns a list of event dicts for the audit log.
        """
        events = []
        try:
            ts = df['Timestamp']
            diffs = ts.diff().dt.total_seconds() / 60  # in minutes

            # Spring-forward: gap of ~70 min instead of 10 min
            spring = diffs[(diffs > 60) & (diffs < 90)]
            for idx, val in spring.items():
                events.append({
                    'type': 'spring_forward',
                    'index': int(idx),
                    'timestamp': _ts(ts.iloc[idx]),
                    'gap_minutes': round(float(val), 1),
                    'note': 'Clock moved forward 1h; 6 steps (~1h) created as structural gap and filled normally',
                })
                log.info(f"[DST] Spring-forward at index {idx} ({_ts(ts.iloc[idx])})")

            # Fall-back: negative or very small step (duplicate hour)
            fallback = diffs[(diffs < 0) | ((diffs > -60) & (diffs < 5) & (diffs != 10))]
            for idx, val in fallback.items():
                if idx == 0:
                    continue
                events.append({
                    'type': 'fall_back',
                    'index': int(idx),
                    'timestamp': _ts(ts.iloc[idx]),
                    'gap_minutes': round(float(val), 1),
                    'note': 'Clock moved back 1h; ambiguous=False keeps the first (pre-fallback) occurrence',
                })
                log.info(f"[DST] Fall-back at index {idx} ({_ts(ts.iloc[idx])})")

        except Exception as e:
            log.debug(f"[DST] Detection error: {e}")

        return events

    # ─────────────────────────────────────────────────────────────────────────
    # NEW v2: Audit log initialisation and saving
    # ─────────────────────────────────────────────────────────────────────────

    def _init_audit_log(self, df: pd.DataFrame):
        """
        Reset and initialise the audit log at the start of an impute() run.
        Records run-level metadata and the information the algorithm has before
        any gap is touched.
        """
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        ts_min = _ts(df['Timestamp'].min())
        ts_max = _ts(df['Timestamp'].max())
        total_rows = len(df)
        nan_counts = {site: int(df[site].isna().sum()) for site in self.site_cols if site in df.columns}
        sites_with_gaps = [s for s, n in nan_counts.items() if n > 0]

        self._audit_log = {
            'run_id': run_id,
            'run_started_at': datetime.now().isoformat(),
            'run_meta': {
                'timezone': self.timezone,
                'data_range': {'start': ts_min, 'end': ts_max},
                'total_rows': total_rows,
                'nan_counts_before': nan_counts,
                'sites_with_gaps': sites_with_gaps,
                'template_lookback_days': self.template_lookback_days,
                'adaptive_template_bias': self.adaptive_template_bias,
                'gap_chunk_size': self.gap_chunk_size,
                'methods_enabled': {
                    'mice': self.use_mice,
                    'kalman': self.use_kalman,
                    'knn': self.use_knn,
                    'multi_week_templates': self.use_multi_week_templates,
                    'chunked_recovery': self.use_chunked_recovery,
                    'smart_chunking': self.use_smart_chunking,
                },
            },
            'inputs_snapshot': {
                'peer_ratios': {},        # filled after _build_peer_ratios
                'uncertainty_bounds': {}, # filled after _build_uncertainty_bounds
                'peer_correlations': {},  # filled after _build_multi_site_correlations
            },
            'detection_summary': {
                'gaps_found_per_site': {},
                'total_missing_steps': int(sum(nan_counts.values())),
                'detection_method': 'reindex_to_10min_grid_then_isna_scan',
                'dst_events': [],
            },
            'gaps': [],                    # one entry per (site, gap_start, gap_end)
            'raw_anomaly_corrections': [], # implausible raw values → NaN before filling
            'zero_fill_corrections': [],   # near-zero imputed values → re-imputed
            'run_summary': {},             # filled at end of impute()
        }
        log.info(f"[AUDIT] Run {run_id} started. Sites with gaps: {sites_with_gaps}")

    def _record_gap(
        self,
        df: pd.DataFrame,
        site: str,
        gap_start: int,
        gap_end: int,
        strategy: str,
        confidence: float,
        routing_trace: List[str],
        is_chunk: bool = False,
        chunk_index: Optional[int] = None,
        zero_fill_corrected: bool = False,
    ):
        """
        Append a fully-detailed gap record to the audit log.

        Records:
          - Which datapoints (timestamps) were missing
          - Day names and occupancy types for those timestamps
          - The method chosen and its confidence score
          - The inputs the algorithm used at decision time:
              template values (median, std) for each (day, hour) cell in the gap
              peer ratio used (if PEER_CORRELATION)
              uncertainty bounds applied
          - The routing trace (ordered list of decisions made)
        """
        try:
            gap_timestamps = df.loc[gap_start:gap_end - 1, 'Timestamp'].tolist()
            gap_days = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].tolist()
            gap_dates = df.loc[gap_start:gap_end - 1, 'Date'].tolist()
            gap_hours = df.loc[gap_start:gap_end - 1, 'Hour'].tolist()
            gap_occupancy = (
                df.loc[gap_start:gap_end - 1, 'OccupancyType'].tolist()
                if 'OccupancyType' in df.columns else []
            )
            is_holiday_flags = (
                df.loc[gap_start:gap_end - 1, 'IsHoliday'].tolist()
                if 'IsHoliday' in df.columns else []
            )

            # Days of week present in this gap
            unique_days = list(dict.fromkeys(gap_days))  # ordered unique

            # Template values used (first occurrence of each day+hour cell)
            template_inputs_used = {}
            seen_cells = set()
            for h, day_name in zip(gap_hours, gap_days):
                cell = (day_name, h)
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                tmpl = self._weekly_templates.get(site, {}).get(day_name, {}).get(h)
                if tmpl is not None:
                    template_inputs_used[f'{day_name}_h{h:02d}'] = {
                        'median': round(float(tmpl['median']), 4) if tmpl['median'] is not None else None,
                        'std': round(float(tmpl['std']), 4),
                        'recent_count': int(tmpl['recent_count']),
                        'historical_count': int(tmpl['historical_count']),
                    }

            peer_ratio_used = self._peer_ratios.get(site) if strategy == 'PEER_CORRELATION' else None
            ub = self._uncertainty_bounds.get(site)

            record = {
                'site': site,
                'gap_start_index': int(gap_start),
                'gap_end_index': int(gap_end),
                'gap_size_steps': int(gap_end - gap_start),
                'gap_size_minutes': int((gap_end - gap_start) * 10),
                'first_missing_timestamp': _ts(gap_timestamps[0]) if gap_timestamps else None,
                'last_missing_timestamp': _ts(gap_timestamps[-1]) if gap_timestamps else None,
                'missing_timestamps': [_ts(t) for t in gap_timestamps],
                'missing_dates': [str(d) for d in gap_dates],
                'missing_day_names': gap_days,
                'unique_days_in_gap': unique_days,
                'missing_hours': gap_hours,
                'occupancy_types': gap_occupancy,
                'is_holiday_flags': [bool(f) for f in is_holiday_flags],
                'method': strategy,
                'confidence': round(float(confidence), 4),
                'is_chunk': is_chunk,
                'chunk_index': chunk_index,
                'zero_fill_corrected': zero_fill_corrected,
                'routing_trace': routing_trace,
                'inputs_used': {
                    'template_values': template_inputs_used,
                    'peer_ratio': round(float(peer_ratio_used), 6) if peer_ratio_used is not None else None,
                    'uncertainty_bounds': {
                        'lower': round(float(ub[0]), 4),
                        'upper': round(float(ub[1]), 4),
                    } if ub else None,
                },
            }

            self._audit_log['gaps'].append(record)

            # Update per-site gap count in detection_summary
            det = self._audit_log['detection_summary']['gaps_found_per_site']
            det[site] = det.get(site, 0) + 1

        except Exception as e:
            log.debug(f"[AUDIT] Failed to record gap ({site} {gap_start}-{gap_end}): {e}")

    def _finalise_audit_log(self, df: pd.DataFrame):
        """
        Write run_summary and inputs_snapshot into the audit log, then
        persist to a JSON file in self.audit_log_dir.
        """
        try:
            # Snapshot the lookups that were actually used
            self._audit_log['inputs_snapshot']['peer_ratios'] = {
                k: round(float(v), 6) for k, v in self._peer_ratios.items()
            }
            self._audit_log['inputs_snapshot']['uncertainty_bounds'] = {
                k: {'lower': round(float(v[0]), 4), 'upper': round(float(v[1]), 4)}
                for k, v in self._uncertainty_bounds.items()
            }
            self._audit_log['inputs_snapshot']['peer_correlations'] = {
                s1: {s2: round(float(c), 4) for s2, c in inner.items()}
                for s1, inner in self._multi_site_correlations.items()
            }

            # Run summary
            all_gaps = self._audit_log['gaps']
            methods_used = list(dict.fromkeys(g['method'] for g in all_gaps))
            confidences = [g['confidence'] for g in all_gaps if g['confidence'] is not None]
            low_conf = [g for g in all_gaps if g['confidence'] is not None and g['confidence'] < 0.5]

            self._audit_log['run_summary'] = {
                'run_completed_at': datetime.now().isoformat(),
                'total_gaps_processed': len(all_gaps),
                'total_sites_affected': len(self._audit_log['detection_summary']['gaps_found_per_site']),
                'methods_used': methods_used,
                'mean_confidence': round(float(np.mean(confidences)), 4) if confidences else None,
                'min_confidence': round(float(np.min(confidences)), 4) if confidences else None,
                'low_confidence_gap_count': len(low_conf),
                'raw_anomaly_corrections_count': len(self._audit_log.get('raw_anomaly_corrections', [])),
                'zero_fill_corrections_count': len(self._audit_log['zero_fill_corrections']),
                'dst_events_count': len(self._audit_log['detection_summary']['dst_events']),
                'nan_counts_after': {
                    site: int(df[site].isna().sum())
                    for site in self.site_cols if site in df.columns
                },
            }

            # Persist to JSON
            os.makedirs(self.audit_log_dir, exist_ok=True)
            run_id = self._audit_log['run_id']
            filepath = os.path.join(self.audit_log_dir, f'imputation_audit_{run_id}.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(_json_safe(self._audit_log), f, indent=2, ensure_ascii=False)

            log.info(
                f"[AUDIT] Run {run_id} complete. "
                f"{len(all_gaps)} gaps, mean confidence "
                f"{self._audit_log['run_summary']['mean_confidence']}. "
                f"Saved to {filepath}"
            )

        except Exception as e:
            log.error(f"[AUDIT] Failed to finalise audit log: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # NEW v2: Zero-fill detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_raw_anomalies(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        PRE-FILL anomaly detection: scan raw input values (before any gap-filling)
        for implausible readings that are NOT explicit NaN — stuck sensors, flat
        lines, near-zero readings on active meters, and statistical outliers.

        Detected rows are converted to NaN in-place so the normal routing pipeline
        treats them as gaps and fills them with the appropriate method.

        Three complementary detectors run in sequence per site:

        1. STUCK-SENSOR  — a run of ≥ stuck_run_min consecutive identical (or
           near-identical) non-zero values.  E.g. a meter frozen at 42.0 kW for
           3+ hours signals a communication fault, not real consumption.
           Threshold: consecutive readings with |val - val[i-1]| < 0.01% of the
           site median for ≥ stuck_run_min steps.

        2. NEAR-ZERO     — a non-NaN value below 1% of the site median during
           hours that are historically active (work_hours / evening).  A meter
           reading 0 during a busy Tuesday afternoon is implausible.
           Not applied to weekend or holiday rows (legitimate low-consumption).

        3. ISOLATION FOREST — global statistical outlier detection fitted on the
           valid (non-stuck, non-zero) portion of each site column.  Flags the
           contamination% most anomalous single-point spikes or dips.  Uses a
           2-feature vector: [value, hour_of_day] so the model understands that
           a high reading at 14:00 is normal but the same value at 03:00 is not.
           Falls back to a 3-σ z-score check if IsolationForest is unavailable.

        All three detectors contribute to a unified anomaly_mask per site.
        Overlapping detections are merged before NaN-conversion so each
        contiguous bad run is logged as a single audit entry.

        Returns a dict  { site: [ {anomaly_record}, … ] }  which is merged into
        self._audit_log['raw_anomaly_corrections'].
        """
        MIN_VALID_ROWS = 30          # need enough data to fit the model
        STUCK_RUN_MIN = 18           # 3 hours of identical readings = stuck
        NEAR_ZERO_PCT = 0.01         # below 1% of median = suspect zero
        ACTIVE_OCCUPANCY = {'work_hours', 'evening'}

        all_corrections: Dict[str, List[Dict]] = {}

        for site in self.site_cols:
            if site not in df.columns:
                continue

            col = df[site].copy()
            valid_mask = col.notna()
            valid_vals = col[valid_mask]

            if len(valid_vals) < MIN_VALID_ROWS:
                log.debug(f"[RAW-ANOMALY] {site}: too few valid rows ({len(valid_vals)}), skipping")
                continue

            site_median = float(valid_vals.median())
            site_std = float(valid_vals.std())
            if site_median == 0:
                log.debug(f"[RAW-ANOMALY] {site}: site median is 0, skipping")
                continue

            n = len(df)
            anomaly_mask = np.zeros(n, dtype=bool)
            detection_reasons = [''] * n   # track which detector flagged each row

            # ── 1. STUCK-SENSOR DETECTION ────────────────────────────────────
            stuck_threshold = site_median * 0.0001  # 0.01% tolerance
            idx_valid = np.where(valid_mask.values)[0]
            vals_valid = col.values[idx_valid]

            run_start_i = 0
            for k in range(1, len(vals_valid)):
                if abs(vals_valid[k] - vals_valid[k - 1]) <= stuck_threshold:
                    continue
                run_len = k - run_start_i
                if run_len >= STUCK_RUN_MIN:
                    # Flag the run (keep the first value; it may be legitimate)
                    for r in range(run_start_i + 1, k):
                        row_idx = int(idx_valid[r])
                        anomaly_mask[row_idx] = True
                        detection_reasons[row_idx] = 'stuck_sensor'
                run_start_i = k
            # Handle the tail
            run_len = len(vals_valid) - run_start_i
            if run_len >= STUCK_RUN_MIN:
                for r in range(run_start_i + 1, len(vals_valid)):
                    row_idx = int(idx_valid[r])
                    anomaly_mask[row_idx] = True
                    detection_reasons[row_idx] = 'stuck_sensor'

            # ── 2. NEAR-ZERO DETECTION ───────────────────────────────────────
            zero_threshold = site_median * NEAR_ZERO_PCT
            if 'OccupancyType' in df.columns:
                active_rows = df['OccupancyType'].isin(ACTIVE_OCCUPANCY)
            else:
                # No occupancy column yet — flag all non-NaN near-zeros
                active_rows = pd.Series(True, index=df.index)

            near_zero_mask = (
                valid_mask &
                active_rows &
                (col <= zero_threshold) &
                (~anomaly_mask)  # don't double-count stuck sensor zeros
            )
            anomaly_mask |= near_zero_mask.values
            for row_idx in np.where(near_zero_mask.values)[0]:
                if not detection_reasons[row_idx]:
                    detection_reasons[row_idx] = 'near_zero'

            # ── 3. ISOLATION FOREST (or z-score fallback) ────────────────────
            # Fit only on rows that passed the first two detectors and are valid.
            clean_mask = valid_mask.values & ~anomaly_mask
            clean_indices = np.where(clean_mask)[0]

            if len(clean_indices) >= MIN_VALID_ROWS:
                clean_vals = col.values[clean_indices]
                clean_hours = df['Hour'].values[clean_indices] if 'Hour' in df.columns else np.zeros(len(clean_indices))
                # 2-feature: [value, hour] — lets the model learn that high values
                # at night are anomalous even if the value itself is plausible by day
                X_clean = np.column_stack([clean_vals, clean_hours])

                if_flagged = np.zeros(len(clean_indices), dtype=bool)

                if _IF_AVAILABLE and IsolationForest is not None:
                    try:
                        clf = IsolationForest(
                            contamination=self.zero_fill_contamination,
                            random_state=42,
                            n_estimators=100,
                        )
                        clf.fit(X_clean)
                        preds = clf.predict(X_clean)
                        if_flagged = preds == -1
                        log.debug(f"[RAW-ANOMALY] {site}: IsolationForest flagged "
                                  f"{if_flagged.sum()} outliers out of {len(clean_indices)} clean rows")
                    except Exception as e:
                        log.warning(f"[RAW-ANOMALY] {site}: IsolationForest failed: {e}. "
                                    f"Using z-score fallback.")
                        if_flagged = np.abs((clean_vals - site_median) / (site_std + 1e-9)) > 3.0
                else:
                    if_flagged = np.abs((clean_vals - site_median) / (site_std + 1e-9)) > 3.0
                    log.debug(f"[RAW-ANOMALY] {site}: z-score fallback flagged {if_flagged.sum()} outliers")

                for k, row_idx in enumerate(clean_indices):
                    if if_flagged[k]:
                        anomaly_mask[row_idx] = True
                        detection_reasons[row_idx] = 'isolation_forest_outlier'

            # ── Apply: convert flagged rows to NaN ───────────────────────────
            flagged_row_indices = np.where(anomaly_mask)[0]
            if len(flagged_row_indices) == 0:
                continue

            # Group into contiguous runs for clean audit records
            anomaly_runs: List[Tuple[int, int]] = []
            run_s = int(flagged_row_indices[0])
            prev = int(flagged_row_indices[0])
            for ri in flagged_row_indices[1:]:
                ri = int(ri)
                if ri != prev + 1:
                    anomaly_runs.append((run_s, prev + 1))
                    run_s = ri
                prev = ri
            anomaly_runs.append((run_s, prev + 1))

            site_corrections = []
            for run_start, run_end in anomaly_runs:
                original_vals = df.loc[run_start:run_end - 1, site].tolist()
                reasons_in_run = list(dict.fromkeys(
                    detection_reasons[i] for i in range(run_start, run_end) if detection_reasons[i]
                ))
                correction_record = {
                    'site': site,
                    'run_start_index': run_start,
                    'run_end_index': run_end,
                    'run_size_steps': run_end - run_start,
                    'run_size_minutes': (run_end - run_start) * 10,
                    'first_timestamp': _ts(df.loc[run_start, 'Timestamp']),
                    'last_timestamp': _ts(df.loc[run_end - 1, 'Timestamp']),
                    'original_values': [
                        round(float(v), 4) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None
                        for v in original_vals
                    ],
                    'detection_reasons': reasons_in_run,
                    'site_median_at_detection': round(site_median, 4),
                    'zero_threshold_used': round(zero_threshold, 4),
                }
                # Overwrite with NaN — the router will fill them in the gap loop
                df.loc[run_start:run_end - 1, site] = np.nan
                site_corrections.append(correction_record)

            if site_corrections:
                n_flagged = sum(r['run_size_steps'] for r in site_corrections)
                log.info(
                    f"[RAW-ANOMALY] {site}: {len(site_corrections)} anomalous run(s), "
                    f"{n_flagged} steps converted to NaN "
                    f"(stuck={sum(1 for r in site_corrections if 'stuck_sensor' in r['detection_reasons'])}, "
                    f"near_zero={sum(1 for r in site_corrections if 'near_zero' in r['detection_reasons'])}, "
                    f"if_outlier={sum(1 for r in site_corrections if 'isolation_forest_outlier' in r['detection_reasons'])})"
                )
                all_corrections[site] = site_corrections

        # Merge into audit log
        self._audit_log.setdefault('raw_anomaly_corrections', [])
        for site_corrections in all_corrections.values():
            self._audit_log['raw_anomaly_corrections'].extend(site_corrections)

        return all_corrections

    def _detect_zero_fills(self, df: pd.DataFrame, imputed_gaps: List[Dict]):
        """
        After all gap-filling, use IsolationForest (or z-score fallback) to
        detect implausible near-zero fills in the imputed regions.

        For each flagged region:
          1. Log it in audit_log['zero_fill_corrections'].
          2. Reset those indices to NaN.
          3. Re-impute via _intelligent_router.

        A 'zero fill' is defined as a filled value below 1% of the site's
        historical (pre-gap) median, combined with an IsolationForest anomaly
        score of -1.
        """
        if not imputed_gaps:
            return

        for site in self.site_cols:
            if site not in df.columns:
                continue

            valid_pre = df[site].dropna()
            if len(valid_pre) < 20:
                continue

            site_median = float(valid_pre.median())
            zero_threshold = site_median * 0.01  # below 1% of median = suspected zero fill

            # Collect all imputed indices for this site
            site_gaps = [g for g in imputed_gaps if g['site'] == site]
            if not site_gaps:
                continue

            imputed_indices = []
            for g in site_gaps:
                imputed_indices.extend(range(g['gap_start'], g['gap_end']))
            imputed_indices = sorted(set(imputed_indices))

            if not imputed_indices:
                continue

            imputed_vals = df.loc[imputed_indices, site].values

            # Anomaly detection
            anomaly_mask = np.zeros(len(imputed_vals), dtype=bool)

            if _IF_AVAILABLE and IsolationForest is not None:
                try:
                    # Fit on valid non-imputed data, score imputed values
                    valid_data = valid_pre.values.reshape(-1, 1)
                    clf = IsolationForest(
                        contamination=self.zero_fill_contamination,
                        random_state=42,
                        n_estimators=100,
                    )
                    clf.fit(valid_data)
                    scores = clf.predict(imputed_vals.reshape(-1, 1))
                    anomaly_mask = (scores == -1) & (imputed_vals <= zero_threshold)
                    log.debug(f"[ZERO-FILL] IsolationForest detected {anomaly_mask.sum()} suspect zeros in {site}")
                except Exception as e:
                    log.warning(f"[ZERO-FILL] IsolationForest failed for {site}: {e}. Using z-score fallback.")
                    _if_failed = True
                else:
                    _if_failed = False
            else:
                _if_failed = True

            if _if_failed:
                # Z-score fallback: flag values more than 3 std below mean AND near zero
                mean_val = float(valid_pre.mean())
                std_val = float(valid_pre.std())
                if std_val > 0:
                    z_scores = (imputed_vals - mean_val) / std_val
                    anomaly_mask = (z_scores < -3) & (imputed_vals <= zero_threshold)
                log.debug(f"[ZERO-FILL] Z-score fallback: {anomaly_mask.sum()} suspect zeros in {site}")

            # Process flagged regions
            flagged_indices = [imputed_indices[i] for i in range(len(imputed_indices)) if anomaly_mask[i]]

            if not flagged_indices:
                continue

            # Group into contiguous runs
            flagged_gaps = []
            run_start = flagged_indices[0]
            prev = flagged_indices[0]
            for idx in flagged_indices[1:]:
                if idx != prev + 1:
                    flagged_gaps.append((run_start, prev + 1))
                    run_start = idx
                prev = idx
            flagged_gaps.append((run_start, prev + 1))

            log.info(f"[ZERO-FILL] {site}: {len(flagged_gaps)} suspect zero-fill region(s) → re-imputing")

            for zf_start, zf_end in flagged_gaps:
                original_vals = df.loc[zf_start:zf_end - 1, site].tolist()

                correction_record = {
                    'site': site,
                    'gap_start': int(zf_start),
                    'gap_end': int(zf_end),
                    'gap_size': int(zf_end - zf_start),
                    'first_timestamp': _ts(df.loc[zf_start, 'Timestamp']),
                    'last_timestamp': _ts(df.loc[zf_end - 1, 'Timestamp']),
                    'original_filled_values': [round(float(v), 4) if not np.isnan(v) else None for v in original_vals],
                    'detection_method': 'isolation_forest' if not _if_failed else 'zscore_fallback',
                    'zero_threshold_used': round(zero_threshold, 4),
                }

                # Reset and re-impute
                df.loc[zf_start:zf_end - 1, site] = np.nan
                self._intelligent_router(df, site, zf_start, zf_end, zero_fill_correction=True)

                new_vals = df.loc[zf_start:zf_end - 1, site].tolist()
                correction_record['new_filled_values'] = [
                    round(float(v), 4) if not np.isnan(v) else None for v in new_vals
                ]
                self._audit_log['zero_fill_corrections'].append(correction_record)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def impute(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main entry point.  Returns a fully-imputed dataframe with tz-aware
        timestamps.  Saves a JSON audit log to self.audit_log_dir after each run.
        """
        df = df.copy().sort_values('Timestamp').reset_index(drop=True)

        # ── NEW v2: localise timestamps first ────────────────────────────────
        df = self._localise_timestamps(df)

        if weather_df is not None:
            self.weather_df = weather_df
            df = self._merge_weather(df)

        # ── NEW v2: initialise audit log (after tz conversion) ───────────────
        self._init_audit_log(df)

        # Reindex to complete 10-min grid (tz-aware)
        start = df['Timestamp'].min().floor('10min')
        end = df['Timestamp'].max().ceil('10min')
        full_index = pd.date_range(start, end, freq='10min', tz=df['Timestamp'].dt.tz)
        df = (
            df.set_index('Timestamp')
              .reindex(full_index)
              .reset_index()
        )
        df.columns = ['Timestamp'] + list(df.columns[1:])

        # Log reindex gap detection
        for site in self.site_cols:
            if site in df.columns:
                n_gaps = int(df[site].isna().sum())
                self._audit_log['detection_summary']['gaps_found_per_site'][site] = 0
                if n_gaps > 0:
                    log.info(f"[DETECT] {site}: {n_gaps} missing steps after reindex")

        # Feature engineering
        self._add_datetime_features(df)
        self._fill_airtemp_forward(df)
        self._classify_thermal_regimes(df)
        self._add_occupancy_features(df)
        self._add_external_features(df)

        # ── NEW v2: pre-fill anomaly detection (stuck sensors, near-zeros, IF outliers)
        # Must run AFTER feature engineering so occupancy columns are available,
        # and BEFORE template building so templates are not contaminated by bad values.
        self._detect_raw_anomalies(df)

        # Build lookups
        self._build_peer_ratios(df)
        self._build_day_specific_templates(df)
        self._build_seasonal_templates(df)
        if self.use_multi_week_templates:
            self._build_weekly_templates(df)
            self._build_uncertainty_bounds(df)
        self._build_multi_site_correlations(df)

        # Track all imputed gap regions for zero-fill detection later
        imputed_gaps: List[Dict] = []

        # Gap filling
        for site in self.site_cols:
            if site not in df.columns:
                continue
            gap_mask = df[site].isna()
            if not gap_mask.any():
                continue
            for gap_start, gap_end in self._find_gap_groups(gap_mask):
                gap_size = gap_end - gap_start
                imputed_gaps.append({'site': site, 'gap_start': gap_start, 'gap_end': gap_end})
                if self.use_chunked_recovery and gap_size > self.gap_chunk_size:
                    self._fill_chunked_gap(df, site, gap_start, gap_end)
                else:
                    self._intelligent_router(df, site, gap_start, gap_end)

        # NaN guard
        self._nan_guard_final_pass(df)

        # ── NEW v2: zero-fill detection and correction ───────────────────────
        self._detect_zero_fills(df, imputed_gaps)

        # Smoothing
        for site in self.site_cols:
            if site in df.columns:
                self._smooth_junctions(df, site)

        # ── NEW v2: finalise and save audit log ──────────────────────────────
        self._finalise_audit_log(df)

        out_cols = ['Timestamp'] + self.site_cols + (['AirTemp'] if 'AirTemp' in df.columns else [])
        return df[[c for c in out_cols if c in df.columns]]

    # ─────────────────────────────────────────────────────────────────────────
    # Public accessors for audit data
    # ─────────────────────────────────────────────────────────────────────────

    def get_audit_log(self) -> Dict:
        """Return the full audit log dict from the most recent impute() run."""
        return self._audit_log

    def get_run_summary(self) -> Dict:
        """Return just the flat run-level summary from the most recent run."""
        return self._audit_log.get('run_summary', {})

    def get_gap_log(self) -> pd.DataFrame:
        """Return a DataFrame with one row per imputed gap (detailed view)."""
        gaps = self._audit_log.get('gaps', [])
        if not gaps:
            return pd.DataFrame()
        rows = []
        for g in gaps:
            rows.append({
                'site':                g['site'],
                'first_missing':       g['first_missing_timestamp'],
                'last_missing':        g['last_missing_timestamp'],
                'gap_size_steps':      g['gap_size_steps'],
                'gap_size_minutes':    g['gap_size_minutes'],
                'unique_days':         ', '.join(g['unique_days_in_gap']),
                'occupancy_types':     ', '.join(dict.fromkeys(g['occupancy_types'])),
                'is_holiday':          any(g['is_holiday_flags']),
                'method':              g['method'],
                'confidence':          g['confidence'],
                'zero_fill_corrected': g['zero_fill_corrected'],
                'routing_trace':       ' → '.join(g['routing_trace']),
            })
        return pd.DataFrame(rows)

    def get_zero_fill_report(self) -> pd.DataFrame:
        """Return a DataFrame of zero-fill corrections made during the last run."""
        corrections = self._audit_log.get('zero_fill_corrections', [])
        if not corrections:
            return pd.DataFrame()
        return pd.DataFrame(corrections)

    def get_raw_anomaly_report(self) -> pd.DataFrame:
        """
        Return a DataFrame of raw input anomalies detected and converted to NaN
        before gap-filling (stuck sensors, near-zeros, IsolationForest outliers).
        Each row is one contiguous anomalous run.
        """
        corrections = self._audit_log.get('raw_anomaly_corrections', [])
        if not corrections:
            return pd.DataFrame()
        rows = []
        for r in corrections:
            rows.append({
                'site':            r['site'],
                'first_timestamp': r['first_timestamp'],
                'last_timestamp':  r['last_timestamp'],
                'run_size_steps':  r['run_size_steps'],
                'run_size_minutes':r['run_size_minutes'],
                'detection_reasons': ', '.join(r['detection_reasons']),
                'site_median':     r['site_median_at_detection'],
                'zero_threshold':  r['zero_threshold_used'],
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Chunked gap recovery
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_chunked_gap(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        gap_size = gap_end - gap_start
        if self.use_smart_chunking:
            chunks = self._get_smart_chunks(df, gap_start, gap_end)
            log.info(f"[CHUNKED-SMART] {site}: {gap_size} steps → {len(chunks)} chunks")
        else:
            n = int(np.ceil(gap_size / self.gap_chunk_size))
            chunks = [
                (gap_start + i * self.gap_chunk_size,
                 min(gap_start + (i + 1) * self.gap_chunk_size, gap_end))
                for i in range(n)
            ]
            log.info(f"[CHUNKED] {site}: {gap_size} steps → {n} chunks")

        for chunk_idx, (cs, ce) in enumerate(chunks):
            if self.use_multi_week_templates and self._weekly_templates:
                self._fill_with_multi_week_template(df, site, cs, ce, chunk_index=chunk_idx)
            else:
                self._intelligent_router(df, site, cs, ce, chunk_index=chunk_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Smart chunking
    # ─────────────────────────────────────────────────────────────────────────

    def _get_smart_chunks(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> List[Tuple[int, int]]:
        chunks = []
        current_chunk_start = gap_start
        current_chunk_size = 0
        max_chunk_size = self.gap_chunk_size * 1.5

        gap_size = gap_end - gap_start
        gap_dates = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values
        gap_hours = df.loc[gap_start:gap_end - 1, 'Hour'].values

        reference_site = next((s for s in self.site_cols if s in self._day_variance), None)
        site_day_variance = self._day_variance.get(reference_site, {}) if reference_site else {}

        if site_day_variance:
            all_variances = list(site_day_variance.values())
            v_threshold = float(np.median(all_variances))
            high_variance_days = {
                day for day, var in site_day_variance.items() if var > v_threshold * 1.3
            }
        else:
            v_threshold = 100.0
            high_variance_days = set()

        for i in range(gap_size):
            current_chunk_size += 1
            idx = gap_start + i
            day_name = gap_dates[i]
            hour = gap_hours[i]
            is_high_variance = day_name in high_variance_days
            is_new_day = (hour == 0) and (idx > gap_start)

            should_break = (
                current_chunk_size >= max_chunk_size
                or (is_high_variance and current_chunk_size > self.gap_chunk_size * 0.8)
                or (is_new_day and current_chunk_size >= self.gap_chunk_size)
            )
            if should_break and idx < gap_end:
                chunks.append((current_chunk_start, idx))
                current_chunk_start = idx
                current_chunk_size = 0

        if current_chunk_start < gap_end:
            chunks.append((current_chunk_start, gap_end))

        log.debug(f"[SMART-CHUNK] {len(chunks)} chunks, high-var threshold: {v_threshold:.1f}")
        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-week template building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_weekly_templates(self, df: pd.DataFrame):
        try:
            max_date = df['Timestamp'].max()
            min_date = max_date - pd.Timedelta(days=self.template_lookback_days)
            recent_cutoff = max_date - pd.Timedelta(
                days=int(self.template_lookback_days * self.adaptive_template_bias)
            )
            recent_mask = df['Timestamp'] >= recent_cutoff
            historical_mask = (df['Timestamp'] >= min_date) & (df['Timestamp'] < recent_cutoff)

            for site in self.site_cols:
                if site not in df.columns:
                    continue
                self._weekly_templates[site] = {}
                self._day_variance[site] = {}

                for day_name in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
                    day_templates = {}
                    day_values_all = []

                    for hour in range(24):
                        recent = df.loc[
                            recent_mask & (df['DayOfWeek'] == day_name) &
                            (df['Hour'] == hour) & df[site].notna(), site
                        ].values
                        historical = df.loc[
                            historical_mask & (df['DayOfWeek'] == day_name) &
                            (df['Hour'] == hour) & df[site].notna(), site
                        ].values

                        if len(recent) > 0 and len(historical) > 0:
                            blended = (self.adaptive_template_bias * np.median(recent) +
                                       (1 - self.adaptive_template_bias) * np.median(historical))
                            all_values = np.concatenate([recent, historical])
                        elif len(recent) > 0:
                            blended = np.median(recent)
                            all_values = recent
                        elif len(historical) > 0:
                            blended = np.median(historical)
                            all_values = historical
                        else:
                            blended = None
                            all_values = np.array([])

                        if len(all_values) > 0:
                            day_templates[hour] = {
                                'median': blended,
                                'mean': float(np.mean(all_values)),
                                'std': float(np.std(all_values)),
                                'q25': float(np.percentile(all_values, 25)),
                                'q75': float(np.percentile(all_values, 75)),
                                'recent_count': int(len(recent)),
                                'historical_count': int(len(historical)),
                                'all_values': all_values[:100].copy(),
                            }
                            day_values_all.extend(all_values)
                        else:
                            day_templates[hour] = None

                    self._weekly_templates[site][day_name] = day_templates
                    if day_values_all:
                        self._day_variance[site][day_name] = float(np.std(day_values_all))

            log.info(f"[TEMPLATES] Built {self.template_lookback_days}-day adaptive templates "
                     f"({int(self.adaptive_template_bias*100)}% recency bias)")
        except Exception as e:
            log.error(f"[ERROR] _build_weekly_templates: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-week template fill
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_with_multi_week_template(
        self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int,
        chunk_index: Optional[int] = None,
    ):
        routing_trace = ['MULTI_WEEK_TEMPLATE_LOOKUP']
        try:
            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            day_names = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values
            filled_vals = []
            confidences = []

            for h, day_name in zip(hours, day_names):
                tmpl = self._weekly_templates.get(site, {}).get(day_name, {}).get(h)
                if tmpl is not None:
                    filled_vals.append(tmpl['median'])
                    conf = (1.0 / (1.0 + tmpl['std'] / (tmpl['median'] + 1e-6))
                            if tmpl['std'] > 0 else 1.0)
                    confidences.append(conf)
                else:
                    routing_trace.append('HOURLY_MEDIAN_FALLBACK')
                    mask = (df['Hour'] == h) & df[site].notna()
                    if mask.any():
                        filled_vals.append(float(df.loc[mask, site].median()))
                        confidences.append(0.5)
                    else:
                        filled_vals.append(np.nan)
                        confidences.append(0.0)

            filled_vals = self._validate_and_clip(np.array(filled_vals), site, df)
            df.loc[gap_start:gap_end - 1, site] = filled_vals

            avg_conf = float(np.nanmean(confidences))
            self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = avg_conf

            if avg_conf < 0.5:
                self._low_confidence_flags.append({
                    'site': site, 'gap_start': gap_start, 'gap_end': gap_end,
                    'confidence': avg_conf, 'reason': 'multi_week_template_low_confidence',
                })

            self.strategy_log.append({
                'site': site, 'gap_start': gap_start, 'gap_end': gap_end,
                'gap_size': gap_end - gap_start, 'strategy': 'MULTI_WEEK_TEMPLATE',
                'confidence': avg_conf,
            })

            self._record_gap(df, site, gap_start, gap_end, 'MULTI_WEEK_TEMPLATE', avg_conf,
                             routing_trace, is_chunk=(chunk_index is not None),
                             chunk_index=chunk_index)

        except Exception as e:
            log.error(f"[ERROR] _fill_with_multi_week_template ({site}): {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    # ─────────────────────────────────────────────────────────────────────────
    # Occupancy features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_occupancy_features(self, df: pd.DataFrame):
        try:
            weekend_mask = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

            if self.calendar_data is not None:
                holiday_dates = pd.to_datetime(self.calendar_data.get('holiday_list', [])).dt.date
                holiday_mask = df['Date'].isin(holiday_dates)
            else:
                holiday_mask = df['Date'].isin(FRANCE_HOLIDAYS_2026)

            evening_mask = (df['Hour'] >= 18) | (df['Hour'] < 6)
            plain_weekday_mask = ~weekend_mask & ~holiday_mask
            evening_weekday_mask = evening_mask & plain_weekday_mask

            df['OccupancyType'] = 'work_hours'
            df['IsOccupied'] = True
            df.loc[weekend_mask, 'OccupancyType'] = 'weekend'
            df.loc[weekend_mask, 'IsOccupied'] = False
            df.loc[holiday_mask, 'OccupancyType'] = 'holiday'
            df.loc[holiday_mask, 'IsOccupied'] = False
            df.loc[evening_weekday_mask, 'OccupancyType'] = 'evening'
            df.loc[evening_weekday_mask, 'IsOccupied'] = False

            log.info("[FEATURES] Occupancy features added")
        except Exception as e:
            log.error(f"[ERROR] _add_occupancy_features: {e}")
            df['IsOccupied'] = True
            df['OccupancyType'] = 'unknown'

    # ─────────────────────────────────────────────────────────────────────────
    # External features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_external_features(self, df: pd.DataFrame):
        try:
            df['IsHoliday'] = df['Date'].isin(FRANCE_HOLIDAYS_2026)
            df['IsSpecialDay'] = (
                df['Date'].isin(FRANCE_CLOSE_DAYS_2026) |
                df['Date'].isin(FRANCE_SPECIAL_DAYS_2026)
            )
            df['IsHolidayClose'] = False
            for hd in FRANCE_HOLIDAYS_2026:
                mask = (
                    (df['Date'] >= hd - timedelta(days=1)) &
                    (df['Date'] <= hd + timedelta(days=1))
                )
                df.loc[mask, 'IsHolidayClose'] = True

            df['IsEventDay'] = False
            for site in self.site_cols:
                if site in df.columns:
                    daily_std = df.groupby('Date')[site].std()
                    thr = daily_std.quantile(0.75)
                    high_dates = daily_std[daily_std > thr].index
                    df.loc[df['Date'].isin(high_dates), 'IsEventDay'] = True

            df['WeatherSpike'] = False
            if 'AirTemp' in df.columns:
                daily_range = df.groupby('Date')['AirTemp'].apply(
                    lambda x: abs(x.max() - x.min()) if len(x) > 0 else 0
                )
                thr = daily_range.quantile(0.80)
                spike_dates = daily_range[daily_range > thr].index
                df.loc[df['Date'].isin(spike_dates), 'WeatherSpike'] = True

            df['Season'] = df['Timestamp'].dt.month.map({
                12:'winter',1:'winter',2:'winter',
                3:'spring',4:'spring',5:'spring',
                6:'summer',7:'summer',8:'summer',
                9:'fall',10:'fall',11:'fall',
            })
            log.info("[FEATURES] External features added")
        except Exception as e:
            log.error(f"[ERROR] _add_external_features: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Uncertainty bounds
    # ─────────────────────────────────────────────────────────────────────────

    def _build_uncertainty_bounds(self, df: pd.DataFrame):
        try:
            for site in self.site_cols:
                if site not in df.columns:
                    continue
                valid = df[site].dropna()
                if len(valid) < 10:
                    self._uncertainty_bounds[site] = (0.0, float(valid.max()) * 1.5)
                    continue
                mean, std = float(valid.mean()), float(valid.std())
                self._uncertainty_bounds[site] = (max(0.0, mean - 2 * std), mean + 2 * std)
            log.info("[BOUNDS] Uncertainty bounds built")
        except Exception as e:
            log.error(f"[ERROR] _build_uncertainty_bounds: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_confidence_with_uncertainty(
        self, site: str, gap_size: int, day_type: str, strategy: str,
        occupancy_type: Optional[str] = None, is_holiday: bool = False,
    ) -> float:
        strategy_map = {
            'MULTI_WEEK_TEMPLATE': 0.90, 'MICE': 0.85, 'KNN_CONTEXT': 0.80,
            'KALMAN_FILTER': 0.75, 'PEER_CORRELATION': 0.75, 'THERMAL_TEMPLATE': 0.70,
            'ENHANCED_TEMPLATE': 0.65, 'WEEKEND_TEMPLATE': 0.60,
            'LINEAR_SHORT': 0.50, 'SAFE_LINEAR_MEDIAN': 0.45,
            'LINEAR_MICRO': 0.40, 'SAFE_MEDIAN': 0.30,
        }
        sf = strategy_map.get(strategy, 0.5)
        gf = (1.0 if gap_size < 10 else 0.9 if gap_size < 50
              else 0.7 if gap_size < STEPS_PER_DAY
              else 0.5 if gap_size < STEPS_PER_DAY * 7 else 0.3)
        of = {'work_hours': 1.0, 'evening': 0.8, 'weekend': 0.7, 'holiday': 0.5}.get(
            occupancy_type or '', 0.6)
        hf = 0.7 if is_holiday else 1.0
        return float(np.clip(sf * gf * of * hf, 0.0, 1.0))

    def _flag_low_confidence(
        self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int,
        confidence: float, strategy: str,
    ):
        if confidence < 0.50:
            self._low_confidence_flags.append({
                'site': site, 'gap_start': gap_start, 'gap_end': gap_end,
                'gap_size': gap_end - gap_start, 'confidence': confidence,
                'strategy': strategy,
                'occupancy_type': df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None,
                'is_holiday': bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False,
                'reason': 'low_confidence_score',
            })

    # ─────────────────────────────────────────────────────────────────────────
    # Intelligent router (updated to write audit records + routing trace)
    # ─────────────────────────────────────────────────────────────────────────

    def _intelligent_router(
        self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int,
        chunk_index: Optional[int] = None,
        zero_fill_correction: bool = False,
    ):
        """Route each gap and record a full audit entry with routing trace."""
        gap_size = gap_end - gap_start
        meter_tier = METER_HIERARCHY.get(site, {}).get('tier', 'unknown')
        is_weekend = self._is_weekend_gap(df, gap_start, gap_end)
        is_submeter = meter_tier == 'sub'
        is_entry = meter_tier == 'entry'
        routing_trace: List[str] = [f'START: size={gap_size}, tier={meter_tier}']

        def _log_and_record(strategy: str) -> float:
            day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)
            occ = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None
            hol = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False
            conf = self._calculate_confidence_with_uncertainty(site, gap_size, day_type, strategy, occ, hol)
            self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = conf
            self._flag_low_confidence(df, site, gap_start, gap_end, conf, strategy)
            self.strategy_log.append({
                'site': site, 'gap_start': gap_start, 'gap_end': gap_end,
                'gap_size': gap_size, 'strategy': strategy, 'confidence': conf,
                'day_type': day_type, 'occupancy_type': occ, 'is_holiday': hol,
            })
            routing_trace.append(f'SELECTED: {strategy} (confidence={round(conf, 3)})')
            self._record_gap(df, site, gap_start, gap_end, strategy, conf, routing_trace,
                             is_chunk=(chunk_index is not None), chunk_index=chunk_index,
                             zero_fill_corrected=zero_fill_correction)
            return conf

        # ML cascade
        if gap_size > 50 and not is_entry:
            routing_trace.append('BRANCH: ML_CASCADE (size>50, non-entry)')
            if self.use_mice:
                routing_trace.append('TRY: MICE')
                if self._fill_with_mice(df, site, gap_start, gap_end):
                    _log_and_record('MICE')
                    return
                routing_trace.append('FAIL: MICE')
            if self.use_kalman:
                routing_trace.append('TRY: KALMAN_FILTER')
                if self._fill_with_kalman_filter(df, site, gap_start, gap_end):
                    _log_and_record('KALMAN_FILTER')
                    return
                routing_trace.append('FAIL: KALMAN_FILTER')
            if self.use_knn:
                routing_trace.append('TRY: KNN_CONTEXT')
                if self._fill_with_knn_context(df, site, gap_start, gap_end, k=5):
                    _log_and_record('KNN_CONTEXT')
                    return
                routing_trace.append('FAIL: KNN_CONTEXT')

        # Sub-meter peer correlation
        if is_submeter:
            routing_trace.append('BRANCH: PEER_CORRELATION (sub-meter)')
            parent = METER_HIERARCHY[site]['parent']
            if parent and parent in df.columns:
                if df.loc[gap_start:gap_end - 1, parent].notna().any():
                    self._fill_via_peer_correlation(df, site, parent, gap_start, gap_end)
                    _log_and_record('PEER_CORRELATION')
                    return
                routing_trace.append('SKIP: parent has no data in gap window')

        # Rule-based by size
        if gap_size <= 3:
            routing_trace.append('BRANCH: LINEAR_MICRO (size<=3)')
            self._fill_linear(df, site, gap_start, gap_end)
            _log_and_record('LINEAR_MICRO')
        elif gap_size <= 18:
            routing_trace.append('BRANCH: LINEAR_SHORT (size<=18)')
            self._fill_linear(df, site, gap_start, gap_end)
            _log_and_record('LINEAR_SHORT')
        elif self._is_pure_weekend_gap(df, gap_start, gap_end):
            day_type = self._get_weekend_day_type(df, gap_start, gap_end)
            routing_trace.append(f'BRANCH: WEEKEND_TEMPLATE ({day_type})')
            self._fill_weekend_template(df, site, gap_start, gap_end, day_type)
            _log_and_record(f'WEEKEND_TEMPLATE_{day_type.upper()}')
        elif not is_entry and gap_size <= STEPS_PER_DAY:
            routing_trace.append('BRANCH: THERMAL_TEMPLATE (non-entry, <=1 day)')
            self._fill_with_thermal_template(df, site, gap_start, gap_end)
            _log_and_record('THERMAL_TEMPLATE')
        elif not is_entry:
            routing_trace.append('BRANCH: ENHANCED_TEMPLATE (non-entry, >1 day)')
            self._fill_enhanced_template(df, site, gap_start, gap_end)
            _log_and_record('ENHANCED_TEMPLATE')
        else:
            routing_trace.append('BRANCH: SAFE_LINEAR_MEDIAN (entry meter fallback)')
            self._fill_safe_linear_median(df, site, gap_start, gap_end)
            _log_and_record('SAFE_LINEAR_MEDIAN')

    # ─────────────────────────────────────────────────────────────────────────
    # Fill primitives (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_linear(self, df, site, gap_start, gap_end):
        try:
            left = df.loc[gap_start - 1, site] if gap_start > 0 else np.nan
            right = df.loc[gap_end, site] if gap_end < len(df) else np.nan
            if np.isnan(left) or np.isnan(right):
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                return
            df.loc[gap_start:gap_end - 1, site] = np.linspace(left, right, gap_end - gap_start + 2)[1:-1]
        except Exception as e:
            log.error(f"[ERROR] _fill_linear ({site}): {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_weekend_template(self, df, site, gap_start, gap_end, day_type):
        try:
            tmpl = self.templates.get(day_type.lower(), {}).get(site, {})
            if not tmpl:
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                return
            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            vals = self._validate_and_clip(
                np.array([tmpl.get(h, np.nan) for h in hours]), site, df)
            df.loc[gap_start:gap_end - 1, site] = vals
        except Exception as e:
            log.error(f"[ERROR] _fill_weekend_template ({site}): {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_with_thermal_template(self, df, site, gap_start, gap_end):
        try:
            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            vals = []
            for h in hours:
                mask = (df['Hour'] == h) & df[site].notna()
                vals.append(float(df.loc[mask, site].median()) if mask.any() else np.nan)
            df.loc[gap_start:gap_end - 1, site] = self._validate_and_clip(np.array(vals), site, df)
        except Exception as e:
            log.error(f"[ERROR] _fill_with_thermal_template ({site}): {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_enhanced_template(self, df, site, gap_start, gap_end):
        self._fill_with_thermal_template(df, site, gap_start, gap_end)

    def _fill_safe_linear_median(self, df, site, gap_start, gap_end):
        try:
            left = df.loc[gap_start - 1, site] if gap_start > 0 else np.nan
            right = df.loc[gap_end, site] if gap_end < len(df) else np.nan
            if np.isnan(left) or np.isnan(right):
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                return
            filled = np.linspace(left, right, gap_end - gap_start + 2)[1:-1]
            valid = df[site].dropna()
            if len(valid) > 0:
                filled = filled * 0.9 + float(valid.median()) * 0.1
            df.loc[gap_start:gap_end - 1, site] = filled
        except Exception as e:
            log.error(f"[ERROR] _fill_safe_linear_median ({site}): {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_safe_median_template(self, df, site, gap_start, gap_end):
        try:
            valid = df[site].dropna()
            df.loc[gap_start:gap_end - 1, site] = float(valid.median()) if len(valid) > 0 else 0.0
        except Exception as e:
            log.error(f"[ERROR] _fill_safe_median_template ({site}): {e}")
            df.loc[gap_start:gap_end - 1, site] = 0.0

    def _fill_via_peer_correlation(self, df, child, parent, gap_start, gap_end):
        try:
            ratio = self._peer_ratios.get(child, 0.5)
            vals = self._validate_and_clip(
                df.loc[gap_start:gap_end - 1, parent].values * ratio, child, df)
            df.loc[gap_start:gap_end - 1, child] = vals
        except Exception as e:
            log.error(f"[ERROR] _fill_via_peer_correlation ({child}): {e}")
            self._fill_safe_median_template(df, child, gap_start, gap_end)

    def _fill_with_mice(self, df, site, gap_start, gap_end, iterations=3):
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            cols = [site] + [c for c in self.site_cols if c != site and c in df.columns]
            imputed = IterativeImputer(max_iter=iterations, random_state=0).fit_transform(
                df.loc[:, cols].values)
            df.loc[gap_start:gap_end - 1, site] = imputed[gap_start:gap_end, 0]
            return True
        except Exception as e:
            log.error(f"[ERROR] _fill_with_mice ({site}): {e}")
            return False

    def _fill_with_kalman_filter(self, df, site, gap_start, gap_end):
        try:
            try:
                from pykalman import KalmanFilter
                values = df[site].values.copy()
                values[np.isnan(values)] = 0
                smoothed, _ = KalmanFilter().smooth(values)
                df.loc[gap_start:gap_end - 1, site] = smoothed[gap_start:gap_end, 0]
            except ImportError:
                df[site] = df[site].interpolate(method='linear')
            return True
        except Exception as e:
            log.error(f"[ERROR] _fill_with_kalman_filter ({site}): {e}")
            return False

    def _fill_with_knn_context(self, df, site, gap_start, gap_end, k=5):
        try:
            ctx = [c for c in self.site_cols if c != site and c in df.columns]
            if not ctx:
                return False
            valid = df[site].notna()
            if valid.sum() < k:
                return False
            nbrs = NearestNeighbors(n_neighbors=k).fit(df.loc[valid, ctx].values)
            y = df.loc[valid, site].values
            dists, idxs = nbrs.kneighbors(df.loc[gap_start:gap_end - 1, ctx].values)
            df.loc[gap_start:gap_end - 1, site] = np.array([y[i].mean() for i in idxs])
            return True
        except Exception as e:
            log.error(f"[ERROR] _fill_with_knn_context ({site}): {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Validation, cleanup, smoothing
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_and_clip(self, values: np.ndarray, site: str, df: pd.DataFrame) -> np.ndarray:
        values = np.array(values, dtype=float)
        nan_mask = np.isnan(values)
        valid = df[site].dropna()
        if np.any(nan_mask):
            values[nan_mask] = float(valid.median()) if len(valid) > 0 else 0.0
        if len(valid) > 0:
            values = np.clip(values, 0, float(valid.max()) * 1.5)
        return values

    def _nan_guard_final_pass(self, df: pd.DataFrame):
        for site in self.site_cols:
            if site not in df.columns:
                continue
            nan_mask = df[site].isna()
            if not nan_mask.any():
                continue
            for gs, ge in self._find_gap_groups(nan_mask):
                self._fill_safe_median_template(df, site, gs, ge)

    def _smooth_junctions(self, df: pd.DataFrame, site: str):
        try:
            gap_mask = df[site].isna()
            if not gap_mask.any():
                return
            for gap_start, gap_end in self._find_gap_groups(gap_mask):
                w = 21
                if gap_start > w // 2 and gap_start + w // 2 < len(df):
                    window = df.loc[gap_start - w//2:gap_start + w//2, site].values
                    if not np.all(np.isnan(window)):
                        try:
                            smoothed = savgol_filter(
                                pd.Series(window).ffill().bfill().values, w, 3)
                            valid_mask = ~pd.isna(
                                df.loc[gap_start - w//2:gap_start + w//2, site].values)
                            df.loc[gap_start - w//2:gap_start + w//2, site] = np.where(
                                valid_mask, smoothed,
                                df.loc[gap_start - w//2:gap_start + w//2, site].values)
                        except Exception:
                            pass
        except Exception as e:
            log.debug(f"[DEBUG] _smooth_junctions ({site}): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def _add_datetime_features(self, df: pd.DataFrame):
        """
        Extract Hour, DayOfWeek, and Date from tz-aware Timestamps.
        Hour is taken from local time (dt.hour on a tz-aware Series gives
        local wall-clock hour), so a 2 AM summer reading is correctly labelled
        hour=2, not hour=1 as it would be in UTC.
        """
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['Date'] = df['Timestamp'].dt.date

    def _fill_airtemp_forward(self, df: pd.DataFrame):
        if 'AirTemp' in df.columns:
            df['AirTemp'] = df['AirTemp'].ffill().bfill()

    def _merge_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.weather_df is None:
            return df
        weather = self.weather_df.copy()
        weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])
        df = df.merge(weather[['Timestamp', 'AirTemp']], on='Timestamp', how='left')
        if 'AirTemp' in df.columns:
            df['AirTemp'] = df['AirTemp'].ffill().bfill()
        return df

    def _classify_thermal_regimes(self, df: pd.DataFrame):
        if 'AirTemp' not in df.columns:
            df['ThermalRegime'] = 'Mild'
        else:
            df['ThermalRegime'] = pd.cut(
                df['AirTemp'], bins=[-np.inf, 10, 20, np.inf], labels=['Cold', 'Mild', 'Hot'])

    def _build_day_specific_templates(self, df: pd.DataFrame):
        self.templates = {'saturday': {}, 'sunday': {}, 'holiday': {}}
        for site in self.site_cols:
            if site not in df.columns:
                continue
            for day, key in [('Saturday', 'saturday'), ('Sunday', 'sunday')]:
                mask = (df['DayOfWeek'] == day) & df[site].notna()
                if mask.any():
                    self.templates[key][site] = df.loc[mask].groupby('Hour')[site].median().to_dict()
            if 'IsHoliday' in df.columns:
                mask = df['IsHoliday'] & df[site].notna()
                if mask.any():
                    self.templates['holiday'][site] = df.loc[mask].groupby('Hour')[site].median().to_dict()

    def _build_seasonal_templates(self, df: pd.DataFrame):
        for site in self.site_cols:
            if site in df.columns:
                self._seasonal_templates[site] = {}

    def _build_peer_ratios(self, df: pd.DataFrame):
        for site, info in METER_HIERARCHY.items():
            if info['tier'] != 'sub':
                continue
            parent = info['parent']
            if not parent or parent not in df.columns:
                continue
            mask = df[site].notna() & df[parent].notna()
            if mask.any():
                idxs = np.where(mask)[0][-min(STEPS_PER_DAY, mask.sum()):]
                c, p = df.loc[idxs, site].values, df.loc[idxs, parent].values
                nonzero = p > 1
                if nonzero.any():
                    ratios = c[nonzero] / p[nonzero]
                    self._peer_ratios[site] = float(np.median(ratios[~np.isnan(ratios)]))
                else:
                    self._peer_ratios[site] = 0.5
            else:
                self._peer_ratios[site] = 0.5

    def _build_multi_site_correlations(self, df: pd.DataFrame):
        for s1 in self.site_cols:
            if s1 not in df.columns:
                continue
            self._multi_site_correlations[s1] = {}
            for s2 in self.site_cols:
                if s2 == s1 or s2 not in df.columns:
                    continue
                mask = df[s1].notna() & df[s2].notna()
                if mask.sum() > 10:
                    corr = df.loc[mask, [s1, s2]].corr().iloc[0, 1]
                    self._multi_site_correlations[s1][s2] = float(corr) if not np.isnan(corr) else 0.0

    def _is_weekend_gap(self, df, gap_start, gap_end):
        return any(d in ['Saturday', 'Sunday']
                   for d in df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique())

    def _is_pure_weekend_gap(self, df, gap_start, gap_end):
        days = set(df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique())
        return days.isdisjoint({'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'})

    def _get_weekend_day_type(self, df, gap_start, gap_end):
        days = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()
        has_sat, has_sun = 'Saturday' in days, 'Sunday' in days
        if has_sat and has_sun:
            return 'mixed'
        return 'saturday' if has_sat else 'sunday' if has_sun else 'mixed'

    def _find_gap_groups(self, mask: pd.Series) -> List[Tuple[int, int]]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        gaps, start = [], indices[0]
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > 1:
                gaps.append((start, indices[i - 1] + 1))
                start = indices[i]
        gaps.append((start, indices[-1] + 1))
        return gaps

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy accessors
    # ─────────────────────────────────────────────────────────────────────────

    def get_low_confidence_report(self) -> pd.DataFrame:
        return pd.DataFrame(self._low_confidence_flags) if self._low_confidence_flags else pd.DataFrame()

    def get_strategy_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.strategy_log) if self.strategy_log else pd.DataFrame()