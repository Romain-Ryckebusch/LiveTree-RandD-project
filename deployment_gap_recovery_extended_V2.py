"""

Extended Deployment Gap Recovery Algorithm with Multi-Week Templates & Chunked Recovery

Extends the enhanced version with long-gap handling capabilities for 7-14 day gaps



NEW FEATURES:

  ✓ Occupancy data integration (weekend/holiday detection)

  ✓ Multi-week template matching (14-day lookback for weekly patterns)

  ✓ Chunked gap recovery (split 7+ day gaps into 4-day chunks)

  ✓ External features (calendar, holidays, occupancy events)

  ✓ Uncertainty bounds (confidence-based auto-flagging)

"""



import hashlib

from autoencoder_imputer import ProfileImputer

import logging

import os

import pickle

from datetime import date, timedelta

from typing import Dict, List, Optional, Tuple



import numpy as np

import pandas as pd

from scipy.signal import savgol_filter

from scipy.signal import lfilter

from scipy.stats import gaussian_kde

from sklearn.neighbors import NearestNeighbors



try:

    from sklearn.ensemble import IsolationForest

except ImportError:

    IsolationForest = None



log = logging.getLogger(__name__)



SITE_COLS = ['Ptot_HA', 'Ptot_HEI', 'Ptot_HEI_13RT', 'Ptot_HEI_5RNS', 'Ptot_RIZOMM', 'Ptot_Ilot']



# 10-minute interval data: 6 intervals/hour × 24 hours = 144 steps per day

STEPS_PER_DAY = 144



# Meter Hierarchy for Peer Correlation

METER_HIERARCHY = {

    'Ptot_HEI': {

        'parent': None,

        'children': ['Ptot_HEI_13RT', 'Ptot_HEI_5RNS'],

        'tier': 'main',

    },

    'Ptot_HEI_13RT': {

        'parent': 'Ptot_HEI',

        'children': [],

        'tier': 'sub',

    },

    'Ptot_HEI_5RNS': {

        'parent': 'Ptot_HEI',

        'children': [],

        'tier': 'sub',

    },

    'Ptot_HA': {

        'parent': None,

        'children': [],

        'tier': 'entry',

    },

    'Ptot_RIZOMM': {

        'parent': None,

        'children': [],

        'tier': 'entry',

    },

}





def _load_all_holidays():

    """Load holidays, close days, and special days from 2021-2026 files."""

    holidays: set = set()

    close_days: set = set()

    special_days: set = set()

    years = [2021, 2022, 2023, 2024, 2025, 2026]



    for year in years:

        file_map = {'Holiday': holidays, 'Close': close_days, 'Special': special_days}

        for ftype, target_set in file_map.items():

            fname = f'data/Consumption_{year}_{ftype}.csv'

            try:

                df = pd.read_csv(fname, header=0, names=['date'], parse_dates=['date'])

                target_set.update(df['date'].dt.date.unique())

            except FileNotFoundError:

                pass



    fallback_holidays = [

        (1, 1), (4, 5), (4, 6), (5, 1), (5, 8), (5, 14), (5, 25),

        (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)

    ]

    for m, d in fallback_holidays:

        holidays.add(date(2026, m, d))



    return holidays, close_days, special_days





FRANCE_HOLIDAYS_2026, FRANCE_CLOSE_DAYS_2026, FRANCE_SPECIAL_DAYS_2026 = _load_all_holidays()





class ExtendedDeploymentAlgorithm:

    """

    Extended deployment algorithm with multi-week templates and chunked recovery for long gaps.

    Inherits from enhanced version but adds occupancy-aware, multi-week, and uncertainty features.

    """



    def __init__(

        self,

        site_cols=None,

        use_mice: bool = True,

        use_knn: bool = True,

        use_kalman: bool = True,

        use_deep_learning: bool = False,  # Autoencoder-based imputation

        # Extended features

        use_multi_week_templates: bool = True,

        use_chunked_recovery: bool = True,

        gap_chunk_size: int = 96,  # ~6.7 hours at 10-min intervals (approx 1 day worth)

        occupancy_data: Optional[pd.DataFrame] = None,

        calendar_data: Optional[pd.DataFrame] = None,

        # Enhanced features (v2)

        template_lookback_days: int = 28,  # Extended from 14 to 28 days

        use_smart_chunking: bool = True,  # Smart chunking across day boundaries

        adaptive_template_bias: float = 0.7,  # Recency weight (0.7 = 70% recent, 30% historical)

        # Audit and timezone

        audit_log_dir: str = "audit_logs",

        timezone: str = "Europe/Paris",

    ):

        self.site_cols = site_cols or SITE_COLS

        self.use_mice = use_mice

        self.use_knn = use_knn

        self.use_kalman = use_kalman

        self.use_deep_learning = use_deep_learning



        # Extended features

        self.use_multi_week_templates = use_multi_week_templates

        self.use_chunked_recovery = use_chunked_recovery

        self.gap_chunk_size = gap_chunk_size

        self.occupancy_data = occupancy_data

        self.calendar_data = calendar_data

        # Enhanced features (v2)

        self.template_lookback_days = template_lookback_days

        self.use_smart_chunking = use_smart_chunking

        self.adaptive_template_bias = adaptive_template_bias



        self.strategy_log: List[Dict] = []

        self.weather_df = None

        # Deep Learning: Autoencoder-based imputation
        self._profile_imputer: Optional[ProfileImputer] = None
        self._profile_imputer_fitted: bool = False

        self._peer_ratios: Dict[str, float] = {}

        self.templates = {}



        # ML Method State Variables

        self._kalman_states: Dict[str, np.ndarray] = {}

        self._kalman_covariance: Dict[str, float] = {}

        self._knn_models: Dict[str, NearestNeighbors] = {}

        self._fitted_knn: bool = False



        # Operational Enhancement: Confidence Scoring

        self._reconstruction_confidence: Dict[str, float] = {}

        self._low_confidence_flags: List[Dict] = []

        self._seasonal_templates: Dict[str, Dict] = {}

        self._multi_site_correlations: Dict[str, Dict[str, float]] = {}



        # EXTENDED: Multi-week templates and uncertainty bounds

        self._weekly_templates: Dict[str, Dict[str, Dict]] = {}  # site -> dayofweek -> hour -> values

        self._uncertainty_bounds: Dict[str, Tuple[float, float]] = {}  # (lower, upper) bounds

        self._occupancy_patterns: Dict[str, Dict] = {}  # Occupancy by day type

        # Enhanced features (v2)

        self._day_variance: Dict[str, float] = {}  # Variance by day-of-week

        self._external_event_flags: Dict[str, List[str]] = {}  # External flags per datetime

        self._weather_variance: Dict[str, float] = {}  # Weather impact per day



        # Audit log and run meta

        self._audit_log: Dict = {}

        self._run_meta: Dict = {}

        self._audit_log_dir = audit_log_dir

        self._timezone = timezone

        if not os.path.exists(self._audit_log_dir):

            os.makedirs(self._audit_log_dir)



    # ─────────────────────────────────────────────────────────────────────────

    # Persistence

    # ─────────────────────────────────────────────────────────────────────────



    def save_templates(self, filepath: str):

        """Save templates and statistics to a file for later reuse."""

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

        log.info(f"[PERSIST] Saved templates/statistics to {filepath}")



    def load_templates(self, filepath: str):

        """Load templates and statistics from a file."""

        with open(filepath, 'rb') as f:

            data = pickle.load(f)

        self._weekly_templates = data.get('weekly_templates', {})

        self._day_variance = data.get('day_variance', {})

        self._seasonal_templates = data.get('seasonal_templates', {})

        self._peer_ratios = data.get('peer_ratios', {})

        self._multi_site_correlations = data.get('multi_site_correlations', {})

        self._uncertainty_bounds = data.get('uncertainty_bounds', {})

        log.info(f"[PERSIST] Loaded templates/statistics from {filepath}")



    # ─────────────────────────────────────────────────────────────────────────

    # Timezone & DST Handling

    # ─────────────────────────────────────────────────────────────────────────



    def _localise_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:

        """Localise naive timestamps to Europe/Paris timezone."""

        try:

            if df['Timestamp'].dt.tz is None:

                df['Timestamp'] = df['Timestamp'].dt.tz_localize(self._timezone, ambiguous='raise', nonexistent='shift_forward')

                log.info(f"[TZ] Localised naive timestamps to {self._timezone}")

            elif df['Timestamp'].dt.tz.zone != self._timezone:

                df['Timestamp'] = df['Timestamp'].dt.tz_convert(self._timezone)

                log.info(f"[TZ] Converted timestamps to {self._timezone}")

        except Exception as e:

            log.warning(f"[TZ] Localisation failed: {e}. Assuming naive UTC/local.")

        return df



    def _detect_dst_events(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:

        """Detect DST transitions (spring forward & fall back)."""

        dst_events = {'spring_forward': [], 'fall_back': []}

        try:

            if df['Timestamp'].dt.tz is None:

                return dst_events

            ts_diffs = df['Timestamp'].diff()

            # Spring forward: 1 hour gap (6 missing 10-min slots)

            spring_gaps = df[ts_diffs == pd.Timedelta('70min')]

            for idx, row in spring_gaps.iterrows():

                dst_events['spring_forward'].append({

                    'timestamp': row['Timestamp'],

                    'index': idx,

                    'description': 'Clock advanced 1 hour (DST spring forward)'

                })

            # Fall back: duplicate hour (second occurrence; we keep first and flag second)

            fall_dups = df[ts_diffs == pd.Timedelta('0min')]

            for idx, row in fall_dups.iterrows():

                dst_events['fall_back'].append({

                    'timestamp': row['Timestamp'],

                    'index': idx,

                    'description': 'Duplicate hour during DST fall back (removed)'

                })

            if dst_events['spring_forward'] or dst_events['fall_back']:

                log.info(f"[DST] Detected {len(dst_events['spring_forward'])} spring-forward and {len(dst_events['fall_back'])} fall-back events")

        except Exception as e:

            log.warning(f"[DST] Detection failed: {e}")

        return dst_events



    # ─────────────────────────────────────────────────────────────────────────

    # Zero-Fill Detection

    # ─────────────────────────────────────────────────────────────────────────



    def _detect_implausible_values(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:

        """

        Detect implausible values (stuck sensors, outliers, physically impossible readings).

        Replaces flagged values with NaN so they're handled by the recovery pipeline.

        """

        implausible_flags = {}

        try:

            for site in self.site_cols:

                if site not in df.columns:

                    continue

                implausible_flags[site] = []

                valid_mask = df[site].notna()

                valid_data = df.loc[valid_mask, site]

                

                if len(valid_data) < 10:

                    continue

                

                # STEP 1: Detect stuck sensors (constant values over sliding windows)

                window_size = 144  # 1 day at 10-min intervals

                for idx in range(len(df) - window_size):

                    window = df.loc[idx:idx + window_size, site].values

                    non_nan_window = window[~np.isnan(window)]

                    if len(non_nan_window) > 0.9 * window_size:  # At least 90% data available

                        if np.std(non_nan_window) < 1e-6:  # Effectively constant

                            stuck_val = non_nan_window[0]

                            for i in range(idx, min(idx + window_size, len(df))):

                                if not np.isnan(df.loc[i, site]):

                                    implausible_flags[site].append({

                                        'index': i,

                                        'timestamp': df.loc[i, 'Timestamp'],

                                        'value': df.loc[i, site],

                                        'detection_method': 'stuck_sensor',

                                        'reason': f'Constant value {stuck_val} over {window_size} points (1 day)',

                                    })

                                    df.loc[i, site] = np.nan  # Replace with NaN

                

                # STEP 2: IsolationForest for general outliers (including zero-fills and anomalies)

                if IsolationForest is not None:

                    try:

                        X = valid_data.values.reshape(-1, 1)

                        clf = IsolationForest(contamination=0.05, random_state=0, n_estimators=100)

                        predictions = clf.fit_predict(X)

                        anomaly_indices = np.where(predictions == -1)[0]

                        

                        for idx in anomaly_indices:

                            orig_idx = valid_data.index[idx]

                            val = valid_data.iloc[idx]

                            # Skip if already flagged as stuck sensor

                            if any(flag['index'] == orig_idx for flag in implausible_flags[site]):

                                continue

                            

                            reason_parts = []

                            # Check for zero/near-zero fills

                            median_val = valid_data.median()

                            if val < median_val * 0.01:

                                reason_parts.append(f'suspiciously low ({val:.1f} < 1% median)')

                            # Check for extreme high values

                            if val > valid_data.quantile(0.95) * 2:

                                reason_parts.append(f'extreme high ({val:.1f} >> 95th percentile)')

                            

                            reason = 'Outlier detected: ' + '; '.join(reason_parts) if reason_parts else 'General anomaly (IsolationForest)'

                            

                            implausible_flags[site].append({

                                'index': orig_idx,

                                'timestamp': df.loc[orig_idx, 'Timestamp'],

                                'value': val,

                                'detection_method': 'IsolationForest',

                                'reason': reason,

                            })

                            df.loc[orig_idx, site] = np.nan  # Replace with NaN for recovery

                    except Exception as e:

                        log.debug(f"[OUTLIER] IsolationForest failed for {site}: {e}")

                else:

                    # Fallback: Z-score method

                    mean_val = valid_data.mean()

                    std_val = valid_data.std()

                    if std_val > 0:

                        z_scores = np.abs((valid_data - mean_val) / std_val)

                        outlier_indices = np.where(z_scores > 3.0)[0]

                        

                        for idx in outlier_indices:

                            orig_idx = valid_data.index[idx]

                            val = valid_data.iloc[idx]

                            if any(flag['index'] == orig_idx for flag in implausible_flags[site]):

                                continue

                            

                            implausible_flags[site].append({

                                'index': orig_idx,

                                'timestamp': df.loc[orig_idx, 'Timestamp'],

                                'value': val,

                                'detection_method': 'z-score',

                                'reason': f'Statistical outlier (z-score = {z_scores.iloc[idx]:.1f} > 3.0)',

                            })

                            df.loc[orig_idx, site] = np.nan

                

                if implausible_flags[site]:

                    log.info(f"[OUTLIER] Detected and flagged {len(implausible_flags[site])} implausible values for {site}")

        except Exception as e:

            log.error(f"[ERROR] Implausible value detection failed: {e}")

        

        return implausible_flags



    def _detect_zero_fills(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:

        """Detect suspected zero-fills using IsolationForest or Z-score fallback (post-imputation check)."""

        zero_fill_flags = {}

        try:

            for site in self.site_cols:

                if site not in df.columns:

                    continue

                zero_fill_flags[site] = []

                valid_data = df[site].dropna()

                if len(valid_data) < 10:

                    continue

                zero_threshold = valid_data.median() * 0.01  # < 1% of median

                if IsolationForest is not None:

                    try:

                        # Fit IsolationForest on post-imputation data

                        X = valid_data.values.reshape(-1, 1)

                        clf = IsolationForest(contamination=0.05, random_state=0)

                        predictions = clf.fit_predict(X)

                        anomaly_indices = np.where(predictions == -1)[0]

                        # Check if anomalies are near-zero

                        for idx in anomaly_indices:

                            val = valid_data.iloc[idx]

                            if val < zero_threshold:

                                orig_idx = valid_data.index[idx]

                                zero_fill_flags[site].append({

                                    'index': orig_idx,

                                    'timestamp': df.loc[orig_idx, 'Timestamp'],

                                    'value': val,

                                    'method': 'IsolationForest',

                                    'reason': f'Anomalous low value (< {zero_threshold:.1f})',

                                })

                    except Exception as e:

                        log.debug(f"[ZERO-FILL] IsolationForest failed for {site}, using z-score: {e}")

                        # Fallback to z-score

                        z_scores = np.abs((valid_data - valid_data.mean()) / valid_data.std())

                        outlier_indices = np.where(z_scores > 3.0)[0]

                        for idx in outlier_indices:

                            val = valid_data.iloc[idx]

                            if val < zero_threshold:

                                orig_idx = valid_data.index[idx]

                                zero_fill_flags[site].append({

                                    'index': orig_idx,

                                    'timestamp': df.loc[orig_idx, 'Timestamp'],

                                    'value': val,

                                    'method': 'z-score',

                                    'reason': f'Outlier (z > 3.0) and low value (< {zero_threshold:.1f})',

                                })

                else:

                    # Fallback: z-score only

                    z_scores = np.abs((valid_data - valid_data.mean()) / valid_data.std())

                    outlier_indices = np.where(z_scores > 3.0)[0]

                    for idx in outlier_indices:

                        val = valid_data.iloc[idx]

                        if val < zero_threshold:

                            orig_idx = valid_data.index[idx]

                            zero_fill_flags[site].append({

                                'index': orig_idx,

                                'timestamp': df.loc[orig_idx, 'Timestamp'],

                                'value': val,

                                'method': 'z-score',

                                'reason': f'Outlier (z > 3.0) and low value (< {zero_threshold:.1f})',

                            })

                if zero_fill_flags[site]:

                    log.info(f"[ZERO-FILL] Detected {len(zero_fill_flags[site])} suspected zero-fills for {site}")

        except Exception as e:

            log.error(f"[ERROR] Zero-fill detection failed: {e}")

        return zero_fill_flags



    # ─────────────────────────────────────────────────────────────────────────

    # Audit Log Management

    # ─────────────────────────────────────────────────────────────────────────



    def _save_audit_log(self, num_gaps_total: int, sites_affected: List[str], 

                       zero_fill_corrections: int = 0, dst_events: Optional[Dict] = None):

        """Save audit log to JSON file with timestamp."""

        try:

            import json

            from datetime import datetime as dt

            timestamp_str = dt.now().strftime('%Y%m%d_%H%M%S')

            filename = f"{self._audit_log_dir}/audit_log_{timestamp_str}.json"

            self._run_meta['timestamp'] = timestamp_str

            self._run_meta['total_gaps_found'] = num_gaps_total

            self._run_meta['sites_affected'] = list(set(sites_affected))

            self._run_meta['zero_fill_corrections'] = zero_fill_corrections

            implausible_count = sum(len(flags) for flags in self._audit_log.get('implausible_detections', {}).values())

            self._run_meta['implausible_values_flagged'] = implausible_count

            self._run_meta['dst_events'] = dst_events or {}

            self._run_meta['total_strategies_used'] = len(set(self.strategy_log and [g.get('strategy') for g in self.strategy_log]))

            if self.strategy_log:

                self._run_meta['mean_confidence'] = float(np.mean([g.get('confidence', 0) for g in self.strategy_log]))

            audit_output = {

                'run_meta': self._run_meta,

                'gaps': self._audit_log.get('gaps', []),

                'detection_summary': self._audit_log.get('detection_summary', {}),

                'implausible_detections': self._audit_log.get('implausible_detections', {}),

                'zero_fill_detections': self._audit_log.get('zero_fill_detections', {}),

            }

            with open(filename, 'w') as f:

                json.dump(audit_output, f, indent=2, default=str)

            log.info(f"[AUDIT] Saved audit log to {filename}")

        except Exception as e:

            log.error(f"[ERROR] Failed to save audit log: {e}")



    def get_audit_log(self) -> Dict:

        """Return the full audit log dictionary."""

        return self._audit_log



    def get_run_summary(self) -> Dict:

        """Return run-level statistics."""

        return self._run_meta



    def impute(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        """

        Main entry point: impute gaps with extended features for 7+ day gaps.

        Includes timezone localization, DST handling, implausible value detection,

        zero-fill detection, and full audit logging.

        """

        df = df.copy().sort_values('Timestamp').reset_index(drop=True)



        # STEP 1: Timezone localization

        df = self._localise_timestamps(df)



        if weather_df is not None:

            self.weather_df = weather_df

            df = self._merge_weather(df)



        # Reindex for complete time range

        start = df['Timestamp'].min().floor('10min')

        end = df['Timestamp'].max().ceil('10min')

        df = (

            df.set_index('Timestamp')

              .reindex(pd.date_range(start, end, freq='10min'))

              .reset_index()

        )

        df.columns = ['Timestamp'] + list(df.columns[1:])



        # STEP 2: Detect DST events

        dst_events = self._detect_dst_events(df)



        # STEP 3: Detect implausible values (stuck sensors, outliers, etc.)

        # These are replaced with NaN to be handled by the normal imputation pipeline

        implausible_detections = self._detect_implausible_values(df)



        # Add helper columns

        self._add_datetime_features(df)

        self._fill_airtemp_forward(df)

        self._classify_thermal_regimes(df)



        # EXTENDED: Add occupancy and external features

        self._add_occupancy_features(df)

        self._add_external_features(df)



        # Build peer ratios before processing

        self._build_peer_ratios(df)



        # Build traditional templates

        self._build_day_specific_templates(df)

        self._build_seasonal_templates(df)



        # EXTENDED: Build multi-week templates and uncertainty bounds

        if self.use_multi_week_templates:

            self._build_weekly_templates(df)

            self._build_uncertainty_bounds(df)



        # Build multi-site correlations

        self._build_multi_site_correlations(df)



        # DEEP LEARNING: Initialize and fit autoencoder-based imputer

        if self.use_deep_learning:

            try:

                self._profile_imputer = ProfileImputer(latent_dim=8, use_nonlinear=False)

                self._profile_imputer.fit(df)

                self._profile_imputer_fitted = True

                log.info("[DL] Autoencoder (ProfileImputer) fitted successfully")

            except Exception as e:

                log.warning(f"[DL] Autoencoder fitting failed: {e}. Falling back to other methods.")

                self._profile_imputer_fitted = False

        # Initialize audit log for this run

        self._audit_log = {'gaps': [], 'detection_summary': {}, 'zero_fill_detections': {}, 'implausible_detections': implausible_detections}

        total_gaps_processed = 0

        sites_with_gaps = []



        # TIER 1: Intelligent Router for each site

        for site in self.site_cols:

            if site not in df.columns:

                continue



            gap_mask = df[site].isna()

            if not gap_mask.any():

                continue



            sites_with_gaps.append(site)

            for gap_start, gap_end, metadata in self._find_gap_groups(gap_mask, df):

                total_gaps_processed += 1

                # EXTENDED: Check if chunked recovery should be used

                gap_size = gap_end - gap_start

                if self.use_chunked_recovery and gap_size > self.gap_chunk_size:

                    self._fill_chunked_gap(df, site, gap_start, gap_end)

                else:

                    self._intelligent_router(df, site, gap_start, gap_end, metadata)



        # TIER 4: NaN Guard - final pass to eliminate any remaining NaN

        self._nan_guard_final_pass(df)



        # TIER 5: Signal Smoothing at junctions

        for site in self.site_cols:

            if site in df.columns:

                self._smooth_junctions(df, site)



        # STEP 4: Detect zero-fills and re-impute

        zero_fill_detections = self._detect_zero_fills(df)

        self._audit_log['zero_fill_detections'] = zero_fill_detections

        zero_fill_count = sum(len(flags) for flags in zero_fill_detections.values())



        for site, flags in zero_fill_detections.items():

            for flag in flags:

                idx = flag['index']

                log.info(f"[ZERO-FILL-CORRECT] Re-imputing {site} at {flag['timestamp']}")

                df.loc[idx, site] = np.nan  # Reset to NaN

                # Re-impute this point

                gap_mask = df[site].isna()

                for gap_start, gap_end, metadata in self._find_gap_groups(gap_mask, df):

                    if gap_start <= idx < gap_end:

                        self._intelligent_router(df, site, gap_start, gap_end, metadata)

                        break

                flag['zero_fill_corrected'] = True



        implausible_count = sum(len(flags) for flags in implausible_detections.values())



        # STEP 5: Save audit log

        self._save_audit_log(total_gaps_processed, sites_with_gaps, zero_fill_count, dst_events)



        # Return output columns (convert back to naive if needed for compatibility)

        out_cols = ['Timestamp'] + self.site_cols + (['AirTemp'] if 'AirTemp' in df.columns else [])

        result_df = df[[c for c in out_cols if c in df.columns]].copy()

        

        # Convert timestamps back to naive if they have timezone info

        if result_df['Timestamp'].dt.tz is not None:

            result_df['Timestamp'] = result_df['Timestamp'].dt.tz_localize(None)

        

        log.info(f"[IMPUTE-SUMMARY] Total gaps: {total_gaps_processed}, Implausible values flagged: {implausible_count}, Zero-fills corrected: {zero_fill_count}")

        return result_df



    # ─────────────────────────────────────────────────────────────────────────

    # EXTENDED: Chunked Gap Recovery for Long Gaps (7+ days)

    # ─────────────────────────────────────────────────────────────────────────



    def _fill_chunked_gap(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """

        Split long gaps (7+ days) into intelligent chunks.

        - If use_smart_chunking: detect high-variance days and keep them together

        - Otherwise: use fixed chunk sizes

        """

        gap_size = gap_end - gap_start



        if self.use_smart_chunking:

            chunks = self._get_smart_chunks(df, gap_start, gap_end)

            log.info(f"[CHUNKED-SMART] Splitting {gap_size} points ({site}) into {len(chunks)} smart chunks")

        else:

            num_chunks = int(np.ceil(gap_size / self.gap_chunk_size))

            chunks = [(gap_start + (i * self.gap_chunk_size),

                      min(gap_start + ((i + 1) * self.gap_chunk_size), gap_end))

                     for i in range(num_chunks)]

            log.info(f"[CHUNKED] Splitting {gap_size} points ({site}) into {num_chunks} chunks")



        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):

            chunk_size = chunk_end - chunk_start

            chunk_metadata = {

                'timestamps': df.loc[chunk_start:chunk_end - 1, 'Timestamp'].tolist(),

                'row_count': chunk_size,

                'date_range': f"{df.loc[chunk_start, 'Timestamp']} to {df.loc[chunk_end - 1, 'Timestamp']}",

                'chunk_idx': chunk_idx,

                'chunk_total': len(chunks),

            }



            # Use multi-week template if available for this chunk

            if self.use_multi_week_templates and self._weekly_templates:

                self._fill_with_multi_week_template(df, site, chunk_start, chunk_end)

            else:

                # Fall back to intelligent router for this chunk

                self._intelligent_router(df, site, chunk_start, chunk_end, chunk_metadata)



            log.debug(f"[CHUNKED] Filled chunk {chunk_idx+1}/{len(chunks)}: "

                     f"indices {chunk_start}-{chunk_end}, size {chunk_size}")



    # ─────────────────────────────────────────────────────────────────────────

    # ENHANCED-V2: Smart Chunking (Keep High-Variance Days Together)

    # ─────────────────────────────────────────────────────────────────────────



    def _get_smart_chunks(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> List[Tuple[int, int]]:

        """

        Intelligently chunk a gap by detecting high-variance days and keeping them together.

        - Low variance days (stable patterns): group in ~4-day chunks

        - High variance days (changeable patterns): keep separate

        - Avoids splitting across Mon-Wed boundaries where patterns change

        Returns list of (chunk_start, chunk_end) tuples.



        FIX: Variance lookup is now keyed by day name (string), not a numeric index

             derived from position within the gap. Daily variance list was previously

             indexed with `i // 144` where 144 is STEPS_PER_DAY, but the list length

             equals the number of unique days in the gap — not len(gap_dates) // 144 —

             so index-based lookup caused out-of-bounds errors. Now we look up variance

             directly by day name from self._day_variance.



        FIX: Removed the unused `current_date` variable.

        """

        chunks = []

        current_chunk_start = gap_start

        current_chunk_size = 0

        max_chunk_size = self.gap_chunk_size * 1.5  # Allow up to 1.5x normal chunk size for high-variance



        gap_size = gap_end - gap_start

        gap_dates = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values

        gap_hours = df.loc[gap_start:gap_end - 1, 'Hour'].values



        # Build a set of "high variance" day names using per-site day_variance dicts.

        # Use the first available site as the reference signal.

        reference_site = next((s for s in self.site_cols if s in self._day_variance), None)

        site_day_variance = self._day_variance.get(reference_site, {}) if reference_site else {}



        if site_day_variance:

            all_variances = list(site_day_variance.values())

            v_threshold = float(np.median(all_variances))

            high_variance_days = {

                day_name for day_name, var in site_day_variance.items()

                if var > v_threshold * 1.3

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



            # Decide if we should end current chunk

            should_chunk_break = False

            if current_chunk_size >= max_chunk_size:

                should_chunk_break = True  # Too large, must break

            elif is_high_variance and current_chunk_size > self.gap_chunk_size * 0.8:

                should_chunk_break = True  # High-variance day taking too long, extract it

            elif is_new_day and current_chunk_size >= self.gap_chunk_size:

                should_chunk_break = True  # Good place to break (new day boundary)



            if should_chunk_break and idx < gap_end:

                chunks.append((current_chunk_start, idx))

                current_chunk_start = idx

                current_chunk_size = 0



        # Add final chunk

        if current_chunk_start < gap_end:

            chunks.append((current_chunk_start, gap_end))



        log.debug(f"[SMART-CHUNK] Identified {len(chunks)} chunks, high-var threshold: {v_threshold:.1f}")

        return chunks



    # ─────────────────────────────────────────────────────────────────────────

    # EXTENDED: Multi-Week Templates (28-day lookback with adaptive bias)

    # ─────────────────────────────────────────────────────────────────────────



    def _build_weekly_templates(self, df: pd.DataFrame):

        """

        Build weekly templates with 28-day lookback and adaptive recency bias.

        - Uses {template_lookback_days} days for richer patterns

        - Weights recent data 70% (configurable) vs historical 30% to adapt to trends

        - Detects high-variance days for smart chunking



        FIX: recent_cutoff was previously computed as max_date minus 30% of the lookback

             window, making "recent" only the last 30% of days and "historical" the prior

             70%.  The adaptive_template_bias (default 0.7) is supposed to give more weight

             to recent data, so "recent" must be the *larger* portion.  The cutoff is now

             set so that recent data spans the last (bias × lookback) days and historical

             data spans the earlier (1 - bias) × lookback days.



        FIX: `all_values[:100][:]` was a redundant double-slice creating an unnecessary

             copy.  Changed to `all_values[:100].copy()` to make the intent explicit.

        """

        try:

            max_date = df['Timestamp'].max()

            min_date = max_date - pd.Timedelta(days=self.template_lookback_days)



            # "Recent" = last (bias) fraction of the lookback window (e.g. last 70% = ~20 days)

            # "Historical" = earlier (1-bias) fraction (e.g. first 30% = ~8 days)

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



                # For each day name (Monday - Sunday)

                for day_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:

                    day_templates = {}

                    day_values_all = []



                    # For each hour (0-23)

                    for hour in range(24):

                        # Get recent and historical values separately

                        recent = df.loc[

                            recent_mask & (df['DayOfWeek'] == day_name) &

                            (df['Hour'] == hour) & (df[site].notna()), site

                        ].values



                        historical = df.loc[

                            historical_mask & (df['DayOfWeek'] == day_name) &

                            (df['Hour'] == hour) & (df[site].notna()), site

                        ].values



                        # Adaptive blending: bias% recent, (1-bias)% historical

                        if len(recent) > 0 and len(historical) > 0:

                            recent_median = np.median(recent)

                            historical_median = np.median(historical)

                            blended_median = (self.adaptive_template_bias * recent_median +

                                           (1 - self.adaptive_template_bias) * historical_median)

                            all_values = np.concatenate([recent, historical])

                        elif len(recent) > 0:

                            blended_median = np.median(recent)

                            all_values = recent

                        elif len(historical) > 0:

                            blended_median = np.median(historical)

                            all_values = historical

                        else:

                            blended_median = None

                            all_values = np.array([])



                        if len(all_values) > 0:

                            day_templates[hour] = {

                                'median': blended_median,

                                'mean': np.mean(all_values),

                                'std': np.std(all_values),

                                'q25': np.percentile(all_values, 25),

                                'q75': np.percentile(all_values, 75),

                                'recent_count': len(recent),

                                'historical_count': len(historical),

                                'all_values': all_values[:100].copy(),  # Keep subset for KDE

                            }

                            day_values_all.extend(all_values)

                        else:

                            day_templates[hour] = None



                    self._weekly_templates[site][day_name] = day_templates

                    # Store daily variance for smart chunking

                    if day_values_all:

                        self._day_variance[site][day_name] = float(np.std(day_values_all))



            log.info(f"[EXTENDED-V2] Built adaptive weekly templates ({self.template_lookback_days}-day lookback, "

                    f"{int(self.adaptive_template_bias*100)}% recency bias)")

        except Exception as e:

            log.error(f"[ERROR] Failed to build weekly templates: {e}")



    def _fill_with_multi_week_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """

        Fill gap using multi-week lookback templates by day-of-week and hour.

        More stable than KNN for extended gaps.

        """

        try:

            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values

            day_names = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values



            filled_vals = []

            confidences = []



            for h, day_name in zip(hours, day_names):

                # Lookup template for this day+hour

                template = self._weekly_templates.get(site, {}).get(day_name, {}).get(h)



                if template is not None:

                    filled_vals.append(template['median'])

                    # Confidence based on how consistent this (day, hour) is

                    if template['std'] > 0:

                        confidence = 1.0 / (1.0 + template['std'] / (template['median'] + 1e-6))

                    else:

                        confidence = 1.0

                    confidences.append(confidence)

                else:

                    # Fall back to hourly median across all days

                    mask = (df['Hour'] == h) & (df[site].notna())

                    if mask.any():

                        filled_vals.append(df.loc[mask, site].median())

                        confidences.append(0.5)  # Lower confidence for fallback

                    else:

                        filled_vals.append(np.nan)

                        confidences.append(0.0)



            filled_vals = np.array(filled_vals)

            filled_vals = self._validate_and_clip(filled_vals, site, df)

            df.loc[gap_start:gap_end - 1, site] = filled_vals



            # Store confidence info

            avg_confidence = np.nanmean(confidences)

            self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = avg_confidence



            # Flag if low confidence

            if avg_confidence < 0.5:

                self._low_confidence_flags.append({

                    'site': site,

                    'gap_start': gap_start,

                    'gap_end': gap_end,

                    'gap_size': gap_end - gap_start,

                    'reason': 'multi_week_template_low_confidence',

                    'confidence': avg_confidence,

                    'occupancy': df.loc[gap_start, 'IsOccupied'] if 'IsOccupied' in df.columns else None,

                    'is_holiday': df.loc[gap_start, 'IsHoliday'] if 'IsHoliday' in df.columns else None

                })



            self.strategy_log.append({

                'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                'gap_size': gap_end - gap_start, 'strategy': 'MULTI_WEEK_TEMPLATE',

                'confidence': avg_confidence

            })



        except Exception as e:

            log.error(f"[ERROR] Multi-week template fill failed for {site}: {e}")

            self._fill_safe_median_template(df, site, gap_start, gap_end)



    # ─────────────────────────────────────────────────────────────────────────

    # EXTENDED: Occupancy-Aware Features

    # ─────────────────────────────────────────────────────────────────────────



    def _add_occupancy_features(self, df: pd.DataFrame):

        """

        Add occupancy-related features:

        - IsOccupied: Estimated occupancy from time patterns and external data

        - OccupancyType: 'work_hours', 'evening', 'weekend', 'holiday'



        FIX: The original code applied the evening mask conditionally on

             `OccupancyType == 'work_hours'`, but that check was evaluated against

             the column's state at that moment — after weekend/holiday rows had already

             been updated to 'weekend'/'holiday'.  The evening filter must therefore

             exclude those rows explicitly rather than relying on the residual

             'work_hours' label, which could still be stale due to pandas

             copy-on-write semantics.  We now use a single pass: first build a

             complete boolean array for each category, then assign in one shot so

             later assignments cannot silently overwrite earlier ones.

        """

        try:

            weekend_mask = df['DayOfWeek'].isin(['Saturday', 'Sunday'])



            # Holiday mask: external calendar takes priority, else built-in list

            if hasattr(self, 'calendar_data') and self.calendar_data is not None:

                holiday_dates = pd.to_datetime(self.calendar_data.get('holiday_list', [])).dt.date

                holiday_mask = df['Date'].isin(holiday_dates)

            else:

                holiday_mask = df['Date'].isin(FRANCE_HOLIDAYS_2026)



            evening_mask = (df['Hour'] >= 18) | (df['Hour'] < 6)



            # A row is "evening" only if it is a plain weekday that is not a holiday

            plain_weekday_mask = ~weekend_mask & ~holiday_mask

            evening_weekday_mask = evening_mask & plain_weekday_mask



            # Assign in priority order (last write wins, so most-specific goes last)

            df['OccupancyType'] = 'work_hours'

            df['IsOccupied'] = True



            df.loc[weekend_mask, 'OccupancyType'] = 'weekend'

            df.loc[weekend_mask, 'IsOccupied'] = False



            df.loc[holiday_mask, 'OccupancyType'] = 'holiday'

            df.loc[holiday_mask, 'IsOccupied'] = False



            # Evening on plain weekdays (must come after holiday so holidays are not re-tagged)

            df.loc[evening_weekday_mask, 'OccupancyType'] = 'evening'

            df.loc[evening_weekday_mask, 'IsOccupied'] = False



            log.info("[EXTENDED] Added occupancy features")



        except Exception as e:

            log.error(f"[ERROR] Failed to add occupancy features: {e}")

            df['IsOccupied'] = True

            df['OccupancyType'] = 'unknown'



    # ─────────────────────────────────────────────────────────────────────────

    # EXTENDED: External Features (Calendar, Events, Holidays)

    # ─────────────────────────────────────────────────────────────────────────



    def _add_external_features(self, df: pd.DataFrame):

        """

        Add external calendar & weather features:

        - IsHoliday: Boolean for holidays

        - Season: winter, spring, summer, fall

        - IsSpecialDay: Close days, special events

        - IsHolidayClose: Multi-day holiday period (holiday + day before/after)

        - IsEventDay: High-consumption spike indicator

        - WeatherSpike: Detected temperature anomaly

        """

        try:

            # Holiday detection

            df['IsHoliday'] = df['Date'].isin(FRANCE_HOLIDAYS_2026)

            df['IsSpecialDay'] = (

                df['Date'].isin(FRANCE_CLOSE_DAYS_2026) |

                df['Date'].isin(FRANCE_SPECIAL_DAYS_2026)

            )



            # ENHANCED-V2: Holiday close periods (day before + day of + day after)

            df['IsHolidayClose'] = False

            for holiday_date in FRANCE_HOLIDAYS_2026:

                period_mask = (

                    (df['Date'] >= holiday_date - timedelta(days=1)) &

                    (df['Date'] <= holiday_date + timedelta(days=1))

                )

                df.loc[period_mask, 'IsHolidayClose'] = True



            # ENHANCED-V2: Event day detection (high variance in historical patterns)

            df['IsEventDay'] = False

            for site in self.site_cols:

                if site in df.columns:

                    # Calculate rolling std to detect anomalies

                    daily_std = df.groupby('Date')[site].std()

                    high_std_threshold = daily_std.quantile(0.75)

                    high_std_dates = daily_std[daily_std > high_std_threshold].index

                    df.loc[df['Date'].isin(high_std_dates), 'IsEventDay'] = True



            # ENHANCED-V2: Weather spike detection

            df['WeatherSpike'] = False

            if 'AirTemp' in df.columns:

                daily_temp_change = df.groupby('Date')['AirTemp'].apply(

                    lambda x: abs(x.max() - x.min()) if len(x) > 0 else 0

                )

                temp_spike_threshold = daily_temp_change.quantile(0.80)

                spike_dates = daily_temp_change[daily_temp_change > temp_spike_threshold].index

                df.loc[df['Date'].isin(spike_dates), 'WeatherSpike'] = True



            # Season classification

            df['Season'] = df['Timestamp'].dt.month.map({

                12: 'winter', 1: 'winter', 2: 'winter',

                3: 'spring', 4: 'spring', 5: 'spring',

                6: 'summer', 7: 'summer', 8: 'summer',

                9: 'fall', 10: 'fall', 11: 'fall'

            })



            log.info("[EXTENDED-V2] Added rich external features (holiday close, event days, weather spikes)")



        except Exception as e:

            log.error(f"[ERROR] Failed to add external features: {e}")



    # ─────────────────────────────────────────────────────────────────────────

    # EXTENDED: Uncertainty Bounds (Confidence Intervals)

    # ─────────────────────────────────────────────────────────────────────────



    def _build_uncertainty_bounds(self, df: pd.DataFrame):

        """

        Build uncertainty bounds (confidence intervals) for each site based on

        historical variance patterns and gap characteristics.

        """

        try:

            for site in self.site_cols:

                if site not in df.columns:

                    continue



                valid_data = df[site].dropna()

                if len(valid_data) < 10:

                    self._uncertainty_bounds[site] = (0, valid_data.max() * 1.5)

                    continue



                # Calculate bounds based on historical distribution

                mean = valid_data.mean()

                std = valid_data.std()



                # 95% confidence interval

                lower_bound = max(0, mean - 2 * std)

                upper_bound = mean + 2 * std



                self._uncertainty_bounds[site] = (lower_bound, upper_bound)



                log.debug(f"[UNCERTAINTY] {site}: bounds ({lower_bound:.1f}, {upper_bound:.1f})")



            log.info("[EXTENDED] Built uncertainty bounds")



        except Exception as e:

            log.error(f"[ERROR] Failed to build uncertainty bounds: {e}")



    def _calculate_confidence_with_uncertainty(

        self, site: str, gap_size: int, day_type: str, strategy: str,

        occupancy_type: Optional[str] = None, is_holiday: bool = False

    ) -> float:

        """

        Calculate confidence score considering:

        - Gap size (longer = lower confidence)

        - Day type (weekend > holiday > weekday)

        - Recovery strategy (MICE > KNN > template > linear)

        - Occupancy type (work_hours > evening > weekend > holiday)

        """

        try:

            # Strategy quality factor

            strategy_confidence_map = {

                'AUTOENCODER': 0.92,

                'MULTI_WEEK_TEMPLATE': 0.90,

                'MICE': 0.85,

                'KNN_CONTEXT': 0.80,

                'KALMAN_FILTER': 0.75,

                'THERMAL_TEMPLATE': 0.70,

                'ENHANCED_TEMPLATE': 0.65,

                'WEEKEND_TEMPLATE': 0.60,

                'PEER_CORRELATION': 0.75,

                'LINEAR_SHORT': 0.50,

                'LINEAR_MICRO': 0.40,

                'SAFE_LINEAR_MEDIAN': 0.45,

                'SAFE_MEDIAN': 0.30,

            }

            strategy_factor = strategy_confidence_map.get(strategy, 0.5)



            # Gap size penalty (non-linear)

            if gap_size < 10:

                gap_factor = 1.0

            elif gap_size < 50:

                gap_factor = 0.9

            elif gap_size < STEPS_PER_DAY:  # 1 day

                gap_factor = 0.7

            elif gap_size < STEPS_PER_DAY * 7:  # 7 days

                gap_factor = 0.5

            else:

                gap_factor = 0.3  # Very large gaps: low confidence



            # Occupancy factor (occupancy affects predictability)

            if occupancy_type == 'work_hours':

                occupancy_factor = 1.0

            elif occupancy_type == 'evening':

                occupancy_factor = 0.8

            elif occupancy_type == 'weekend':

                occupancy_factor = 0.7

            elif occupancy_type == 'holiday':

                occupancy_factor = 0.5

            else:

                occupancy_factor = 0.6



            # Holiday/special day penalty

            holiday_factor = 0.7 if is_holiday else 1.0



            # Combine factors

            confidence = strategy_factor * gap_factor * occupancy_factor * holiday_factor

            confidence = float(np.clip(confidence, 0.0, 1.0))



            return confidence



        except Exception as e:

            log.debug(f"[DEBUG] Confidence calculation failed: {e}")

            return 0.5



    def _generate_uncertainty_bounds_for_gap(

        self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int

    ) -> Tuple[np.ndarray, np.ndarray]:

        """

        Generate upper and lower uncertainty bounds for a gap based on

        historical variability and gap characteristics.

        """

        try:

            lower_bound, upper_bound = self._uncertainty_bounds.get(site, (0.0, np.inf))



            gap_size = gap_end - gap_start



            # Widen bounds based on gap size

            if gap_size > STEPS_PER_DAY:  # > 1 day

                expansion = min(0.5, gap_size / 1000.0)  # Up to 50% wider

                lower_bound = lower_bound * (1.0 - expansion)

                upper_bound = upper_bound * (1.0 + expansion)



            # Check occupancy to refine bounds

            occupancy = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else 'unknown'

            if occupancy == 'holiday':

                upper_bound = upper_bound * 0.7  # Holidays typically have lower energy



            lower_array = np.full(gap_size, lower_bound)

            upper_array = np.full(gap_size, upper_bound)



            return lower_array, upper_array



        except Exception as e:

            log.debug(f"[DEBUG] Uncertainty bounds generation failed: {e}")

            filled_vals = df.loc[gap_start:gap_end - 1, site].values

            return filled_vals * 0.8, filled_vals * 1.2



    def _flag_low_confidence(

        self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int,

        confidence: float, strategy: str

    ):

        """

        Flag gaps with low confidence for manual review or additional enrichment.



        FIX: The original method tried to access `df` via `hasattr(self, 'df')` but

             `df` was never stored on `self` — it is a local parameter passed through

             the call chain.  The method signature now accepts `df` explicitly so

             occupancy and holiday information can be read directly and safely.

        """

        if confidence < 0.50:

            occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

            is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False



            self._low_confidence_flags.append({

                'site': site,

                'gap_start': gap_start,

                'gap_end': gap_end,

                'gap_size': gap_end - gap_start,

                'confidence': confidence,

                'strategy': strategy,

                'occupancy_type': occupancy_type,

                'is_holiday': is_holiday,

                'reason': 'low_confidence_score'

            })



    # ─────────────────────────────────────────────────────────────────────────

    # TIER 1: Intelligent Router

    # ─────────────────────────────────────────────────────────────────────────



    def _intelligent_router(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, 

                           metadata: Optional[Dict] = None):

        """

        Route gap based on size, meter type, and context.

        Logs routing trace and method selection to audit log.



        FIX: `_flag_low_confidence` now receives `df` as its first argument so it

             can read occupancy and holiday columns without relying on the removed

             `hasattr(self, 'df')` pattern.

        """

        if metadata is None:

            metadata = {}

        

        routing_trace = [

            f"[ROUTE] Gap detected: {site} from {gap_start} to {gap_end}",

            f"  Date range: {metadata.get('date_range', 'N/A')}",

            f"  Row count: {metadata.get('row_count', gap_end - gap_start)}",

        ]

        

        gap_size = gap_end - gap_start

        meter_tier = METER_HIERARCHY.get(site, {}).get('tier', 'unknown')

        is_weekend = self._is_weekend_gap(df, gap_start, gap_end)

        is_submeter = meter_tier == 'sub'

        is_entry = meter_tier == 'entry'

        

        routing_trace.append(f"  Meter tier: {meter_tier}, Is weekend: {is_weekend}, Gap size: {gap_size}")



        # ML methods for large gaps

        if gap_size > 50 and not is_entry:

            routing_trace.append("  -> Attempting ML methods (gap_size > 50 and not entry)")

            # DEEP LEARNING: Try autoencoder first (most sophisticated)

            if self.use_deep_learning and self._profile_imputer_fitted:

                if self._fill_with_autoencoder(df, site, gap_start, gap_end):

                    strategy = 'AUTOENCODER'

                    routing_trace.append(f"  -> Selected: {strategy}")

                    day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)

                    occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

                    is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False

                    confidence = self._calculate_confidence_with_uncertainty(

                        site, gap_size, day_type, strategy, occupancy_type, is_holiday

                    )

                    self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

                    self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

                    self.strategy_log.append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

                    })

                    self._audit_log['gaps'].append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

                        'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

                    })

                    return

            if self.use_mice:

                if self._fill_with_mice(df, site, gap_start, gap_end):

                    strategy = 'MICE'

                    routing_trace.append(f"  -> Selected: {strategy}")

                    day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)

                    occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

                    is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False

                    confidence = self._calculate_confidence_with_uncertainty(

                        site, gap_size, day_type, strategy, occupancy_type, is_holiday

                    )

                    self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

                    self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

                    self.strategy_log.append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

                    })

                    self._audit_log['gaps'].append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

                        'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

                    })

                    return

            if self.use_kalman:

                if self._fill_with_kalman_filter(df, site, gap_start, gap_end):

                    strategy = 'KALMAN_FILTER'

                    routing_trace.append(f"  -> Selected: {strategy}")

                    day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)

                    occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

                    is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False

                    confidence = self._calculate_confidence_with_uncertainty(

                        site, gap_size, day_type, strategy, occupancy_type, is_holiday

                    )

                    self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

                    self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

                    self.strategy_log.append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

                    })

                    self._audit_log['gaps'].append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

                        'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

                    })

                    return

            if self.use_knn:

                if self._fill_with_knn_context(df, site, gap_start, gap_end, k=5):

                    strategy = 'KNN_CONTEXT'

                    routing_trace.append(f"  -> Selected: {strategy}")

                    day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)

                    occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

                    is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False

                    confidence = self._calculate_confidence_with_uncertainty(

                        site, gap_size, day_type, strategy, occupancy_type, is_holiday

                    )

                    self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

                    self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

                    self.strategy_log.append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

                    })

                    self._audit_log['gaps'].append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

                        'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

                    })

                    return



        # Sub-meter correlation

        if is_submeter:

            routing_trace.append("  -> Sub-meter detected, checking parent")

            parent = METER_HIERARCHY[site]['parent']

            if parent and parent in df.columns:

                parent_valid = df.loc[gap_start:gap_end - 1, parent].notna().any()

                if parent_valid:

                    routing_trace.append(f"  -> Parent data available ({parent})")

                    self._fill_via_peer_correlation(df, site, parent, gap_start, gap_end)

                    strategy = 'PEER_CORRELATION'

                    routing_trace.append(f"  -> Selected: {strategy}")

                    day_type = 'weekday' if not is_weekend else self._get_weekend_day_type(df, gap_start, gap_end)

                    occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

                    is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False

                    confidence = self._calculate_confidence_with_uncertainty(

                        site, gap_size, day_type, strategy, occupancy_type, is_holiday

                    )

                    self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

                    self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

                    self.strategy_log.append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

                    })

                    self._audit_log['gaps'].append({

                        'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

                        'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

                        'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

                        'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

                    })

                    return



        # Route by gap size and context

        routing_trace.append("  -> Template-based routing")

        if gap_size <= 3:

            self._fill_linear(df, site, gap_start, gap_end)

            strategy = 'LINEAR_MICRO'

            routing_trace.append(f"  -> Selected: {strategy} (gap_size <= 3)")

        elif gap_size <= 18:

            self._fill_linear(df, site, gap_start, gap_end)

            strategy = 'LINEAR_SHORT'

            routing_trace.append(f"  -> Selected: {strategy} (gap_size <= 18)")

        elif self._is_pure_weekend_gap(df, gap_start, gap_end):

            day_type = self._get_weekend_day_type(df, gap_start, gap_end)

            self._fill_weekend_template(df, site, gap_start, gap_end, day_type)

            strategy = f'WEEKEND_TEMPLATE_{day_type.upper()}'

            routing_trace.append(f"  -> Selected: {strategy} (pure weekend)")

        elif not is_entry and gap_size <= STEPS_PER_DAY:

            self._fill_with_thermal_template(df, site, gap_start, gap_end)

            strategy = 'THERMAL_TEMPLATE'

            routing_trace.append(f"  -> Selected: {strategy} (thermal, gap_size <= 1 day)")

        elif not is_entry:

            self._fill_enhanced_template(df, site, gap_start, gap_end)

            strategy = 'ENHANCED_TEMPLATE'

            routing_trace.append(f"  -> Selected: {strategy} (enhanced)")

        else:

            self._fill_safe_linear_median(df, site, gap_start, gap_end)

            strategy = 'SAFE_LINEAR_MEDIAN'

            routing_trace.append(f"  -> Selected: {strategy} (entry meter fallback)")



        # Calculate extended confidence

        day_type = 'weekday'

        if is_weekend:

            day_type = self._get_weekend_day_type(df, gap_start, gap_end)



        occupancy_type = df.loc[gap_start, 'OccupancyType'] if 'OccupancyType' in df.columns else None

        is_holiday = bool(df.loc[gap_start, 'IsHoliday']) if 'IsHoliday' in df.columns else False



        confidence = self._calculate_confidence_with_uncertainty(

            site, gap_size, day_type, strategy, occupancy_type, is_holiday

        )



        self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence

        self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)



        self.strategy_log.append({

            'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

            'gap_size': gap_size, 'strategy': strategy, 'confidence': confidence,

            'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday

        })

        

        # Log to audit log

        self._audit_log['gaps'].append({

            'site': site, 'gap_start': gap_start, 'gap_end': gap_end,

            'gap_size': gap_size, 'dates': metadata.get('timestamps', []),

            'method': strategy, 'confidence': confidence, 'routing_trace': routing_trace,

            'day_type': day_type, 'occupancy_type': occupancy_type, 'is_holiday': is_holiday,

        })



    # ─────────────────────────────────────────────────────────────────────────

    # Fill primitives

    # ─────────────────────────────────────────────────────────────────────────



    def _fill_linear(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Linear interpolation."""

        try:

            left_val = df.loc[gap_start - 1, site] if gap_start > 0 else np.nan

            right_val = df.loc[gap_end, site] if gap_end < len(df) else np.nan



            if np.isnan(left_val) or np.isnan(right_val):

                self._fill_safe_median_template(df, site, gap_start, gap_end)

                return



            filled = np.linspace(left_val, right_val, gap_end - gap_start + 2)[1:-1]

            df.loc[gap_start:gap_end - 1, site] = filled

        except Exception as e:

            log.error(f"[ERROR] Linear fill failed for {site}: {e}")

            self._fill_safe_median_template(df, site, gap_start, gap_end)



    def _fill_weekend_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, day_type: str):

        """Fill weekend gap using day-specific template."""

        try:

            template = self.templates.get(day_type.lower(), {}).get(site, {})

            if not template:

                self._fill_safe_median_template(df, site, gap_start, gap_end)

                return



            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values

            filled_vals = [template.get(h, np.nan) for h in hours]

            filled_vals = np.array(filled_vals)

            filled_vals = self._validate_and_clip(filled_vals, site, df)

            df.loc[gap_start:gap_end - 1, site] = filled_vals

        except Exception as e:

            log.error(f"[ERROR] Weekend template fill failed: {e}")

            self._fill_safe_median_template(df, site, gap_start, gap_end)



    def _fill_with_thermal_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Thermal-aware template fill."""

        try:

            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values

            template_vals = []

            for h in hours:

                hist_mask = (df['Hour'] == h) & (df[site].notna())

                if hist_mask.any():

                    template_vals.append(df.loc[hist_mask, site].median())

                else:

                    template_vals.append(np.nan)



            filled_vals = np.array(template_vals)

            filled_vals = self._validate_and_clip(filled_vals, site, df)

            df.loc[gap_start:gap_end - 1, site] = filled_vals

        except Exception as e:

            log.error(f"[ERROR] Thermal template failed: {e}")

            self._fill_safe_median_template(df, site, gap_start, gap_end)



    def _fill_enhanced_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Enhanced template with multi-mode selection."""

        self._fill_with_thermal_template(df, site, gap_start, gap_end)



    def _fill_safe_linear_median(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Blend linear with median for entry points."""

        try:

            left_val = df.loc[gap_start - 1, site] if gap_start > 0 else np.nan

            right_val = df.loc[gap_end, site] if gap_end < len(df) else np.nan



            if np.isnan(left_val) or np.isnan(right_val):

                self._fill_safe_median_template(df, site, gap_start, gap_end)

                return



            filled = np.linspace(left_val, right_val, gap_end - gap_start + 2)[1:-1]

            valid_vals = df[site].dropna()

            if len(valid_vals) > 0:

                median_val = valid_vals.median()

                filled = filled * 0.9 + median_val * 0.1



            df.loc[gap_start:gap_end - 1, site] = filled

        except Exception as e:

            log.error(f"[ERROR] Safe linear-median failed: {e}")

            self._fill_safe_median_template(df, site, gap_start, gap_end)



    def _fill_safe_median_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Fallback: simple median."""

        try:

            valid_vals = df[site].dropna()

            if len(valid_vals) > 0:

                df.loc[gap_start:gap_end - 1, site] = valid_vals.median()

            else:

                df.loc[gap_start:gap_end - 1, site] = 0.0

        except Exception as e:

            log.error(f"[ERROR] Safe median failed: {e}")

            df.loc[gap_start:gap_end - 1, site] = 0.0



    def _fill_via_peer_correlation(self, df: pd.DataFrame, child: str, parent: str, gap_start: int, gap_end: int):

        """Fill sub-meter using parent ratio."""

        try:

            ratio = self._peer_ratios.get(child, 0.5)

            parent_values = df.loc[gap_start:gap_end - 1, parent].values

            filled_vals = parent_values * ratio

            filled_vals = self._validate_and_clip(filled_vals, child, df)

            df.loc[gap_start:gap_end - 1, child] = filled_vals

        except Exception as e:

            log.error(f"[ERROR] Peer correlation failed: {e}")

            self._fill_safe_median_template(df, child, gap_start, gap_end)



    def _fill_with_autoencoder(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int) -> bool:

        """Fill gap using autoencoder-based profile imputation (ProfileImputer)."""

        try:

            if not self._profile_imputer_fitted or self._profile_imputer is None:

                return False

            

            # Use the fitted autoencoder to fill the gap

            df_subset = df[[c for c in df.columns if c in ['Timestamp'] + self.site_cols]].copy()

            df_filled = self._profile_imputer.transform(df_subset, site=site, temp_col='AirTemp', noise=False)

            

            # Copy filled values back to original dataframe

            df.loc[gap_start:gap_end - 1, site] = df_filled.loc[gap_start:gap_end - 1, site].values

            

            log.debug(f"[DL] Autoencoder filled {gap_end - gap_start} points for {site}")

            return True

        except Exception as e:

            log.debug(f"[DL] Autoencoder fill failed for {site}: {e}")

            return False



    def _fill_with_mice(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, iterations: int = 3):

        """MICE imputation using sklearn IterativeImputer."""

        try:

            from sklearn.experimental import enable_iterative_imputer

            from sklearn.impute import IterativeImputer

            cols = [site] + [c for c in self.site_cols if c != site and c in df.columns]

            data = df.loc[:, cols].values

            imp = IterativeImputer(max_iter=iterations, random_state=0)

            imputed = imp.fit_transform(data)

            df.loc[gap_start:gap_end - 1, site] = imputed[gap_start:gap_end, 0]

            return True

        except Exception as e:

            log.error(f"[ERROR] MICE imputation failed: {e}")

            return False



    def _fill_with_kalman_filter(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):

        """Simple Kalman filter imputation for missing values."""

        try:

            try:

                from pykalman import KalmanFilter

                values = df[site].values

                mask = np.isnan(values)

                values_filled = values.copy()

                values_filled[mask] = 0

                kf = KalmanFilter()

                values_smoothed, _ = kf.smooth(values_filled)

                df.loc[gap_start:gap_end - 1, site] = values_smoothed[gap_start:gap_end, 0]

                return True

            except ImportError:

                df[site] = df[site].interpolate(method='linear')

                return True

        except Exception as e:

            log.error(f"[ERROR] Kalman filter imputation failed: {e}")

            return False



    def _fill_with_knn_context(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, k: int = 5):

        """KNN context matching for imputation."""

        try:

            context_cols = [c for c in self.site_cols if c != site and c in df.columns]

            if not context_cols:

                return False

            valid_mask = df[site].notna()

            if valid_mask.sum() < k:

                return False

            X = df.loc[valid_mask, context_cols].values

            y = df.loc[valid_mask, site].values

            nbrs = NearestNeighbors(n_neighbors=k)

            nbrs.fit(X)

            gap_X = df.loc[gap_start:gap_end - 1, context_cols].values

            distances, indices = nbrs.kneighbors(gap_X)

            preds = np.array([y[inds].mean() for inds in indices])

            df.loc[gap_start:gap_end - 1, site] = preds

            return True

        except Exception as e:

            log.error(f"[ERROR] KNN context imputation failed: {e}")

            return False



    def _validate_and_clip(self, values: np.ndarray, site: str, df: pd.DataFrame) -> np.ndarray:

        """Validate and clip values."""

        values = np.array(values, dtype=float)

        nan_mask = np.isnan(values)

        if np.any(nan_mask):

            valid_historical = df[site].dropna()

            if len(valid_historical) > 0:

                values[nan_mask] = valid_historical.median()

            else:

                values[nan_mask] = 0.0



        valid_historical = df[site].dropna()

        if len(valid_historical) > 0:

            peak = valid_historical.max()

            values = np.clip(values, 0, peak * 1.5)



        return values



    def _nan_guard_final_pass(self, df: pd.DataFrame):

        """Final NaN elimination pass."""

        for site in self.site_cols:

            if site not in df.columns:

                continue



            nan_mask = df[site].isna()

            if not nan_mask.any():

                continue



            for gap_start, gap_end in self._find_gap_groups(nan_mask):

                self._fill_safe_median_template(df, site, gap_start, gap_end)



    def _smooth_junctions(self, df: pd.DataFrame, site: str):

        """Smooth junction points."""

        try:

            gap_mask = df[site].isna()

            if not gap_mask.any():

                return



            for gap_start, gap_end in self._find_gap_groups(gap_mask):

                window_size = 21

                if gap_start > window_size // 2 and gap_start + window_size // 2 < len(df):

                    window = df.loc[gap_start - window_size // 2:gap_start + window_size // 2, site].values

                    if not np.all(np.isnan(window)):

                        window_filled = pd.Series(window).ffill().bfill().values

                        try:

                            smoothed = savgol_filter(window_filled, window_size, 3)

                            valid_mask = ~pd.isna(

                                df.loc[gap_start - window_size // 2:gap_start + window_size // 2, site].values

                            )

                            df.loc[gap_start - window_size // 2:gap_start + window_size // 2, site] = \
                                np.where(

                                    valid_mask, smoothed,

                                    df.loc[gap_start - window_size // 2:gap_start + window_size // 2, site].values

                                )

                        except Exception:

                            pass

        except Exception as e:

            log.debug(f"[DEBUG] Smoothing failed: {e}")



    # ─────────────────────────────────────────────────────────────────────────

    # Helper Methods

    # ─────────────────────────────────────────────────────────────────────────



    def _add_datetime_features(self, df: pd.DataFrame):

        """Add datetime features."""

        df['Hour'] = df['Timestamp'].dt.hour

        df['DayOfWeek'] = df['Timestamp'].dt.day_name()

        df['Date'] = df['Timestamp'].dt.date



    def _fill_airtemp_forward(self, df: pd.DataFrame):

        """Forward fill air temperature."""

        if 'AirTemp' in df.columns:

            df['AirTemp'] = df['AirTemp'].ffill().bfill()



    def _merge_weather(self, df: pd.DataFrame) -> pd.DataFrame:

        """Merge weather data."""

        if self.weather_df is None:

            return df



        weather = self.weather_df.copy()

        weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])

        df = df.merge(weather[['Timestamp', 'AirTemp']], on='Timestamp', how='left')

        if 'AirTemp' in df.columns:

            df['AirTemp'] = df['AirTemp'].ffill().bfill()

        return df



    def _classify_thermal_regimes(self, df: pd.DataFrame):

        """Classify thermal regimes."""

        if 'AirTemp' not in df.columns:

            df['ThermalRegime'] = 'Mild'

        else:

            df['ThermalRegime'] = pd.cut(

                df['AirTemp'],

                bins=[-np.inf, 10, 20, np.inf],

                labels=['Cold', 'Mild', 'Hot']

            )



    def _build_day_specific_templates(self, df: pd.DataFrame):

        """Build Saturday/Sunday/holiday templates."""

        self.templates['saturday'] = {}

        self.templates['sunday'] = {}

        self.templates['holiday'] = {}



        for site in self.site_cols:

            if site not in df.columns:

                continue



            sat_mask = (df['DayOfWeek'] == 'Saturday') & (df[site].notna())

            if sat_mask.any():

                self.templates['saturday'][site] = \
                    df.loc[sat_mask].groupby('Hour')[site].median().to_dict()



            sun_mask = (df['DayOfWeek'] == 'Sunday') & (df[site].notna())

            if sun_mask.any():

                self.templates['sunday'][site] = \
                    df.loc[sun_mask].groupby('Hour')[site].median().to_dict()



            if 'IsHoliday' in df.columns:

                holiday_mask = (df['IsHoliday']) & (df[site].notna())

                if holiday_mask.any():

                    self.templates['holiday'][site] = \
                        df.loc[holiday_mask].groupby('Hour')[site].median().to_dict()



    def _build_seasonal_templates(self, df: pd.DataFrame):

        """Build seasonal templates."""

        try:

            for site in self.site_cols:

                if site not in df.columns:

                    continue

                self._seasonal_templates[site] = {}

        except Exception as e:

            log.debug(f"[DEBUG] Seasonal templates failed: {e}")



    def _build_peer_ratios(self, df: pd.DataFrame):

        """Build peer ratios."""

        for site, info in METER_HIERARCHY.items():

            if info['tier'] != 'sub':

                continue



            parent = info['parent']

            if parent is None or parent not in df.columns:

                continue



            last_24h_mask = (df[site].notna()) & (df[parent].notna())

            if last_24h_mask.any():

                last_indices = np.where(last_24h_mask)[0][-min(STEPS_PER_DAY, last_24h_mask.sum()):]

                child_vals = df.loc[last_indices, site].values

                parent_vals = df.loc[last_indices, parent].values



                parent_nonzero = parent_vals > 1

                if parent_nonzero.any():

                    ratios = child_vals[parent_nonzero] / parent_vals[parent_nonzero]

                    self._peer_ratios[site] = np.median(ratios[~np.isnan(ratios)])

                else:

                    self._peer_ratios[site] = 0.5

            else:

                self._peer_ratios[site] = 0.5



    def _build_multi_site_correlations(self, df: pd.DataFrame):

        """Build correlations between sites."""

        try:

            for site1 in self.site_cols:

                if site1 not in df.columns:

                    continue

                self._multi_site_correlations[site1] = {}



                for site2 in self.site_cols:

                    if site2 == site1 or site2 not in df.columns:

                        continue



                    valid_mask = (df[site1].notna()) & (df[site2].notna())

                    if valid_mask.sum() > 10:

                        corr = df.loc[valid_mask, [site1, site2]].corr().iloc[0, 1]

                        self._multi_site_correlations[site1][site2] = corr if not np.isnan(corr) else 0

        except Exception as e:

            log.debug(f"[DEBUG] Multi-site correlation failed: {e}")



    def _is_weekend_gap(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> bool:

        """Check if gap includes weekend."""

        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()

        return any(d in ['Saturday', 'Sunday'] for d in dow_values)



    def _is_pure_weekend_gap(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> bool:

        """Check if gap is pure weekend."""

        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()

        all_days = set(dow_values)

        weekdays = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}

        return all_days.isdisjoint(weekdays)



    def _get_weekend_day_type(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> str:

        """Get weekend day type."""

        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()

        has_sat = 'Saturday' in dow_values

        has_sun = 'Sunday' in dow_values



        if has_sat and has_sun:

            return 'mixed'

        elif has_sat:

            return 'saturday'

        elif has_sun:

            return 'sunday'

        else:

            return 'mixed'



    def _find_gap_groups(self, mask: pd.Series, df: pd.DataFrame) -> List[Tuple[int, int, Dict]]:

        """Find consecutive gap groups with metadata (timestamps, row count, DST flag)."""

        indices = np.where(mask)[0]

        if len(indices) == 0:

            return []



        gaps = []

        start = indices[0]



        for i in range(1, len(indices)):

            if indices[i] - indices[i - 1] > 1:

                gap_start, gap_end = start, indices[i - 1] + 1

                metadata = {

                    'timestamps': df.loc[gap_start:gap_end - 1, 'Timestamp'].tolist(),

                    'row_count': gap_end - gap_start,

                    'date_range': f"{df.loc[gap_start, 'Timestamp']} to {df.loc[gap_end - 1, 'Timestamp']}",

                    'dst_flag': False,  # Can be set by caller if DST event

                }

                gaps.append((gap_start, gap_end, metadata))

                start = indices[i]



        gap_start, gap_end = start, indices[-1] + 1

        metadata = {

            'timestamps': df.loc[gap_start:gap_end - 1, 'Timestamp'].tolist(),

            'row_count': gap_end - gap_start,

            'date_range': f"{df.loc[gap_start, 'Timestamp']} to {df.loc[gap_end - 1, 'Timestamp']}",

            'dst_flag': False,

        }

        gaps.append((gap_start, gap_end, metadata))

        return gaps



    def get_low_confidence_report(self) -> pd.DataFrame:

        """Get report of low-confidence gaps for review."""

        if not self._low_confidence_flags:

            return pd.DataFrame()



        return pd.DataFrame(self._low_confidence_flags)



    def get_strategy_log(self) -> pd.DataFrame:

        """Get full strategy log."""

        if not self.strategy_log:

            return pd.DataFrame()



        return pd.DataFrame(self.strategy_log)