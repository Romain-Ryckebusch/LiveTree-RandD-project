"""Deployment gap-recovery engine used by imputer.py.

TemperatureAwareHybridEngine runs multi-week adaptive templates, smart
chunked recovery, occupancy/calendar features, uncertainty bounds, and
peer-correlation fallback for sub-meters.
"""

import hashlib
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
    """Load France holidays/close/special days from Consumption_<year>_<type>.csv
    files if any are present, otherwise fall back to a hardcoded 2026 list."""
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
            except Exception:
                pass

    fallback_holidays = [
        (1, 1), (4, 5), (4, 6), (5, 1), (5, 8), (5, 14), (5, 25),
        (7, 14), (8, 15), (11, 1), (11, 11), (12, 25),
    ]
    for m, d in fallback_holidays:
        holidays.add(date(2026, m, d))

    return holidays, close_days, special_days


FRANCE_HOLIDAYS_2026, FRANCE_CLOSE_DAYS_2026, FRANCE_SPECIAL_DAYS_2026 = _load_all_holidays()


class TemperatureAwareHybridEngine:
    """Gap-recovery engine with multi-week templates, chunked recovery for
    long gaps, peer correlation for sub-meters, and uncertainty bounds."""

    def __init__(
        self,
        site_cols: Optional[List[str]] = None,
        weather_df: Optional[pd.DataFrame] = None,
        use_historical_data: bool = True,
        template_cache_file: str = 'templates_cache.pkl',
        low_variance_threshold: float = 5000.0,
        use_mice: bool = False,
        use_knn: bool = False,
        use_kalman: bool = False,
        use_deep_learning: bool = False,
        use_multi_week_templates: bool = True,
        use_chunked_recovery: bool = True,
        gap_chunk_size: int = 96,
        occupancy_data: Optional[pd.DataFrame] = None,
        calendar_data: Optional[pd.DataFrame] = None,
        template_lookback_days: int = 28,
        use_smart_chunking: bool = True,
        adaptive_template_bias: float = 0.7,
    ):
        self.site_cols = site_cols or SITE_COLS
        self.weather_df = weather_df
        self.use_historical_data = use_historical_data
        self.historical_df: Optional[pd.DataFrame] = None
        self.template_cache_file = template_cache_file
        self.low_variance_threshold = low_variance_threshold

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

        self.strategy_log: List[Dict] = []
        self._peer_ratios: Dict[str, float] = {}
        self.templates: Dict = {}

        self._kalman_states: Dict[str, np.ndarray] = {}
        self._kalman_covariance: Dict[str, float] = {}
        self._knn_models: Dict[str, NearestNeighbors] = {}
        self._fitted_knn: bool = False

        self._reconstruction_confidence: Dict[str, float] = {}
        self._low_confidence_flags: List[Dict] = []
        self._seasonal_templates: Dict[str, Dict] = {}
        self._multi_site_correlations: Dict[str, Dict[str, float]] = {}

        self._weekly_templates: Dict[str, Dict[str, Dict]] = {}
        self._uncertainty_bounds: Dict[str, Tuple[float, float]] = {}
        self._occupancy_patterns: Dict[str, Dict] = {}
        self._day_variance: Dict[str, Dict[str, float]] = {}
        self._external_event_flags: Dict[str, List[str]] = {}
        self._weather_variance: Dict[str, float] = {}

    def impute(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Main entry. Runs the full pipeline and returns a DataFrame with
        Timestamp + the site columns (+ AirTemp if it's around)."""
        self.strategy_log = []

        df = df.copy().sort_values('Timestamp').reset_index(drop=True)

        if (
            self.use_historical_data
            and self.historical_df is not None
            and isinstance(self.historical_df, pd.DataFrame)
            and not self.historical_df.empty
            and 'Timestamp' in self.historical_df.columns
        ):
            df_start = df['Timestamp'].min()
            hist_past = self.historical_df[self.historical_df['Timestamp'] < df_start]
            if not hist_past.empty:
                df = (
                    pd.concat([hist_past, df], ignore_index=True, sort=False)
                    .drop_duplicates(subset=['Timestamp'], keep='last')
                    .sort_values('Timestamp')
                    .reset_index(drop=True)
                )

        if weather_df is not None:
            self.weather_df = weather_df
            df = self._merge_weather(df)

        start = df['Timestamp'].min().floor('10min')
        end = df['Timestamp'].max().ceil('10min')
        df = (
            df.set_index('Timestamp')
            .reindex(pd.date_range(start, end, freq='10min'))
            .reset_index()
        )
        df.columns = ['Timestamp'] + list(df.columns[1:])

        self._add_datetime_features(df)
        self._fill_airtemp_forward(df)
        self._classify_thermal_regimes(df)

        self._add_occupancy_features(df)
        self._add_external_features(df)

        self._build_peer_ratios(df)

        self._build_day_specific_templates(df)
        self._build_seasonal_templates(df)

        if self.use_multi_week_templates:
            self._build_weekly_templates(df)
            self._build_uncertainty_bounds(df)

        self._build_multi_site_correlations(df)

        for site in self.site_cols:
            if site not in df.columns:
                continue

            gap_mask = df[site].isna()
            if not gap_mask.any():
                continue

            for gap_start, gap_end in self._find_gap_groups(gap_mask):
                gap_size = gap_end - gap_start
                if self.use_chunked_recovery and gap_size > self.gap_chunk_size:
                    self._fill_chunked_gap(df, site, gap_start, gap_end)
                else:
                    self._intelligent_router(df, site, gap_start, gap_end)

        self._nan_guard_final_pass(df)

        for site in self.site_cols:
            if site in df.columns:
                self._smooth_junctions(df, site)

        out_cols = ['Timestamp'] + self.site_cols + (['AirTemp'] if 'AirTemp' in df.columns else [])
        return df[[c for c in out_cols if c in df.columns]]

    def _fill_chunked_gap(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        gap_size = gap_end - gap_start

        if self.use_smart_chunking:
            try:
                chunks = self._get_smart_chunks(df, gap_start, gap_end)
            except Exception as e:
                log.debug(f"smart chunking failed: {e}; falling back to fixed-size")
                chunks = self._fixed_size_chunks(gap_start, gap_end)
            log.info(f"smart-chunked {gap_size} pts ({site}) -> {len(chunks)} chunks")
        else:
            chunks = self._fixed_size_chunks(gap_start, gap_end)
            log.info(f"chunked {gap_size} pts ({site}) -> {len(chunks)} chunks")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_size = chunk_end - chunk_start

            if self.use_multi_week_templates and self._weekly_templates:
                self._fill_with_multi_week_template(df, site, chunk_start, chunk_end)
            else:
                self._intelligent_router(df, site, chunk_start, chunk_end)

            log.debug(f"filled chunk {chunk_idx+1}/{len(chunks)}: "
                      f"indices {chunk_start}-{chunk_end}, size {chunk_size}")

    def _fixed_size_chunks(self, gap_start: int, gap_end: int) -> List[Tuple[int, int]]:
        gap_size = gap_end - gap_start
        num_chunks = int(np.ceil(gap_size / self.gap_chunk_size))
        return [
            (gap_start + (i * self.gap_chunk_size),
             min(gap_start + ((i + 1) * self.gap_chunk_size), gap_end))
            for i in range(num_chunks)
        ]

    def _get_smart_chunks(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> List[Tuple[int, int]]:
        """Prefer day boundaries and keep high-variance days together."""
        chunks = []
        current_chunk_start = gap_start
        current_chunk_size = 0
        max_chunk_size = self.gap_chunk_size * 1.5

        gap_dates = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values
        gap_hours = df.loc[gap_start:gap_end - 1, 'Hour'].values

        primary_site = self.site_cols[0] if self.site_cols else None
        site_variance = self._day_variance.get(primary_site, {}) if primary_site else {}

        daily_variances: List[float] = []
        for i, hour in enumerate(gap_hours):
            if hour == 0 or i == 0:
                day_name = gap_dates[i]
                v = site_variance.get(day_name)
                daily_variances.append(float(v) if v is not None else 100.0)

        v_threshold = float(np.median(daily_variances)) if daily_variances else 100.0
        high_variance_days = set()
        for i in range(len(gap_dates)):
            day_idx = i // 144
            if day_idx < len(daily_variances) and daily_variances[day_idx] > v_threshold * 1.3:
                high_variance_days.add(gap_dates[i])

        for idx in range(gap_start, gap_end):
            current_chunk_size += 1
            is_high_variance = gap_dates[idx - gap_start] in high_variance_days
            is_new_day = (gap_hours[idx - gap_start] == 0) and idx > gap_start

            should_chunk_break = False
            if current_chunk_size >= max_chunk_size:
                should_chunk_break = True
            elif is_high_variance and current_chunk_size > self.gap_chunk_size * 0.8:
                should_chunk_break = True
            elif is_new_day and current_chunk_size >= self.gap_chunk_size:
                should_chunk_break = True

            if should_chunk_break and idx < gap_end:
                chunks.append((current_chunk_start, idx))
                current_chunk_start = idx
                current_chunk_size = 0

        if current_chunk_start < gap_end:
            chunks.append((current_chunk_start, gap_end))

        log.debug(f"smart chunks: {len(chunks)}, high-var threshold: {v_threshold:.1f}")
        return chunks

    def _build_weekly_templates(self, df: pd.DataFrame):
        """28-day adaptive templates per (site, day-of-week, hour). Recent data
        gets `adaptive_template_bias` weight, older data gets the rest."""
        try:
            max_date = df['Timestamp'].max()
            min_date = max_date - pd.Timedelta(days=self.template_lookback_days)

            recent_cutoff = max_date - pd.Timedelta(days=int(self.template_lookback_days * 0.3))
            recent_mask = df['Timestamp'] >= recent_cutoff
            historical_mask = (df['Timestamp'] >= min_date) & (df['Timestamp'] < recent_cutoff)

            # Exclude holidays from the per-DoW template. Otherwise Easter Sunday
            # (say) poisons every non-holiday Sunday's template.
            non_holiday_mask = (
                ~df['IsHoliday'] if 'IsHoliday' in df.columns else pd.Series(True, index=df.index)
            )

            for site in self.site_cols:
                if site not in df.columns:
                    continue

                self._weekly_templates[site] = {}
                self._day_variance[site] = {}

                for day_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    day_templates = {}
                    day_values_all = []

                    for hour in range(24):
                        recent = df.loc[
                            recent_mask & non_holiday_mask & (df['DayOfWeek'] == day_name) &
                            (df['Hour'] == hour) & (df[site].notna()), site
                        ].values

                        historical = df.loc[
                            historical_mask & non_holiday_mask & (df['DayOfWeek'] == day_name) &
                            (df['Hour'] == hour) & (df[site].notna()), site
                        ].values

                        if len(recent) > 0 and len(historical) > 0:
                            recent_median = np.median(recent)
                            historical_median = np.median(historical)
                            blended_median = (
                                self.adaptive_template_bias * recent_median
                                + (1 - self.adaptive_template_bias) * historical_median
                            )
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

            log.info(
                f"weekly templates built ({self.template_lookback_days}-day lookback, "
                f"{int(self.adaptive_template_bias * 100)}% recency bias)"
            )
        except Exception as e:
            log.error(f"failed to build weekly templates: {e}")

    def _fill_with_multi_week_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        try:
            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            day_names = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].values

            filled_vals = []
            confidences = []

            for h, day_name in zip(hours, day_names):
                template = self._weekly_templates.get(site, {}).get(day_name, {}).get(h)

                if template is not None and template.get('median') is not None:
                    filled_vals.append(template['median'])
                    if template['std'] > 0:
                        confidence = 1.0 / (1.0 + template['std'] / (abs(template['median']) + 1e-6))
                    else:
                        confidence = 1.0
                    confidences.append(confidence)
                else:
                    mask = (df['Hour'] == h) & (df[site].notna())
                    if mask.any():
                        filled_vals.append(df.loc[mask, site].median())
                        confidences.append(0.5)
                    else:
                        filled_vals.append(np.nan)
                        confidences.append(0.0)

            filled_vals = np.array(filled_vals, dtype=float)
            filled_vals = self._apply_edge_calibration(df, site, filled_vals, gap_start, gap_end)
            filled_vals = self._validate_and_clip(filled_vals, site, df)
            df.loc[gap_start:gap_end - 1, site] = filled_vals

            avg_confidence = float(np.nanmean(confidences)) if confidences else 0.0
            self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = avg_confidence

            if avg_confidence < 0.5:
                self._low_confidence_flags.append({
                    'site': site,
                    'gap_start': int(gap_start),
                    'gap_end': int(gap_end),
                    'gap_size': int(gap_end - gap_start),
                    'reason': 'multi_week_template_low_confidence',
                    'confidence': avg_confidence,
                    'occupancy': df.loc[gap_start, 'IsOccupied'] if 'IsOccupied' in df.columns else None,
                    'is_holiday': df.loc[gap_start, 'IsHoliday'] if 'IsHoliday' in df.columns else None,
                })

            self.strategy_log.append({
                'site': site,
                'gap_start': int(gap_start),
                'gap_end': int(gap_end),
                'gap_size': int(gap_end - gap_start),
                'strategy': 'MULTI_WEEK_TEMPLATE',
                'confidence': avg_confidence,
            })

        except Exception as e:
            log.error(f"multi-week template fill failed for {site}: {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _add_occupancy_features(self, df: pd.DataFrame):
        """Derive IsOccupied / OccupancyType from weekend, holiday, and hour."""
        try:
            df['OccupancyType'] = 'work_hours'
            df['IsOccupied'] = True

            weekend_mask = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
            df.loc[weekend_mask, 'OccupancyType'] = 'weekend'
            df.loc[weekend_mask, 'IsOccupied'] = False

            if self.calendar_data is not None:
                holiday_list = self.calendar_data.get('holiday_list', []) if hasattr(self.calendar_data, 'get') else []
                holiday_dates = set(pd.to_datetime(holiday_list).date) if len(holiday_list) else set()
                is_holiday = df['Date'].isin(holiday_dates)
            else:
                is_holiday = df['Date'].isin(FRANCE_HOLIDAYS_2026)

            df.loc[is_holiday, 'OccupancyType'] = 'holiday'
            df.loc[is_holiday, 'IsOccupied'] = False

            evening_mask = (df['Hour'] >= 18) | (df['Hour'] < 6)
            work_hours_mask = df['OccupancyType'] == 'work_hours'
            df.loc[evening_mask & ~weekend_mask & work_hours_mask, 'OccupancyType'] = 'evening'
            df.loc[evening_mask & ~weekend_mask & work_hours_mask, 'IsOccupied'] = False

            log.info("occupancy features added")
        except Exception as e:
            log.error(f"failed to add occupancy features: {e}")
            df['IsOccupied'] = True
            df['OccupancyType'] = 'unknown'

    def _add_external_features(self, df: pd.DataFrame):
        try:
            df['IsHoliday'] = df['Date'].isin(FRANCE_HOLIDAYS_2026)
            df['IsSpecialDay'] = (
                df['Date'].isin(FRANCE_CLOSE_DAYS_2026)
                | df['Date'].isin(FRANCE_SPECIAL_DAYS_2026)
            )

            df['IsHolidayClose'] = False
            for holiday_date in FRANCE_HOLIDAYS_2026:
                period_mask = (
                    (df['Date'] >= holiday_date - timedelta(days=1))
                    & (df['Date'] <= holiday_date + timedelta(days=1))
                )
                df.loc[period_mask, 'IsHolidayClose'] = True

            df['IsEventDay'] = False
            for site in self.site_cols:
                if site in df.columns:
                    daily_std = df.groupby('Date')[site].std()
                    if not daily_std.empty:
                        high_std_threshold = daily_std.quantile(0.75)
                        high_std_dates = daily_std[daily_std > high_std_threshold].index
                        df.loc[df['Date'].isin(high_std_dates), 'IsEventDay'] = True

            df['WeatherSpike'] = False
            if 'AirTemp' in df.columns:
                daily_temp_change = df.groupby('Date')['AirTemp'].apply(
                    lambda x: abs(x.max() - x.min()) if len(x) > 0 else 0
                )
                if not daily_temp_change.empty:
                    temp_spike_threshold = daily_temp_change.quantile(0.80)
                    spike_dates = daily_temp_change[daily_temp_change > temp_spike_threshold].index
                    df.loc[df['Date'].isin(spike_dates), 'WeatherSpike'] = True

            df['Season'] = df['Timestamp'].dt.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall',
            })

            log.info("external features added (holiday close, events, weather spikes)")
        except Exception as e:
            log.error(f"failed to add external features: {e}")

    def _build_uncertainty_bounds(self, df: pd.DataFrame):
        try:
            for site in self.site_cols:
                if site not in df.columns:
                    continue

                valid_data = df[site].dropna()
                if len(valid_data) < 10:
                    upper = float(valid_data.max() * 1.5) if len(valid_data) > 0 else 0.0
                    self._uncertainty_bounds[site] = (0.0, upper)
                    continue

                mean = float(valid_data.mean())
                std = float(valid_data.std())
                lower_bound = max(0.0, mean - 2 * std)
                upper_bound = mean + 2 * std

                self._uncertainty_bounds[site] = (lower_bound, upper_bound)
                log.debug(f"{site} bounds ({lower_bound:.1f}, {upper_bound:.1f})")

            log.info("uncertainty bounds built")
        except Exception as e:
            log.error(f"failed to build uncertainty bounds: {e}")

    def _calculate_confidence_with_uncertainty(
        self,
        site: str,
        gap_size: int,
        day_type: str,
        strategy: str,
        occupancy_type: Optional[str] = None,
        is_holiday: bool = False,
    ) -> float:
        try:
            strategy_confidence_map = {
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

            if gap_size < 10:
                gap_factor = 1.0
            elif gap_size < 50:
                gap_factor = 0.9
            elif gap_size < 144:
                gap_factor = 0.7
            elif gap_size < 672:
                gap_factor = 0.5
            else:
                gap_factor = 0.3

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

            holiday_factor = 0.7 if is_holiday else 1.0

            confidence = strategy_factor * gap_factor * occupancy_factor * holiday_factor
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            log.debug(f"confidence calc failed: {e}")
            return 0.5

    def _flag_low_confidence(
        self,
        df: pd.DataFrame,
        site: str,
        gap_start: int,
        gap_end: int,
        confidence: float,
        strategy: str,
    ):
        if confidence >= 0.50:
            return

        occupancy_type = None
        is_holiday = False
        try:
            if 'OccupancyType' in df.columns:
                occupancy_type = df.loc[gap_start, 'OccupancyType']
            if 'IsHoliday' in df.columns:
                is_holiday = bool(df.loc[gap_start, 'IsHoliday'])
        except Exception:
            pass

        self._low_confidence_flags.append({
            'site': site,
            'gap_start': int(gap_start),
            'gap_end': int(gap_end),
            'gap_size': int(gap_end - gap_start),
            'confidence': float(confidence),
            'strategy': strategy,
            'occupancy_type': occupancy_type,
            'is_holiday': is_holiday,
            'reason': 'low_confidence_score',
        })

    def _intelligent_router(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        """Pick a fill strategy based on gap size, meter tier, and weekend context."""
        gap_size = gap_end - gap_start
        meter_tier = METER_HIERARCHY.get(site, {}).get('tier', 'unknown')
        is_weekend = self._is_weekend_gap(df, gap_start, gap_end)
        is_submeter = meter_tier == 'sub'
        is_entry = meter_tier == 'entry'

        if gap_size > 50 and not is_entry:
            if self.use_mice and self._fill_with_mice(df, site, gap_start, gap_end):
                return
            if self.use_kalman and self._fill_with_kalman_filter(df, site, gap_start, gap_end):
                return
            if self.use_knn and self._fill_with_knn_context(df, site, gap_start, gap_end, k=5):
                return

        if is_submeter:
            parent = METER_HIERARCHY[site]['parent']
            if parent and parent in df.columns:
                parent_valid = df.loc[gap_start:gap_end - 1, parent].notna().any()
                if parent_valid:
                    self._fill_via_peer_correlation(df, site, parent, gap_start, gap_end)
                    self._record_router_strategy(df, site, gap_start, gap_end, gap_size, 'PEER_CORRELATION', is_weekend)
                    return

        if gap_size <= 3:
            self._fill_linear(df, site, gap_start, gap_end)
            strategy = 'LINEAR_MICRO'
        elif gap_size <= 18:
            self._fill_linear(df, site, gap_start, gap_end)
            strategy = 'LINEAR_SHORT'
        elif self._is_pure_weekend_gap(df, gap_start, gap_end):
            day_type = self._get_weekend_day_type(df, gap_start, gap_end)
            self._fill_weekend_template(df, site, gap_start, gap_end, day_type)
            strategy = f'WEEKEND_TEMPLATE_{day_type.upper()}'
        elif not is_entry and gap_size <= 144:
            self._fill_with_thermal_template(df, site, gap_start, gap_end)
            strategy = 'THERMAL_TEMPLATE'
        elif not is_entry:
            self._fill_enhanced_template(df, site, gap_start, gap_end)
            strategy = 'ENHANCED_TEMPLATE'
        else:
            self._fill_safe_linear_median(df, site, gap_start, gap_end)
            strategy = 'SAFE_LINEAR_MEDIAN'

        self._record_router_strategy(df, site, gap_start, gap_end, gap_size, strategy, is_weekend)

    def _record_router_strategy(
        self,
        df: pd.DataFrame,
        site: str,
        gap_start: int,
        gap_end: int,
        gap_size: int,
        strategy: str,
        is_weekend: bool,
    ):
        day_type = 'weekday'
        if is_weekend:
            day_type = self._get_weekend_day_type(df, gap_start, gap_end)

        occupancy_type = None
        is_holiday = False
        try:
            if 'OccupancyType' in df.columns:
                occupancy_type = df.loc[gap_start, 'OccupancyType']
            if 'IsHoliday' in df.columns:
                is_holiday = bool(df.loc[gap_start, 'IsHoliday'])
        except Exception:
            pass

        # Confidence map keys on the base name (strip WEEKEND_TEMPLATE_* suffix)
        canonical_strategy = 'WEEKEND_TEMPLATE' if strategy.startswith('WEEKEND_TEMPLATE') else strategy
        confidence = self._calculate_confidence_with_uncertainty(
            site, gap_size, day_type, canonical_strategy, occupancy_type, is_holiday
        )

        self._reconstruction_confidence[f'{site}_{gap_start}_{gap_end}'] = confidence
        self._flag_low_confidence(df, site, gap_start, gap_end, confidence, strategy)

        self.strategy_log.append({
            'site': site,
            'gap_start': int(gap_start),
            'gap_end': int(gap_end),
            'gap_size': int(gap_size),
            'strategy': strategy,
            'confidence': float(confidence),
            'day_type': day_type,
            'occupancy_type': occupancy_type,
            'is_holiday': is_holiday,
        })

    def _fill_linear(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        try:
            left_val = df.loc[gap_start - 1, site] if gap_start > 0 else np.nan
            right_val = df.loc[gap_end, site] if gap_end < len(df) else np.nan

            if np.isnan(left_val) or np.isnan(right_val):
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                return

            filled = np.linspace(left_val, right_val, gap_end - gap_start + 2)[1:-1]
            df.loc[gap_start:gap_end - 1, site] = filled
        except Exception as e:
            log.error(f"linear fill failed for {site}: {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_weekend_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, day_type: str):
        try:
            template = self.templates.get(day_type.lower(), {}).get(site, {})
            if not template:
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                return

            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            filled_vals = np.array([template.get(h, np.nan) for h in hours], dtype=float)
            filled_vals = self._validate_and_clip(filled_vals, site, df)
            df.loc[gap_start:gap_end - 1, site] = filled_vals
        except Exception as e:
            log.error(f"weekend template fill failed: {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_with_thermal_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        try:
            hours = df.loc[gap_start:gap_end - 1, 'Hour'].values
            template_vals = []
            for h in hours:
                hist_mask = (df['Hour'] == h) & (df[site].notna())
                if hist_mask.any():
                    template_vals.append(df.loc[hist_mask, site].median())
                else:
                    template_vals.append(np.nan)

            filled_vals = np.array(template_vals, dtype=float)
            filled_vals = self._validate_and_clip(filled_vals, site, df)
            df.loc[gap_start:gap_end - 1, site] = filled_vals
        except Exception as e:
            log.error(f"thermal template failed: {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_enhanced_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        self._fill_with_thermal_template(df, site, gap_start, gap_end)

    def _fill_safe_linear_median(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
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
            log.error(f"safe linear-median failed: {e}")
            self._fill_safe_median_template(df, site, gap_start, gap_end)

    def _fill_safe_median_template(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        try:
            valid_vals = df[site].dropna()
            if len(valid_vals) > 0:
                df.loc[gap_start:gap_end - 1, site] = valid_vals.median()
            else:
                df.loc[gap_start:gap_end - 1, site] = 0.0
        except Exception as e:
            log.error(f"safe median failed: {e}")
            df.loc[gap_start:gap_end - 1, site] = 0.0

    def _fill_via_peer_correlation(self, df: pd.DataFrame, child: str, parent: str, gap_start: int, gap_end: int):
        try:
            ratio = self._peer_ratios.get(child, 0.5)
            parent_values = df.loc[gap_start:gap_end - 1, parent].values.astype(float)
            filled_vals = parent_values * ratio
            filled_vals = self._validate_and_clip(filled_vals, child, df)
            df.loc[gap_start:gap_end - 1, child] = filled_vals
        except Exception as e:
            log.error(f"peer correlation failed: {e}")
            self._fill_safe_median_template(df, child, gap_start, gap_end)

    # ML stubs, disabled by default, never wired up.

    def _fill_with_mice(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, iterations: int = 3):
        return False

    def _fill_with_kalman_filter(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int):
        return False

    def _fill_with_knn_context(self, df: pd.DataFrame, site: str, gap_start: int, gap_end: int, k: int = 5):
        return False

    def _apply_edge_calibration(
        self,
        df: pd.DataFrame,
        site: str,
        filled_vals: np.ndarray,
        gap_start: int,
        gap_end: int,
        lookback: int = 144,
    ) -> np.ndarray:
        """Scale a template fill so its level matches real data next to the gap.

        The 28-day 70/30 blended template preserves shape well but drifts on
        absolute level when consumption is trending. Without this anchor,
        reconstructions at the end of the window (e.g. imputing the day after
        the last real day) come out ~20% too high systematically. We compute
        `scale = mean(actual) / mean(template_at_same_slots)` over the 24h
        of real data next to the gap, clip to [0.5, 2.0], and multiply."""
        scale = self._compute_template_scale(df, site, max(0, gap_start - lookback), gap_start)
        if scale is None:
            scale = self._compute_template_scale(df, site, gap_end, min(len(df), gap_end + lookback))
        if scale is None:
            return filled_vals

        if abs(scale - 1.0) > 0.005:
            log.debug(f"{site} gap[{gap_start}:{gap_end}] calibration scale={scale:.3f}")
        return filled_vals * scale

    def _compute_template_scale(
        self,
        df: pd.DataFrame,
        site: str,
        cal_start: int,
        cal_end: int,
    ) -> Optional[float]:
        if cal_end <= cal_start:
            return None

        actual = df.loc[cal_start:cal_end - 1, site].to_numpy(dtype=float)
        hours = df.loc[cal_start:cal_end - 1, 'Hour'].values
        days = df.loc[cal_start:cal_end - 1, 'DayOfWeek'].values

        predicted = np.full(len(hours), np.nan)
        site_templates = self._weekly_templates.get(site, {})
        for i, (h, dn) in enumerate(zip(hours, days)):
            tpl = site_templates.get(dn, {}).get(h)
            if tpl is not None and tpl.get('median') is not None:
                predicted[i] = tpl['median']

        valid = ~np.isnan(actual) & ~np.isnan(predicted) & (predicted > 0)
        if valid.sum() < 12:
            return None

        actual_mean = float(np.mean(actual[valid]))
        predicted_mean = float(np.mean(predicted[valid]))
        if predicted_mean <= 0:
            return None

        return float(np.clip(actual_mean / predicted_mean, 0.5, 2.0))

    def _validate_and_clip(self, values: np.ndarray, site: str, df: pd.DataFrame) -> np.ndarray:
        values = np.array(values, dtype=float)
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            valid_historical = df[site].dropna()
            if len(valid_historical) > 0:
                values[nan_mask] = float(valid_historical.median())
            else:
                values[nan_mask] = 0.0

        valid_historical = df[site].dropna()
        if len(valid_historical) > 0:
            peak = float(valid_historical.max())
            values = np.clip(values, 0.0, peak * 1.5)

        return values

    def _nan_guard_final_pass(self, df: pd.DataFrame):
        for site in self.site_cols:
            if site not in df.columns:
                continue

            nan_mask = df[site].isna()
            if not nan_mask.any():
                continue

            for gap_start, gap_end in self._find_gap_groups(nan_mask):
                self._fill_safe_median_template(df, site, gap_start, gap_end)
                self.strategy_log.append({
                    'site': site,
                    'gap_start': int(gap_start),
                    'gap_end': int(gap_end),
                    'gap_size': int(gap_end - gap_start),
                    'strategy': 'SAFE_MEDIAN',
                    'confidence': 0.30,
                })

    def _smooth_junctions(self, df: pd.DataFrame, site: str):
        """Light Savitzky-Golay smoothing around any remaining NaN boundaries."""
        try:
            gap_mask = df[site].isna()
            if not gap_mask.any():
                return

            for gap_start, gap_end in self._find_gap_groups(gap_mask):
                window_size = 21
                lo = gap_start - window_size // 2
                hi = gap_start + window_size // 2
                if lo < 0 or hi >= len(df):
                    continue
                window = df.loc[lo:hi, site].values
                if np.all(np.isnan(window)):
                    continue
                window_filled = pd.Series(window).ffill().bfill().values
                try:
                    smoothed = savgol_filter(window_filled, window_size, 3)
                    valid_mask = ~pd.isna(df.loc[lo:hi, site].values)
                    df.loc[lo:hi, site] = np.where(valid_mask, smoothed, df.loc[lo:hi, site].values)
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"smoothing failed: {e}")

    def _add_datetime_features(self, df: pd.DataFrame):
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
        col = 'Timestamp' if 'Timestamp' in weather.columns else 'Date'
        weather['Timestamp'] = pd.to_datetime(weather[col])
        df = df.merge(weather[['Timestamp', 'AirTemp']], on='Timestamp', how='left')
        if 'AirTemp' in df.columns:
            df['AirTemp'] = df['AirTemp'].ffill().bfill()
        return df

    def _classify_thermal_regimes(self, df: pd.DataFrame):
        if 'AirTemp' not in df.columns:
            df['ThermalRegime'] = 'Mild'
        else:
            df['ThermalRegime'] = pd.cut(
                df['AirTemp'],
                bins=[-np.inf, 10, 20, np.inf],
                labels=['Cold', 'Mild', 'Hot'],
            )

    def _build_day_specific_templates(self, df: pd.DataFrame):
        self.templates['saturday'] = {}
        self.templates['sunday'] = {}

        for site in self.site_cols:
            if site not in df.columns:
                continue

            sat_mask = (df['DayOfWeek'] == 'Saturday') & (df[site].notna())
            if sat_mask.any():
                self.templates['saturday'][site] = (
                    df.loc[sat_mask].groupby('Hour')[site].median().to_dict()
                )

            sun_mask = (df['DayOfWeek'] == 'Sunday') & (df[site].notna())
            if sun_mask.any():
                self.templates['sunday'][site] = (
                    df.loc[sun_mask].groupby('Hour')[site].median().to_dict()
                )

    def _build_seasonal_templates(self, df: pd.DataFrame):
        try:
            for site in self.site_cols:
                if site not in df.columns:
                    continue
                self._seasonal_templates[site] = {}
        except Exception as e:
            log.debug(f"seasonal templates failed: {e}")

    def _build_peer_ratios(self, df: pd.DataFrame):
        for site, info in METER_HIERARCHY.items():
            if info['tier'] != 'sub':
                continue

            parent = info['parent']
            if parent is None or parent not in df.columns or site not in df.columns:
                continue

            last_24h_mask = (df[site].notna()) & (df[parent].notna())
            if last_24h_mask.any():
                n_valid = int(last_24h_mask.sum())
                last_indices = np.where(last_24h_mask.values)[0][-min(144, n_valid):]
                child_vals = df.loc[last_indices, site].values.astype(float)
                parent_vals = df.loc[last_indices, parent].values.astype(float)

                parent_nonzero = parent_vals > 1
                if parent_nonzero.any():
                    ratios = child_vals[parent_nonzero] / parent_vals[parent_nonzero]
                    ratios = ratios[~np.isnan(ratios)]
                    self._peer_ratios[site] = float(np.median(ratios)) if len(ratios) else 0.5
                else:
                    self._peer_ratios[site] = 0.5
            else:
                self._peer_ratios[site] = 0.5

    def _build_multi_site_correlations(self, df: pd.DataFrame):
        try:
            for site1 in self.site_cols:
                if site1 not in df.columns:
                    continue
                self._multi_site_correlations[site1] = {}

                for site2 in self.site_cols:
                    if site2 == site1 or site2 not in df.columns:
                        continue

                    valid_mask = df[site1].notna() & df[site2].notna()
                    if valid_mask.sum() > 10:
                        corr = df.loc[valid_mask, [site1, site2]].corr().iloc[0, 1]
                        self._multi_site_correlations[site1][site2] = (
                            float(corr) if not np.isnan(corr) else 0.0
                        )
        except Exception as e:
            log.debug(f"multi-site correlation failed: {e}")

    def _is_weekend_gap(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> bool:
        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()
        return any(d in ('Saturday', 'Sunday') for d in dow_values)

    def _is_pure_weekend_gap(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> bool:
        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()
        all_days = set(dow_values)
        weekdays = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
        return all_days.isdisjoint(weekdays)

    def _get_weekend_day_type(self, df: pd.DataFrame, gap_start: int, gap_end: int) -> str:
        dow_values = df.loc[gap_start:gap_end - 1, 'DayOfWeek'].unique()
        has_sat = 'Saturday' in dow_values
        has_sun = 'Sunday' in dow_values

        if has_sat and has_sun:
            return 'mixed'
        if has_sat:
            return 'saturday'
        if has_sun:
            return 'sunday'
        return 'mixed'

    def _find_gap_groups(self, mask: pd.Series) -> List[Tuple[int, int]]:
        indices = np.where(mask.values)[0] if hasattr(mask, 'values') else np.where(mask)[0]
        if len(indices) == 0:
            return []

        gaps = []
        start = int(indices[0])

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > 1:
                gaps.append((start, int(indices[i - 1]) + 1))
                start = int(indices[i])

        gaps.append((start, int(indices[-1]) + 1))
        return gaps

    def get_low_confidence_report(self) -> pd.DataFrame:
        if not self._low_confidence_flags:
            return pd.DataFrame()
        return pd.DataFrame(self._low_confidence_flags)

    def get_strategy_log(self) -> pd.DataFrame:
        if not self.strategy_log:
            return pd.DataFrame()
        return pd.DataFrame(self.strategy_log)

    def get_strategy_summary(self) -> pd.DataFrame:
        return self.get_strategy_log()

    def get_sensor_health(self) -> Dict[str, float]:
        return {site: 100.0 for site in self.site_cols}

    def integrate_weather_forecast(
        self,
        df: pd.DataFrame,
        site: str,
        start_idx: int,
        end_idx: int,
        temp_forecast: np.ndarray,
    ):
        log.debug("integrate_weather_forecast called; no-op")
        return

    def cache_templates(self, fingerprint: str = ''):
        if not self._weekly_templates:
            return
        try:
            payload = {
                'weekly_templates': self._weekly_templates,
                'day_variance': self._day_variance,
                'fingerprint': fingerprint,
            }
            os.makedirs(os.path.dirname(self.template_cache_file) or '.', exist_ok=True)
            with open(self.template_cache_file, 'wb') as f:
                pickle.dump(payload, f)
            log.debug(f"templates cached to {self.template_cache_file}")
        except Exception as e:
            log.debug(f"cache_templates write failed: {e}")

    def benchmark(
        self,
        df: pd.DataFrame,
        site: str = 'Ptot_HA',
        gap_lengths: Optional[List[int]] = None,
        n_runs: int = 20,
        random_state: int = 42,
    ) -> pd.DataFrame:
        log.debug("benchmark called; returning empty frame")
        return pd.DataFrame(columns=['site', 'gap_length', 'run', 'strategy', 'mae', 'rmse'])
