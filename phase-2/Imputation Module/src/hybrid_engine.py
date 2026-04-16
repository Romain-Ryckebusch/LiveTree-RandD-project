
# HYBRID ENGINE (FIXED VERSION)
import hashlib
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.fft import fft
from typing import Dict, Optional, Tuple, List
from datetime import date
import pickle
import os

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

SITE_COLS = ['Ptot_HA', 'Ptot_HEI', 'Ptot_HEI_13RT', 'Ptot_HEI_5RNS', 'Ptot_RIZOMM', 'Ptot_Ilot']

def _default_holidays():
    """Hardcoded fallback list of major 2026 French bank holidays."""
    holidays = set()
    for m, d in [
        (1, 1), (4, 5), (4, 6), (5, 1), (5, 8), (5, 14), (5, 25),
        (7, 14), (8, 15), (11, 1), (11, 11), (12, 25),
    ]:
        holidays.add(date(2026, m, d))
    return holidays, set(), set()

FRANCE_HOLIDAYS_2026, FRANCE_CLOSE_DAYS_2026, FRANCE_SPECIAL_DAYS_2026 = _default_holidays()


class TemperatureAwareHybridEngine:
    """Hybrid gap-filling: temperature + anomaly + peer correlation + confidence bounds + health scoring."""

    def __init__(
        self,
        site_cols: Optional[List[str]] = None,
        weather_df: Optional[pd.DataFrame] = None,
        use_historical_data: bool = True,
        template_cache_file: str = 'templates_cache.pkl',
        low_variance_threshold: float = 5000.0,
    ):
        self.site_cols = site_cols or SITE_COLS
        self.weather_df = weather_df
        self.low_variance_threshold = low_variance_threshold
        self.holidays = FRANCE_HOLIDAYS_2026
        self.close_days = FRANCE_CLOSE_DAYS_2026   # Site-specific closures (near-zero load)
        self.special_days = FRANCE_SPECIAL_DAYS_2026  # Special patterns (bridges, academic)
        self._holidays_array = np.array(sorted(list(FRANCE_HOLIDAYS_2026)), dtype='datetime64[D]')
        self._close_days_array = np.array(sorted(list(self.close_days)), dtype='datetime64[D]')
        self._special_days_array = np.array(sorted(list(self.special_days)), dtype='datetime64[D]')
        self.use_historical_data = use_historical_data
        self.historical_df = None

        self.anomaly_scores: Dict = {}
        self.confidence_bounds: Dict = {}
        self.peer_correlations: Dict = {}
        self.strategy_log: List[Dict] = []
        self._templates: Optional[Dict] = None
        self._rolling: Optional[Dict] = None
        self._template_cache_file = template_cache_file
        self._cache_fingerprint: Optional[str] = None
        self._hm_cache: Dict = {}  # Cache for time-of-day formatting

        if self.use_historical_data:
            try:
                self.historical_df = self._load_expanded_historical_data()
                if self.historical_df is not None:
                    print(f"[OK] Loaded {len(self.historical_df):,} historical records (2021-2025)")
                else:
                    print("[WARN] Historical data not found")
            except Exception as e:
                print(f"[WARN] Error loading historical data: {e}")

        self._load_cached_templates()

    def impute(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Detect gaps, route by size, apply all resilience layers."""
        if weather_df is not None:
            self.weather_df = weather_df

        df = df.copy().sort_values('Timestamp').reset_index(drop=True)
        if self.weather_df is not None:
            df = self._merge_weather(df)

        start = df['Timestamp'].min().floor('10min')
        end = df['Timestamp'].max().ceil('10min')
        df = (
            df.set_index('Timestamp')
            .reindex(pd.date_range(start, end, freq='10min'))
            .reset_index()
        )
        df.columns = ['Timestamp'] + list(df.columns[1:])

        self._fill_airtemp(df)

        self._hm_cache = {}
        if 'Timestamp' in df.columns:
            # pd.DatetimeIndex yields pd.Timestamp; a bare .unique() on a
            # datetime64 Series returns numpy.datetime64 in pandas 1.0.x.
            for ts in pd.DatetimeIndex(df['Timestamp'].unique()):
                hm = f"{ts.hour:02d}:{ts.minute:02d}"
                self._hm_cache[ts] = hm

        self._detect_anomalies(df)
        self._calculate_peer_correlations(df)
        bounds = self._calculate_bounds(df)
        self.strategy_log = []

        for site in self.site_cols:
            if site not in df.columns:
                continue
            gap_mask = df[site].isna()
            if not gap_mask.any():
                continue

            for gap_start, gap_end in self._find_gap_groups(gap_mask):
                self._route_gap_7layer(df, site, gap_start, gap_end, log_strategy=True)

            if site in bounds:
                df[site] = df[site].clip(*bounds[site])

        self._compute_output_confidence_bounds(df)
        out_cols = ['Timestamp'] + self.site_cols + (['AirTemp'] if 'AirTemp' in df.columns else [])
        return df[[c for c in out_cols if c in df.columns]]

    def get_strategy_summary(self) -> pd.DataFrame:
        """Return routing decisions used for each filled gap."""
        return pd.DataFrame(self.strategy_log) if self.strategy_log else pd.DataFrame()

    def get_sensor_health(self) -> Dict[str, float]:
        """Health score per sensor (0–100)."""
        health = {}
        for site in self.site_cols:
            score = 100.0
            if site in self.anomaly_scores:
                scores = self.anomaly_scores[site]
                vals = list(scores.values()) if isinstance(scores, dict) else list(scores)
                anom = sum(1 for s in vals if s < -0.5)
                if vals:
                    score -= (anom / len(vals)) * 30
            if site in self.peer_correlations:
                corrs = list(self.peer_correlations[site].values())
                if corrs and np.mean(corrs) < 0.5:
                    score -= 15
            health[site] = max(0.0, score)
        return health

    def integrate_weather_forecast(
        self,
        df: pd.DataFrame,
        site: str,
        start_idx: int,
        end_idx: int,
        temp_forecast: np.ndarray,
    ):
        """Enhance already-imputed gap using weather forecast data."""
        templates = self._templates or {}
        for i, idx in enumerate(range(start_idx, min(end_idx, len(df)))):
            if i >= len(temp_forecast):
                break
            temp = temp_forecast[i]
            ts = df.loc[idx, 'Timestamp']
            regime, season, day_type = self._classify_conditions(ts, temp)
            hour_min = f"{ts.hour:02d}:{ts.minute:02d}"
            template_val = self._get_template_value(site, regime, season, day_type, hour_min)
            if pd.notna(template_val):
                temp_anomaly = (temp - 15) / 10
                adjusted = template_val * (1 + 0.1 * temp_anomaly)
                current = df.loc[idx, site]
                df.loc[idx, site] = 0.7 * current + 0.3 * adjusted if pd.notna(current) else adjusted

    def cache_templates(self, fingerprint: str = ''):
        """Save current templates + rolling stats to disk."""
        if self._templates:
            try:
                payload = {'templates': self._templates, 'rolling': self._rolling, 'fingerprint': fingerprint}
                with open(self._template_cache_file, 'wb') as f:
                    pickle.dump(payload, f)
                print(f"[OK] Templates cached ({self._template_cache_file})")
            except Exception as e:
                print(f"[WARN] Cache write failed: {e}")

    def benchmark(
        self,
        df: pd.DataFrame,
        site: str = 'Ptot_HA',
        gap_lengths: List[int] = None,
        n_runs: int = 20,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Benchmark hybrid imputation vs simple linear baseline.

        PERF #9: all fixed-cost operations (template build, anomaly detection,
        peer correlations, bounds) are computed ONCE before the run loop instead
        of being repeated inside every impute() call.  A lightweight
        _impute_gap_only() path then fills only the injected gap, skipping the
        full pipeline overhead (grid reindex, weather merge, etc.).
        """
        if gap_lengths is None:
            gap_lengths = [6, 24, 72]
        if site not in df.columns:
            raise ValueError(f"Site '{site}' not found in dataframe")

        df = df.copy().sort_values('Timestamp').reset_index(drop=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        print("[BENCH] Precomputing fixed-cost operations…", flush=True)
        if self.weather_df is not None:
            df = self._merge_weather(df)
        self._fill_airtemp(df)
        precomp_bounds = self._precompute_all(df)
        print("[BENCH] Precomputation done, starting runs.", flush=True)

        rng = np.random.default_rng(random_state)

        def _mae(a, b):  return float(np.mean(np.abs(a - b)))
        def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

        records = []
        valid_mask = df[site].notna().values
        n = len(df)

        for gap_len in gap_lengths:
            if gap_len <= 0 or gap_len + 2 > n:
                continue
            for run in range(n_runs):
                start_idx, tries = None, 0
                while tries < 50 and start_idx is None:
                    candidate = int(rng.integers(1, n - gap_len - 1))
                    end_idx = candidate + gap_len
                    if valid_mask[candidate - 1: end_idx + 1].all():
                        start_idx = candidate
                    tries += 1
                if start_idx is None:
                    continue

                end_idx = start_idx + gap_len
                y_true = df.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)

                df_gap_hybrid = df.copy()
                df_gap_hybrid.loc[start_idx:end_idx - 1, site] = np.nan
                self._impute_gap_only(df_gap_hybrid, site, start_idx, end_idx, precomp_bounds)
                y_hybrid = df_gap_hybrid.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)

                df_gap_baseline = df.copy()
                df_gap_baseline.loc[start_idx:end_idx - 1, site] = np.nan
                df_gap_baseline[site] = df_gap_baseline[site].interpolate(method='linear')
                y_baseline = df_gap_baseline.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)

                for strategy, y_pred in [('hybrid', y_hybrid), ('baseline_linear', y_baseline)]:
                    records.append({
                        'site': site, 'gap_length': gap_len, 'run': run,
                        'strategy': strategy,
                        'mae':  _mae(y_true, y_pred),
                        'rmse': _rmse(y_true, y_pred),
                    })

        return pd.DataFrame.from_records(records)

    def _impute_gap_only(
        self,
        df: pd.DataFrame,
        site: str,
        start_idx: int,
        end_idx: int,
        bounds: Dict[str, Tuple[float, float]],
    ):
        """Lightweight gap-fill used by benchmark(), skips full pipeline overhead.

        Assumes templates, anomaly scores, and peer correlations are already
        populated on self (done once by benchmark before the run loop).
        """
        self._route_gap_7layer(df, site, start_idx, end_idx, log_strategy=False)

        if site in bounds:
            df.loc[start_idx:end_idx - 1, site] = df.loc[start_idx:end_idx - 1, site].clip(*bounds[site])

    def _load_expanded_historical_data(self) -> Optional[pd.DataFrame]:
        """Load all available historical data (2021-2025) with unified column mapping and AirTemp."""
        dfs = []
        col_map = {
            'HA': 'Ptot_HA',
            'HEI1': 'Ptot_HEI',
            'HEI2': 'Ptot_HEI_13RT',
            'RIZOMM': 'Ptot_RIZOMM',
            'Campus': 'Ptot_Ilot',
            'AirTemp': 'AirTemp',
        }
        out_cols = ['Timestamp', 'Ptot_HA', 'Ptot_HEI', 'Ptot_HEI_13RT', 'Ptot_HEI_5RNS', 'Ptot_RIZOMM', 'Ptot_Ilot', 'AirTemp']

        def _load_year(path: str, label: str, add_5rns_nan: bool, force_airtemp_nan: bool = False):
            try:
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.rename(columns=col_map)
                df['Timestamp'] = df['Date']
                if add_5rns_nan:
                    df['Ptot_HEI_5RNS'] = np.nan
                if force_airtemp_nan:
                    df['AirTemp'] = np.nan
                dfs.append(df[out_cols])
                print(f"[OK] Loaded {label}: {len(df):,} records")
            except Exception as e:
                print(f"[WARN] {label} data load failed: {e}")

        _load_year('data/Consumption_2021_Power.csv', '2021', add_5rns_nan=True)
        _load_year('data/Consumption_2022_Power.csv', '2022', add_5rns_nan=True)
        _load_year('data/2023 -2025.csv', '2023-2025', add_5rns_nan=False, force_airtemp_nan=True)
        
        if not dfs:
            print("[ERROR] No historical data loaded")
            return None
        
        # Merge all dataframes
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('Timestamp').reset_index(drop=True)
        
        self._fill_airtemp(combined)
        
        date_range = f"{combined['Timestamp'].min()} to {combined['Timestamp'].max()}"
        print(f"[OK] Combined historical data: {len(combined):,} total records ({date_range})")
        return combined

    def _linear_interp_weights(self, gap_size: int) -> np.ndarray:
        """Helper: compute linear interpolation weights (REDUNDANCY #5).
        
        Returns normalized weights from 0 to 1 for gap of size gap_size.
        """
        return np.arange(1, gap_size + 1) / (gap_size + 1)

    def _get_template_value(self, site: str, regime: str, season: str, day_type: str, hour_min: str) -> float:
        """Consolidate nested template dictionary access with 4-D key.
        
        Key structure: templates[site][regime][season][day_type][hour_min]
        Returns np.nan if key not found.
        """
        templates = self._templates or {}
        return templates.get(site, {}).get(regime, {}).get(season, {}).get(day_type, {}).get(hour_min, np.nan)

    def _fill_airtemp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper: consolidate AirTemp forward-fill logic (REDUNDANCY #1)."""
        if 'AirTemp' in df.columns:
            df['AirTemp'] = df['AirTemp'].ffill().bfill()
        return df

    def _precompute_all(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Helper: consolidate precomputation sequence (REDUNDANCY #2).
        
        Used by both impute() and benchmark() paths.
        Returns bounds dict for final clipping.
        """
        self._build_templates_if_needed(df)
        self._detect_anomalies(df)
        self._calculate_peer_correlations(df)
        return self._calculate_bounds(df)

    def _load_cached_templates(self):
        if os.path.exists(self._template_cache_file):
            try:
                with open(self._template_cache_file, 'rb') as f:
                    payload = pickle.load(f)
                self._templates = payload.get('templates')
                self._rolling = payload.get('rolling')
                self._cache_fingerprint = payload.get('fingerprint', '')
                print(f"[OK] Loaded cached templates")
            except Exception as e:
                print(f"[WARN] Failed to load template cache: {e}")

    def _data_fingerprint(self, df: pd.DataFrame) -> str:
        """Hash of (shape, date range, columns) to detect stale cache."""
        key = f"{df.shape}_{df['Timestamp'].min()}_{df['Timestamp'].max()}_{sorted(df.columns.tolist())}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _merge_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data (AirTemp) from external source.
        
        If weather_df is provided, merge it. Otherwise, try to load from file.
        Forward-fill to handle any remaining NaN values.
        """
        if self.weather_df is None:
            # Try to load weather data from file
            try:
                self.weather_df = pd.read_csv('data/2026 weather data.csv')
                print(f"[OK] Loaded weather data: {len(self.weather_df):,} records")
            except Exception:
                print("[WARN] No weather data file found, proceeding without external temperature")
                return df
        
        weather = self.weather_df.copy()
        col = 'Date' if 'Date' in weather.columns else 'Timestamp'
        weather['Timestamp'] = pd.to_datetime(weather[col])
        
        # Merge and fill AirTemp gaps
        result = df.merge(weather[['Timestamp', 'AirTemp']], on='Timestamp', how='left')
        
        # Forward-fill missing AirTemp values
        self._fill_airtemp(result)
        
        return result

    def _classify_day_type(self, ts: pd.Timestamp) -> str:
        """Classify day into 7 categories based on calendar."""
        d = ts.date()
        if d in self.close_days:
            return 'close'          # Site-specific closures (near-zero load)
        if d in self.holidays:
            return 'holiday'        # Bank holidays (reduced load)
        if d in self.special_days:
            return 'special'        # Bridges, academic calendar, etc.
        weekday = ts.weekday()
        if weekday == 4:
            return 'friday'         # Friday (often transition day)
        if weekday == 5:
            return 'saturday'       # Saturday (bell-curve pattern, higher load)
        if weekday == 6:
            return 'sunday'         # Sunday (flat pattern, lower load)
        return 'weekday'            # Monday-Thursday

    def _classify_season(self, ts: pd.Timestamp) -> str:
        """Classify season based on month."""
        m = ts.month
        if m in (12, 1, 2):
            return 'winter'
        if m in (3, 4, 5):
            return 'spring'
        if m in (6, 7, 8):
            return 'summer'
        return 'autumn'  # (9, 10, 11)

    def _classify_conditions(self, ts: pd.Timestamp, temp: float) -> Tuple[str, str, str]:
        """Classify temperature regime, season, and day type.
        
        Returns: (regime, season, day_type)
        - regime: 'cold' (<5°C), 'mild', 'hot' (>20°C)
        - season: 'winter', 'spring', 'summer', 'autumn'
        - day_type: 'weekday', 'friday', 'saturday', 'sunday', 'holiday', 'close', 'special'
        """
        # Temperature regime
        if not np.isnan(temp):
            regime = 'cold' if temp < 5 else ('hot' if temp > 20 else 'mild')
        else:
            regime = 'mild'
        
        # Season and day type
        season = self._classify_season(ts)
        day_type = self._classify_day_type(ts)
        
        return regime, season, day_type

    def _fallback_template(self, site: str, regime: str, season: str, day_type: str) -> Dict:
        """Graceful fallback for rare (regime, season, day_type) combinations.
        
        Fallback hierarchy:
        1. Exact (regime, season, day_type)
        2. Same regime, any season
        3. 'mild' regime, any season
        4. Generic weekday fallback
        """
        t = self._templates.get(site, {})
        
        # Try exact match
        v = t.get(regime, {}).get(season, {}).get(day_type)
        if v:
            return v
        
        # Try same regime, any season
        for seas in ['winter', 'spring', 'summer', 'autumn']:
            v = t.get(regime, {}).get(seas, {}).get(day_type)
            if v:
                return v
        
        # Try 'mild' regime, any season
        for seas in ['winter', 'spring', 'summer', 'autumn']:
            v = t.get('mild', {}).get(seas, {}).get(day_type)
            if v:
                return v
        
        # Last resort: mild, winter, weekday
        return t.get('mild', {}).get('winter', {}).get('weekday', {})

    def _find_gap_groups(self, mask: pd.Series) -> List[Tuple[int, int]]:
        gaps = np.where(mask.values)[0]
        if len(gaps) == 0:
            return []
        
        breaks = np.where(np.diff(gaps) > 1)[0] + 1
        groups = []
        prev = 0
        
        for break_idx in breaks:
            groups.append((int(gaps[prev]), int(gaps[break_idx - 1]) + 1))
            prev = break_idx
        
        groups.append((int(gaps[prev]), int(gaps[-1]) + 1))
        return groups

    def _fill_linear_with_temp(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int):
        if start_idx > 0 and end_idx < len(df):
            before = df.loc[start_idx - 1, site]
            after = df.loc[end_idx, site]
            if pd.isna(before) or pd.isna(after):
                return
            gap_size = end_idx - start_idx
            t = self._linear_interp_weights(gap_size)
            df.loc[start_idx:end_idx - 1, site] = before + t * (after - before)

    def _fill_pilotage_temp_aware(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, gap_size: int):
        self._build_templates_if_needed(df)

        if gap_size in [60, 72] and start_idx > 72 and end_idx < len(df) - 72:
            before = df.loc[max(0, start_idx - 72):start_idx - 1, site].dropna()
            after = df.loc[end_idx:min(len(df) - 1, end_idx + 72), site].dropna()
            if len(before) > 0 and len(after) > 0:
                cv = ((before.std() / (before.mean() + 1e-6)) + (after.std() / (after.mean() + 1e-6))) / 2
                if cv < 0.15:
                    bv, av = df.loc[start_idx - 1, site], df.loc[end_idx, site]
                    if not pd.isna(bv) and not pd.isna(av):
                            t = self._linear_interp_weights(gap_size)
                            df.loc[start_idx:start_idx + gap_size - 1, site] = bv + t * (av - bv)
                            return

        if gap_size > 72:
            self._fill_long_gap(df, site, start_idx, end_idx, gap_size)
        else:
            self._fill_pilotage_standard(df, site, start_idx, end_idx, gap_size)

    def _fill_long_gap(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, gap_size: int):
        self._fill_ensemble(df, site, start_idx, end_idx, gap_size)
        norm_info = self._normalize_long_gap_amplitude(df, site, start_idx, end_idx)
        self._apply_adaptive_smoothing(df, site, start_idx, end_idx)
        align_info = self._align_to_edge_levels(df, site, start_idx, end_idx)
        self._last_postproc = {'norm': norm_info, 'align': align_info}

    def _route_gap_7layer(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, log_strategy: bool = True):
        """Explicit 7-layer pilotage router.

        Layers use progressively heavier methods as gap duration grows.
        """
        gap_size = end_idx - start_idx
        strategy = 'UNKNOWN'
        confidence = 0.5

        if gap_size <= 3:
            self._fill_linear_with_temp(df, site, start_idx, end_idx)
            strategy, confidence = 'L1_LINEAR_MICRO', 0.98

        elif gap_size <= 18:
            self._fill_linear_with_temp(df, site, start_idx, end_idx)
            strategy, confidence = 'L2_LINEAR_TEMP', 0.95

        elif gap_size <= 24:
            if self._try_spline_fill(df, site, start_idx, end_idx):
                strategy, confidence = 'L3_SPLINE', 0.92
            else:
                self._fill_pilotage_standard(df, site, start_idx, end_idx, gap_size)
                strategy, confidence = 'L3_PILOTAGE_FALLBACK', 0.86

        elif gap_size <= 72:
            if self._anomaly_near_gap(site, start_idx, end_idx):
                self._fill_pilotage_standard(df, site, start_idx, end_idx, gap_size)
                strategy, confidence = 'L4_ANOMALY_GUARDED', 0.82
            else:
                self._fill_pilotage_temp_aware(df, site, start_idx, end_idx, gap_size)
                strategy, confidence = 'L4_TEMP_AWARE', 0.88

        elif gap_size <= 144:
            self._fill_pilotage_temp_aware(df, site, start_idx, end_idx, gap_size)
            strategy, confidence = 'L5_DAY_SCALE_PILOTAGE', 0.8

        elif gap_size <= 432:
            self._fill_long_gap(df, site, start_idx, end_idx, gap_size)
            strategy, confidence = 'L6_MULTI_DAY_ENSEMBLE', 0.72

        else:
            self._fill_long_gap(df, site, start_idx, end_idx, gap_size)
            strategy, confidence = 'L7_VERY_LONG_ENSEMBLE', 0.65

        fallback_strategy = self._validate_and_fallback_recovery(df, site, start_idx, end_idx, strategy)
        if fallback_strategy != strategy:
            strategy = fallback_strategy
            confidence *= 0.8

        if log_strategy:
            entry = {
                'site': site,
                'gap_start': int(start_idx),
                'gap_end': int(end_idx),
                'gap_size': int(gap_size),
                'strategy': strategy,
                'confidence': float(confidence),
            }
            pp = getattr(self, '_last_postproc', None)
            if pp is not None:
                entry['postproc'] = pp
                self._last_postproc = None
            self.strategy_log.append(entry)

    def _try_spline_fill(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> bool:
        vals = self._method_spline(df, site, start_idx, end_idx)
        if len(vals) == 0 or np.all(np.isnan(vals)):
            return False
        if np.isnan(vals).any():
            good = np.where(~np.isnan(vals))[0]
            if len(good) == 0:
                return False
            vals = np.interp(np.arange(len(vals)), good, vals[good])
        df.loc[start_idx:end_idx - 1, site] = vals
        return True

    def _anomaly_near_gap(self, site: str, start_idx: int, end_idx: int, margin: int = 24) -> bool:
        scores = self.anomaly_scores.get(site)
        if scores is None or len(scores) == 0:
            return False
        left = max(0, start_idx - margin)
        right = end_idx + margin
        try:
            local = scores[(scores.index >= left) & (scores.index < right)]
            if len(local) == 0:
                return False
            return bool((local < -0.5).any())
        except Exception:
            return False

    def _validate_and_fallback_recovery(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, current_strategy: str) -> str:
        """
        VALIDATION LAYER: Check if recovered values match surrounding context.
        If deviation > 40%, fallback to MEDIAN to prevent anomalous reconstructions.
        
        This prevents silent failures like the Feb 9-12 anomaly (76% error).
        """
        try:
            # Get recovered values
            recovered_vals = df.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)
            if np.all(np.isnan(recovered_vals)):
                return current_strategy
            
            recovered_valid = recovered_vals[~np.isnan(recovered_vals)]
            if len(recovered_valid) == 0:
                return current_strategy
            
            recovered_mean = np.nanmean(recovered_vals)
            
            # Get context: 24 hours before and after gap
            context_margin = 144  # ~1 day in 10-min intervals
            before_start = max(0, start_idx - context_margin)
            after_end = min(len(df), end_idx + context_margin)
            
            before_vals = df.loc[before_start:start_idx - 1, site].to_numpy(dtype=float)
            after_vals = df.loc[end_idx:after_end, site].to_numpy(dtype=float)
            
            before_valid = before_vals[~np.isnan(before_vals)]
            after_valid = after_vals[~np.isnan(after_vals)]
            
            if len(before_valid) == 0 or len(after_valid) == 0:
                return current_strategy
            
            before_mean = np.nanmean(before_valid)
            after_mean = np.nanmean(after_valid)
            context_mean = (before_mean + after_mean) / 2
            
            # Compute deviation
            if context_mean > 0:
                deviation = abs(recovered_mean - context_mean) / context_mean
            else:
                return current_strategy
            
            # If deviation > 40%, use MEDIAN fallback
            if deviation > 0.40:
                # MEDIAN fallback: average of before/after context
                df.loc[start_idx:end_idx - 1, site] = context_mean
                return f"{current_strategy}_FALLBACK_MEDIAN_{deviation:.1%}"
            
            return current_strategy
            
        except Exception as e:
            # On any error, keep original strategy
            return current_strategy

    def _fill_pilotage_standard(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, gap_size: int):
        """Template + rolling + peer-correlation weighted fill for short-medium gaps.
        FIX #3: templates guaranteed built before entry.
        FIX #9: peer weights driven by actual correlation coefficient.
        """
        rolling = self._rolling or {}

        w_t, w_r, w_p_base = (0.2, 0.7, 0.1) if gap_size >= 144 else (0.5, 0.3, 0.2)

        # Pre-sort peers by correlation strength (FIX #9)
        sorted_peers = sorted(
            self.peer_correlations.get(site, {}).items(),
            key=lambda kv: -abs(kv[1]),
        )[:3]

        # Vectorize the loop where possible (PERF #12)
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.loc[idx]
            ts = row['Timestamp']
            air_temp = row.get('AirTemp', np.nan)
            regime, season, day_type = self._classify_conditions(ts, air_temp)
            # Use cached format or compute once per timestamp
            hour_min = self._hm_cache.get(ts)
            if hour_min is None:
                hour_min = f"{ts.hour:02d}:{ts.minute:02d}"
                self._hm_cache[ts] = hour_min

            is_extreme = not np.isnan(air_temp) and (air_temp < 5.0 or air_temp > 25.0)
            wt = w_t * 1.3 if is_extreme else w_t

            sources, weights = [], []

            tv = self._get_template_value(site, regime, season, day_type, hour_min)
            if not np.isnan(tv):
                sources.append(tv)
                weights.append(wt)

            rv = rolling.get(site, {}).get(hour_min, np.nan)
            if not np.isnan(rv):
                sources.append(rv)
                weights.append(w_r)

            # Correlation-weighted peers (FIX #9)
            for peer, corr in sorted_peers:
                if peer in df.columns and not pd.isna(df.loc[idx, peer]):
                    sources.append(df.loc[idx, peer])
                    weights.append(w_p_base * abs(corr))

            if sources:
                w = np.array(weights)
                df.loc[idx, site] = np.average(sources, weights=w / w.sum())

    def _fill_close_day(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int):
        """Fill close days (near-zero load) using median of historical close-day profiles.
        
        Close days have fundamentally different patterns than holidays:
        - Holiday: reduced but structured (heating/AC off, minimal misc load)
        - Close: near-zero or minimal (complete site closure)
        
        This method anchors the close-day template to observed edge values to handle
        partial closures or operational variations.
        """
        templates = self._templates or {}
        ts = df.loc[start_idx, 'Timestamp']
        temp = df.loc[start_idx, 'AirTemp'] if 'AirTemp' in df.columns else np.nan
        regime, season, _ = self._classify_conditions(ts, temp)

        site_tpl = templates.get(site, {})
        close_profile = site_tpl.get(regime, {}).get(season, {}).get('close', {})

        if not close_profile:
            close_profile = self._fallback_template(site, regime, season, 'close')

        # Extract time-of-day values and substitute from template
        rows = df.loc[start_idx:end_idx-1, 'Timestamp']
        hm_arr = rows.dt.strftime('%H:%M').to_numpy()
        vals = np.array([close_profile.get(hm, np.nan) for hm in hm_arr], dtype=float)

        # Anchor to observed edges if available (handle scaling variations)
        if start_idx > 0 and end_idx < len(df):
            before = df.loc[start_idx-1, site]
            after = df.loc[end_idx, site]
            if pd.notna(before) and pd.notna(vals[0]) and vals[0] > 0:
                scale = before / vals[0]
                vals *= np.clip(scale, 0.5, 2.0)

        df.loc[start_idx:end_idx-1, site] = vals

    def _fill_ensemble(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, gap_size: int):
        """Weighted ensemble of template, spline, fourier (+ similar-day for long gaps).

        Long outages tend to drift when only local methods are blended, so we
        add a similar-day component for gap_size > 72 to preserve diurnal shape.
        
        SMOOTHNESS DETECTION: For very flat days (std dev < 5000 kW), reduces template
        weight and increases spline weight to prevent false structure injection.
        """
        gap_len = end_idx - start_idx

        method_results = {}
        for name, fn in [
            ('template', self._method_template_multi_scale),
            ('spline', self._method_spline),
            ('fourier', self._method_fourier),
        ]:
            try:
                method_results[name] = fn(df, site, start_idx, end_idx)
            except Exception:
                method_results[name] = np.full(gap_len, np.nan)

        if gap_size > 72:
            try:
                method_results['similar_day'] = self._method_similar_day_profile(df, site, start_idx, end_idx)
            except Exception:
                method_results['similar_day'] = np.full(gap_len, np.nan)

        # SMOOTHNESS DETECTION: Check if gap pattern is unusually flat
        # This helps detect Sundays or other flat patterns where template shouldn't dominate
        # Use similar-day method (which shows ACTUAL historical days) to detect smoothness
        is_smooth_day = False
        if gap_size > 72 and not np.all(np.isnan(method_results.get('similar_day', [np.nan]))):
            # Check the similar-day method's output (actual historical pattern)
            similar_vals = method_results['similar_day'][~np.isnan(method_results['similar_day'])]
            if len(similar_vals) > 10:
                hourly_similar = []
                for h in range(min(24, gap_len // 6)):
                    h_start = h * 6
                    h_end = min((h + 1) * 6, len(similar_vals))
                    if h_end > h_start:
                        hourly_similar.append(np.mean(similar_vals[h_start:h_end]))
                
                if len(hourly_similar) >= 8:
                    hourly_std = np.std(hourly_similar)
                    is_smooth_day = hourly_std < self.low_variance_threshold

        # Base confidences (FIX #11: percentile bounds inside _assess_method_confidence)
        confidences = {
            name: self._assess_method_confidence(df, site, vals)
            for name, vals in method_results.items()
        }

        # SMOOTHNESS ADJUSTMENT: For very flat days, heavily favor spline over template
        if is_smooth_day:
            # On smooth days, spline (edge-anchored) is much better than template
            # Template often mis-predicts shape for weekly anomalies (flat Sundays, etc.)
            confidences['template'] *= 0.15   # Heavily reduce template weight
            confidences['spline'] *= 4.0      # Strongly boost spline weight
            confidences['fourier'] *= 0.7     # Slightly reduce fourier (less needed for flat days)

        # FIX #10: temperature regime modifier
        if 'AirTemp' in df.columns:
            temp = df.loc[start_idx, 'AirTemp'] if start_idx < len(df) else np.nan
            if not np.isnan(temp) and (temp < 5 or temp > 25):
                confidences['template'] *= 1.3
                confidences['spline'] *= 0.8

        if 'similar_day' in confidences:
            # Prefer pattern-based fill on long gaps where local interpolation drifts.
            confidences['similar_day'] *= 1.4

        total_conf = sum(confidences.values())
        if total_conf == 0:
            return

        # FIX #8: position-aware weights
        # position 0 = gap start, 1 = gap end; spline is strongest at edges
        position = np.linspace(0.0, 1.0, gap_len)
        edge_proximity = 1.0 - 2.0 * np.abs(position - 0.5)   # 1 at edges, 0 at centre

        ensemble_sum = np.zeros(gap_len)
        ensemble_wsum = np.zeros(gap_len)

        for name, vals in method_results.items():
            if np.all(np.isnan(vals)):
                continue
            base_w = confidences[name] / total_conf

            if name == 'spline':
                pos_w = (1.0 + edge_proximity) * base_w      # up to 2× at edges
            elif name == 'template':
                pos_w = (1.0 + (1.0 - edge_proximity)) * base_w  # up to 2× at centre
            elif name == 'similar_day':
                # Similar-day should dominate gap centre but not the edges.
                pos_w = (1.0 + 1.5 * (1.0 - edge_proximity)) * base_w
            else:
                pos_w = np.full(gap_len, base_w)

            valid = ~np.isnan(vals)
            ensemble_sum[valid] += pos_w[valid] * vals[valid]
            ensemble_wsum[valid] += pos_w[valid]

        ensemble_values = np.divide(
            ensemble_sum, ensemble_wsum,
            out=np.full(gap_len, np.nan),
            where=ensemble_wsum > 0,
        )

        # Interpolate any remaining NaN positions
        nan_mask = np.isnan(ensemble_values)
        if np.any(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                ensemble_values[nan_mask] = np.interp(
                    np.where(nan_mask)[0], valid_idx, ensemble_values[valid_idx]
                )
            elif len(valid_idx) == 1:
                ensemble_values[nan_mask] = ensemble_values[valid_idx[0]]
            else:
                ensemble_values[:] = np.nanmean(df[site])

        df.loc[start_idx:end_idx - 1, site] = ensemble_values

    def _method_similar_day_profile(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        """Build a long-gap profile from nearby days with similar edge levels.

        Uses same time-of-day slices from previous days, scores candidates by
        edge consistency, and blends top candidates by inverse error.
        BUGFIX: Only uses candidates with same day_type (weekday/weekend/holiday).
        """
        gap_len = end_idx - start_idx
        if gap_len <= 0:
            return np.array([])

        raw = df[site].to_numpy(dtype=float)
        if start_idx <= 0 or end_idx >= len(raw):
            return np.full(gap_len, np.nan)

        before_edge = raw[start_idx - 1]
        after_edge = raw[end_idx] if end_idx < len(raw) else np.nan
        if np.isnan(before_edge) or np.isnan(after_edge):
            return np.full(gap_len, np.nan)

        if 'Timestamp' not in df.columns:
            return np.full(gap_len, np.nan)

        target_day_type = self._classify_day_type(df.loc[start_idx, 'Timestamp'])

        candidates = []
        scores = []

        for d in range(1, 22):
            s = start_idx - d * 144
            e = s + gap_len
            if s <= 0 or e >= len(raw):
                continue
            seg = raw[s:e]
            if np.isnan(seg).any():
                continue

            seg_before = raw[s - 1]
            seg_after = raw[e] if e < len(raw) else np.nan
            if np.isnan(seg_before) or np.isnan(seg_after):
                continue

            candidate_day_type = self._classify_day_type(df.loc[s, 'Timestamp'])
            
            if candidate_day_type != target_day_type:
                continue

            edge_err = abs(seg_before - before_edge) + abs(seg_after - after_edge)
            level_err = abs(np.mean(seg) - 0.5 * (before_edge + after_edge))
            score = edge_err + 0.25 * level_err

            candidates.append(seg)
            scores.append(score)

        if len(candidates) == 0:
            return np.full(gap_len, np.nan)

        order = np.argsort(scores)[:5]
        top = np.array([candidates[i] for i in order])
        top_scores = np.array([scores[i] for i in order], dtype=float)
        weights = 1.0 / (top_scores + 1e-6)
        weights = weights / weights.sum()
        blended = np.average(top, axis=0, weights=weights)

        start_offset = before_edge - blended[0]
        end_offset = after_edge - blended[-1]
        ramp = np.linspace(start_offset, end_offset, gap_len)
        return blended + ramp

    def _align_to_edge_levels(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, lookback: int = 36, divergence_context: int = 288):
        """Nudge long-gap fill to match edge means and trends.

        Caps the per-edge offset at half the donor's std so the linspace ramp
        cannot shift the whole gap. When the wider context is divergent, the
        slope-blend weight drops to 0, slopes from a flat side and a
        high-cycle side are not comparable.

        Returns a dict describing the decision for the strategy_log.
        """
        filled = df.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)
        if len(filled) == 0 or np.isnan(filled).all():
            return None

        before = df.loc[max(0, start_idx - lookback):start_idx - 1, site].dropna()
        after = df.loc[end_idx:min(len(df) - 1, end_idx + lookback), site].dropna()
        if len(before) == 0 or len(after) == 0:
            return None

        # Reuse the same wide window as _normalize_long_gap_amplitude so the
        # two stages agree on what "divergent" means.
        wb = df.loc[max(0, start_idx - divergence_context):start_idx - 1, site].dropna().to_numpy(dtype=float)
        wa = df.loc[end_idx:min(len(df) - 1, end_idx + divergence_context), site].dropna().to_numpy(dtype=float)
        if len(wb) > 0 and len(wa) > 0:
            div = self._context_divergence(wb, wa)
        else:
            div = {'is_divergent': False}

        raw_start = float(before.mean() - filled[0])
        raw_end = float(after.mean() - filled[-1])

        cap = 0.5 * float(np.std(filled))
        if cap > 0:
            start_offset = float(np.clip(raw_start, -cap, cap))
            end_offset = float(np.clip(raw_end, -cap, cap))
        else:
            start_offset, end_offset = raw_start, raw_end

        if len(before) >= 4:
            before_times = np.arange(len(before))
            before_slope = np.polyfit(before_times[-4:], before.values[-4:], 1)[0]
        else:
            before_slope = 0.0

        if len(after) >= 4:
            after_times = np.arange(len(after))
            after_slope = np.polyfit(after_times[:4], after.values[:4], 1)[0]
        else:
            after_slope = 0.0

        gap_pos = np.linspace(0.0, 1.0, len(filled))

        level_ramp = start_offset + (end_offset - start_offset) * gap_pos

        filled_times = np.arange(len(filled))
        slope_ramp = before_slope + (after_slope - before_slope) * gap_pos

        adjusted = filled + level_ramp

        slope_weight = 0.0 if div.get('is_divergent') else 0.2
        if len(filled) > 1 and slope_weight > 0:
            # NOTE: original code wrote `filled[0]` here, conflating the value
            # of the first filled point with the time index zero. With filled[0]
            # ~ 100k W, the correction blew up to ±10⁵ and made the engine
            # produce wildly extrapolated values that the >40% validation
            # fallback then collapsed to a flat constant. Use the time index.
            quad_correction = (
                (before_slope * (1.0 - gap_pos) + after_slope * gap_pos)
                * (filled_times - filled_times[0])
                / (len(filled) - 1)
            )
            adjusted = adjusted + slope_weight * quad_correction

        df.loc[start_idx:end_idx - 1, site] = adjusted
        capped = (raw_start != start_offset) or (raw_end != end_offset)
        return {
            'mode': 'align_capped' if capped else 'align_normal',
            'cap': float(cap),
            'start_offset_raw': float(raw_start),
            'start_offset_applied': float(start_offset),
            'end_offset_raw': float(raw_end),
            'end_offset_applied': float(end_offset),
            'slope_weight': float(slope_weight),
        }


    def _context_divergence(self, before: np.ndarray, after: np.ndarray) -> dict:
        # A divergent context means averaging the two sides erases the donor's
        # natural cycle, see plan note about Easter (flat-low) vs working week.
        if len(before) == 0 or len(after) == 0:
            return {'mean_ratio': 1.0, 'std_ratio': 1.0, 'is_divergent': False}
        eps = 1e-6
        b_mean, a_mean = float(np.mean(before)), float(np.mean(after))
        b_std, a_std = float(np.std(before)), float(np.std(after))
        mean_ratio = max(b_mean, a_mean) / max(min(b_mean, a_mean), eps)
        std_ratio = max(b_std, a_std) / max(min(b_std, a_std), eps)
        is_divergent = (std_ratio > 3.0) or (mean_ratio > 1.4)
        return {
            'mean_ratio': float(mean_ratio),
            'std_ratio': float(std_ratio),
            'is_divergent': bool(is_divergent),
        }

    def _normalize_long_gap_amplitude(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, context: int = 288):
        """Affine-normalize long-gap fill to nearby observed mean/std.

        Returns a dict describing the decision (mode, ratios, scale) for the
        strategy_log.
        """
        filled = df.loc[start_idx:end_idx - 1, site].to_numpy(dtype=float)
        if len(filled) == 0 or np.isnan(filled).all():
            return None

        before = df.loc[max(0, start_idx - context):start_idx - 1, site].dropna().to_numpy(dtype=float)
        after = df.loc[end_idx:min(len(df) - 1, end_idx + context), site].dropna().to_numpy(dtype=float)
        if len(before) == 0 or len(after) == 0:
            return None

        div = self._context_divergence(before, after)
        # Keep the symmetric average even when the contexts diverge: the
        # edge-alignment cap in _align_to_edge_levels is what bounds the
        # over/under-shoot, not the picker. The divergence flag is still
        # logged so the mode can be read back from the strategy log.
        target_mean = 0.5 * (float(np.mean(before)) + float(np.mean(after)))
        target_std = 0.5 * (float(np.std(before)) + float(np.std(after)))
        mode = 'norm_divergent_symmetric' if div['is_divergent'] else 'norm_symmetric'

        cur_mean = float(np.mean(filled))
        cur_std = float(np.std(filled))
        if cur_std < 1e-6:
            adjusted = filled - cur_mean + target_mean
            scale_applied = 1.0
        else:
            scale = target_std / cur_std if target_std > 1e-6 else 1.0
            scale = float(np.clip(scale, 0.6, 1.8))
            adjusted = (filled - cur_mean) * scale + target_mean
            scale_applied = scale

        df.loc[start_idx:end_idx - 1, site] = adjusted
        return {
            'mode': mode,
            'std_ratio': float(div['std_ratio']),
            'mean_ratio': float(div['mean_ratio']),
            'scale_applied': float(scale_applied),
        }

    def _method_template_multi_scale(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        trend = self._extract_trend(df, site, start_idx, end_idx)
        seasonal = self._extract_seasonal(df, site, start_idx, end_idx, period=144)
        weekly = self._extract_seasonal(df, site, start_idx, end_idx, period=1008)
        yearly = self._extract_yearly_seasonal(df, site, start_idx, end_idx)
        return 0.3 * trend + 0.3 * seasonal + 0.2 * weekly + 0.2 * yearly

    def _extract_trend(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        before = df.loc[max(0, start_idx - 48):start_idx - 1, site].dropna()
        after = df.loc[end_idx:min(len(df) - 1, end_idx + 48), site].dropna()
        if len(before) > 0 and len(after) > 0:
            return np.linspace(before.iloc[-1], after.iloc[0], end_idx - start_idx)
        bv = df.loc[start_idx - 1, site] if start_idx > 0 else np.nan
        av = df.loc[end_idx, site] if end_idx < len(df) else np.nan
        if pd.isna(bv) or pd.isna(av):
            return np.full(end_idx - start_idx, np.nanmean(df[site]))
        return np.linspace(bv, av, end_idx - start_idx)

    def _extract_seasonal(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int, period: int) -> np.ndarray:
        """PERF #5: fully vectorised 2-D numpy gather, zero Python loop over gap positions.

        Builds a (gap_len, 8) index matrix of all lookback positions at once, gathers
        values in a single fancy-index op, then takes nanmedian along axis=1.
        """
        raw = df[site].to_numpy(dtype=float)
        gap_len = end_idx - start_idx

        gap_positions = np.arange(start_idx, end_idx)                    # (gap_len,)
        offsets = np.arange(1, 9) * period                               # (8,)
        candidate_idx = gap_positions[:, None] - offsets[None, :]        # (gap_len, 8)

        valid_mask = (candidate_idx >= 0) & (candidate_idx < start_idx)

        safe_idx = np.clip(candidate_idx, 0, len(raw) - 1)
        vals = raw[safe_idx]                                              # (gap_len, 8)
        vals[~valid_mask] = np.nan

        with np.errstate(all='ignore'):
            seasonal = np.nanmedian(vals, axis=1)                        # (gap_len,)

        nan_mask = np.isnan(seasonal)
        if np.any(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                seasonal[nan_mask] = np.interp(
                    np.where(nan_mask)[0], valid_idx, seasonal[valid_idx]
                )
            elif len(valid_idx) == 1:
                seasonal[nan_mask] = seasonal[valid_idx[0]]
            else:
                seasonal[:] = np.nanmean(raw)

        return seasonal

    def _extract_yearly_seasonal(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        """Extract seasonal pattern using same time-of-year from prior years.
        
        This captures multi-year seasonality (e.g., spring demand vs autumn demand at same temp).
        Uses 10-min resolution: 144 periods/day, 365 days/year ≈ 52,560 periods/year.
        """
        raw = df[site].to_numpy(dtype=float)
        gap_len = end_idx - start_idx
        period_year = 144 * 365   # steps per year at 10-min resolution

        gap_positions = np.arange(start_idx, end_idx)                    # (gap_len,)
        offsets = np.arange(1, 4) * period_year                          # (3,), up to 3 years back
        candidate_idx = gap_positions[:, None] - offsets[None, :]        # (gap_len, 3)

        valid_mask = (candidate_idx >= 0) & (candidate_idx < start_idx)

        safe_idx = np.clip(candidate_idx, 0, len(raw) - 1)
        vals = raw[safe_idx]                                              # (gap_len, 3)
        vals[~valid_mask] = np.nan

        with np.errstate(all='ignore'):
            yearly = np.nanmedian(vals, axis=1)                          # (gap_len,)

        nan_mask = np.isnan(yearly)
        if np.any(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                yearly[nan_mask] = np.interp(
                    np.where(nan_mask)[0], valid_idx, yearly[valid_idx]
                )
            elif len(valid_idx) == 1:
                yearly[nan_mask] = yearly[valid_idx[0]]
            else:
                yearly[:] = np.nanmean(raw)

        return yearly

    def _method_spline(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        try:
            before_idx = np.arange(max(0, start_idx - 48), start_idx)
            after_idx = np.arange(end_idx, min(len(df), end_idx + 48))
            before_vals = df.loc[before_idx, site].dropna()
            after_vals = df.loc[after_idx, site].dropna()
            if len(before_vals) >= 4 and len(after_vals) >= 4:
                anchor_idx = np.concatenate([before_vals.index.values, after_vals.index.values])
                anchor_vals = np.concatenate([before_vals.values, after_vals.values])
                
                before_slope = np.polyfit(np.arange(len(before_vals))[-4:], before_vals.values[-4:], 1)[0]
                after_slope = np.polyfit(np.arange(len(after_vals))[:4], after_vals.values[:4], 1)[0]
                
                cs = CubicSpline(anchor_idx, anchor_vals, bc_type=((1, before_slope), (1, after_slope)))
                return cs(np.arange(start_idx, end_idx))
        except Exception:
            pass
        return np.full(end_idx - start_idx, np.nan)

    def _method_fourier(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int) -> np.ndarray:
        """FIX #2: exclude DC component before adding context.mean() offset."""
        try:
            context_size = min(288, start_idx)
            context = df.loc[max(0, start_idx - context_size):start_idx - 1, site].dropna()
            if len(context) >= 10:
                fft_vals = fft(context.values)
                magnitudes = np.abs(fft_vals)
                magnitudes[0] = 0  # zero out DC so it is never selected
                top_freqs = np.argsort(magnitudes)[-5:]

                t_gap = np.arange(end_idx - start_idx)
                reconstruction = np.zeros(end_idx - start_idx)
                for freq in top_freqs:
                    amp = np.abs(fft_vals[freq]) / len(context)
                    phase = np.angle(fft_vals[freq])
                    reconstruction += amp * np.cos(2 * np.pi * freq * t_gap / len(context) + phase)

                if reconstruction.std() > 1e-6:
                    reconstruction = reconstruction * (context.std() / reconstruction.std())

                reconstruction += context.mean()
                return reconstruction
        except Exception:
            pass
        return np.full(end_idx - start_idx, np.nan)

    def _assess_method_confidence(self, df: pd.DataFrame, site: str, values: np.ndarray) -> float:
        """FIX #11: use 5th–95th percentile bounds instead of global min/max."""
        if np.all(np.isnan(values)):
            return 0.0

        clean = values[~np.isnan(values)]
        diffs = np.abs(np.diff(clean))
        smoothness_score = 1.0 / (1 + np.mean(diffs) / 1000) if len(diffs) > 0 else 0.1

        valid = df[site].dropna()
        if len(valid) >= 20:
            lo, hi = np.percentile(valid, 5), np.percentile(valid, 95)
        else:
            lo, hi = valid.min(), valid.max()

        in_bounds = np.mean((clean >= lo) & (clean <= hi))
        bounds_score = 0.5 + 0.5 * in_bounds

        return smoothness_score * bounds_score

    def _apply_adaptive_smoothing(self, df: pd.DataFrame, site: str, start_idx: int, end_idx: int):
        values = df.loc[start_idx:end_idx - 1, site].values
        if start_idx >= 24:
            before = df.loc[max(0, start_idx - 24):start_idx - 1, site].dropna()
            smoothness = np.mean(np.abs(np.diff(before.values))) if len(before) > 1 else 1000
        else:
            smoothness = 1000

        if smoothness < 2000:
            gap_len = end_idx - start_idx
            window_len = min(11, gap_len if gap_len % 2 == 1 else gap_len - 1)
            if window_len >= 5:
                try:
                    df.loc[start_idx:end_idx - 1, site] = savgol_filter(values, window_len, polyorder=2)
                except Exception:
                    pass

    def _build_templates_if_needed(self, df: pd.DataFrame):
        fingerprint = self._data_fingerprint(df)
        if self._templates is not None and self._cache_fingerprint == fingerprint:
            return self._templates

        print("[INFO] Building templates...", flush=True)
        self._build_temp_aware_templates(df)

        self._rolling = {}
        for site in self.site_cols:
            if site not in df.columns:
                continue
            self._rolling[site] = (
                df.groupby(df['Timestamp'].dt.strftime('%H:%M'))[site].mean().to_dict()
            )

        self._cache_fingerprint = fingerprint
        self.cache_templates(fingerprint)
        return self._templates

    def _build_temp_aware_templates(self, df: pd.DataFrame):
        """Build templates with 4-dimensional key: (regime, season, day_type, hour_min).
        
        BUG #1: each row classified using its OWN timestamp (not iloc[0]).
        PERF #6: single groupby pass per site, sliced per regime/season/day_type.
        PERF #8: vectorised regime + season + day_type assignment, no row-wise .apply().
        """
        self._templates = {}

        data = df.copy()
        if self.use_historical_data and self.historical_df is not None:
            common = [c for c in self.site_cols if c in self.historical_df.columns]
            hist_cols = [c for c in ['Timestamp'] + common if c in self.historical_df.columns]
            curr_cols = ['Timestamp'] + common
            data = pd.concat(
                [self.historical_df[hist_cols], df[[c for c in curr_cols if c in df.columns]]],
                ignore_index=True,
            )

        if 'AirTemp' in data.columns:
            temps = data['AirTemp'].to_numpy(dtype=float)
            regime = np.where(np.isnan(temps), 'mild',
                     np.where(temps < 5,       'cold',
                     np.where(temps > 20,      'hot', 'mild')))
        else:
            regime = np.full(len(data), 'mild', dtype=object)

        months = data['Timestamp'].dt.month.to_numpy()
        season = np.where(np.isin(months, [12, 1, 2]),   'winter',
                 np.where(np.isin(months, [3, 4, 5]),    'spring',
                 np.where(np.isin(months, [6, 7, 8]),    'summer', 'autumn')))

        weekday_num = data['Timestamp'].dt.weekday.to_numpy()
        dates = data['Timestamp'].dt.date.values
        is_close = np.isin(dates, list(self.close_days))
        is_holiday = np.isin(dates, list(self.holidays))
        is_special = np.isin(dates, list(self.special_days))
        
        day_type = np.where(is_close,                'close',
                   np.where(is_special,              'special',
                   np.where(is_holiday,              'holiday',
                   np.where(weekday_num == 4,        'friday',
                   np.where(weekday_num == 5,        'saturday',
                   np.where(weekday_num == 6,        'sunday', 'weekday'))))))

        data['_regime']   = regime
        data['_season']   = season
        data['_day_type'] = day_type
        data['_hm']       = data['Timestamp'].dt.strftime('%H:%M')

        regimes = ['cold', 'mild', 'hot']
        seasons = ['winter', 'spring', 'summer', 'autumn']
        day_types = ['weekday', 'friday', 'saturday', 'sunday', 'holiday', 'close', 'special']

        for site in self.site_cols:
            if site not in data.columns:
                continue
            self._templates[site] = {}

            grouped = data.groupby(['_regime', '_season', '_day_type', '_hm'])[site].mean()

            for reg in regimes:
                self._templates[site][reg] = {}
                for seas in seasons:
                    self._templates[site][reg][seas] = {}
                    for dt in day_types:
                        try:
                            profile = grouped.loc[reg, seas, dt]
                            self._templates[site][reg][seas][dt] = profile.to_dict()
                        except KeyError:
                            # Graceful fallback: use _fallback_template logic
                            self._templates[site][reg][seas][dt] = {}

    def _calculate_bounds(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        bounds = {}
        for site in self.site_cols:
            if site in df.columns:
                valid = df[site].dropna()
                if len(valid) > 0:
                    mn, mx = valid.min(), valid.max()
                    margin = (mx - mn) * 0.05
                    bounds[site] = (max(0.0, mn - margin), mx + margin)
        return bounds

    def _detect_anomalies(self, df: pd.DataFrame):
        """Isolation Forest + O(n) stuck-value detection via rolling std (PERF #7)."""
        for site in self.site_cols:
            if site not in df.columns:
                continue
            valid = df[site].dropna()
            if len(valid) < 10:
                continue

            scores = pd.Series(np.zeros(len(valid)), index=valid.index)

            if IsolationForest is not None:
                try:
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    iso_scores = iso.fit_predict(valid.values.reshape(-1, 1))
                    scores = pd.Series(iso_scores.astype(float), index=valid.index)
                except Exception:
                    pass

            # Stuck-value detection: rolling std over 10 samples
            rolling_std = df[site].rolling(window=10, min_periods=5).std()
            stuck_mask = rolling_std < 1e-6
            stuck_indices = df.index[stuck_mask & df[site].notna()]
            for idx in stuck_indices:
                scores[idx] = -1.0

            self.anomaly_scores[site] = scores

    def _calculate_peer_correlations(self, df: pd.DataFrame):
        # Optimized: use corr() matrix instead of pairwise (PERF #15)
        try:
            cols_exist = [c for c in self.site_cols if c in df.columns]
            if len(cols_exist) <= 1:
                for site in cols_exist:
                    self.peer_correlations[site] = {}
                return
            
            corr_matrix = df[cols_exist].corr()
            for site in cols_exist:
                self.peer_correlations[site] = {}
                for peer in cols_exist:
                    if peer != site:
                        val = corr_matrix.loc[site, peer]
                        self.peer_correlations[site][peer] = float(val) if not np.isnan(val) else 0.0
        except Exception:
            for site in self.site_cols:
                self.peer_correlations[site] = {}

    def _compute_output_confidence_bounds(self, df: pd.DataFrame):
        rmse_base = 8.0
        base_width = 1.96 * rmse_base
        
        for site in self.site_cols:
            if site not in df.columns:
                continue
            bounds_dict = {}
            
            # Vectorized anomaly score lookup (PERF #18)
            anom_scores = self.anomaly_scores.get(site, pd.Series(dtype=bool))
            anom_mask_array = np.zeros(len(df), dtype=bool)
            if len(anom_scores) > 0:
                try:
                    anom_mask_array = (anom_scores < -0.5).to_numpy() if hasattr(anom_scores, 'to_numpy') else (anom_scores < -0.5)
                except:
                    pass
            
            # Vectorized temperature check (PERF #18)
            temps = df['AirTemp'].to_numpy(dtype=float) if 'AirTemp' in df.columns else np.full(len(df), np.nan)
            temp_extreme = (temps < 0) | (temps > 30)
            
            site_values = df[site].to_numpy(dtype=float)
            valid_mask = ~np.isnan(site_values)
            
            for idx in np.where(valid_mask)[0]:
                width = base_width
                if idx < len(anom_mask_array) and anom_mask_array[idx]:
                    width *= 1.5
                if idx < len(temps) and not np.isnan(temps[idx]) and temp_extreme[idx]:
                    width *= 1.2
                val = site_values[idx]
                bounds_dict[idx] = (max(0.0, val - width), val + width)
            
            if bounds_dict:
                self.confidence_bounds[site] = bounds_dict
