# Deployment Gap Recovery Algorithm

**Enhanced smart meter data imputation** for handling missing values in power consumption time series.

## Overview

This project implements an advanced multi-tiered gap recovery algorithm for energy consumption data. It fills missing values (gaps) in power meter data using a combination of machine learning and signal processing techniques.

### Key Features

- **5-Tier Safety Architecture**: Progressively robust recovery strategies
  - **Tier 1**: Intelligent gap routing based on type and meter hierarchy
  - **Tier 2**: Thermal regime filtering for temperature-aware template selection
  - **Tier 3**: Parent-child peer correlation for sub-meter ratio preservation
  - **Tier 4**: NaN guard for recursive exception handling
  - **Tier 5**: Signal smoothing at reconstruction junctions

- **Multiple Recovery Methods**
  - K-Nearest Neighbors (KNN) template matching
  - Multiple Imputation by Chained Equations (MICE)
  - Kalman Filter smoothing
  - Autoencoder-based profile imputation

- **Smart Features**
  - Weekday/weekend pattern recognition
  - Thermal regime classification
  - Multi-site correlation analysis
  - Confidence scoring for recovered values
  - Low-confidence flag system for manual review

## Quick Start

### Installation

1. **Clone/setup the repository**
   ```bash
   cd Deployment_version
   ```

2. **Create a Python virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

**Option 1: Using configuration file**
```bash
python main.py --config config.yaml
```

**Option 2: Using command-line arguments**
```bash
python main.py --data data/2026 historical data.csv \
               --weather data/2026 weather data.csv \
               --output output/recovered_data.csv
```

**Option 3: Python API**
```python
from deployment_gap_recovery_enhanced import DeploymentGapRecovery
import pandas as pd

# Load data
df = pd.read_csv('data/2026 historical data.csv')
weather_df = pd.read_csv('data/2026 weather data.csv')

# Initialize algorithm
algo = DeploymentGapRecovery(
    use_mice=True,
    use_knn=True,
    use_kalman=True
)

# Run imputation
recovered_df = algo.impute(df, weather_df)

# Save results
recovered_df.to_csv('output/recovered_data.csv', index=False)
```

## Configuration

Edit `config.yaml` to customize algorithm behavior:

```yaml
algorithm:
  use_mice: true              # Enable MICE imputation
  use_knn: true               # Enable KNN template matching
  use_kalman: true            # Enable Kalman smoothing
  use_deep_learning: false    # Enable autoencoder (optional)

  knn:
    n_neighbors: 5            # Number of historical templates
    window_size: 14           # Days of training history

  confidence:
    high_threshold: 0.85      # High confidence level
    low_threshold: 0.50       # Low confidence (flagged for review)

tiers:
  tier_1_intelligent_router: true
  tier_2_thermal_filtering: true
  tier_3_parent_child_correlation: true
  tier_4_nan_guard: true
  tier_5_signal_smoothing: true
```

## Testing

### Run all tests
```bash
pytest test_deployment.py -v
```

### Run specific test category
```bash
pytest test_deployment.py::TestBasicFunctionality -v
pytest test_deployment.py::TestAlgorithmConfiguration -v
pytest test_deployment.py::TestWithRealData -v
```

### Run with real data
```bash
pytest test_deployment.py::TestWithRealData::test_with_2026_data -v
```

### Verbose output
```bash
pytest test_deployment.py -v --tb=short
```

## Project Structure

```
Deployment_version/
├── deployment_gap_recovery_enhanced.py    # Main algorithm (5 tiers)
├── deployment_gap_recovery.py             # Original algorithm
├── autoencoder_imputer.py                 # Deep learning component
├── main.py                                # CLI entry point
├── config.yaml                            # Configuration file
├── requirements.txt                       # Python dependencies
├── test_deployment.py                     # Test suite
├── README.md                              # This file
├── data/                                  # Input data directory
│   ├── 2026 historical data.csv
│   ├── 2026 weather data.csv
│   └── ...
├── output/                                # Output directory
│   ├── recovered_data.csv
│   ├── strategy_log.csv
│   └── low_confidence_flags.csv
├── logs/                                  # Log directory
│   └── deployment.log
└── gap_simulations/                       # Test scripts and results
    ├── run_weekend_test.py
    ├── run_4day_gap_deployment.py
    └── ...
```

## Data Format

### Input: Historical Data
```csv
Timestamp,Ptot_HA,Ptot_HEI,Ptot_HEI_13RT,Ptot_HEI_5RNS,Ptot_RIZOMM
2026-01-01 00:00:00,45.2,67.3,30.1,25.2,12.4
2026-01-01 00:10:00,46.1,,31.2,26.1,13.1
...
```

### Input: Weather Data
```csv
Timestamp,AirTemp
2026-01-01 00:00:00,5.2
2026-01-01 00:10:00,5.3
...
```

### Output: Recovered Data
```csv
Timestamp,Ptot_HA,Ptot_HEI,Ptot_HEI_13RT,Ptot_HEI_5RNS,Ptot_RIZOMM,AirTemp
2026-01-01 00:00:00,45.2,67.3,30.1,25.2,12.4,5.2
2026-01-01 00:10:00,46.1,68.2,31.2,26.1,13.1,5.3
...
```

## Algorithm Details

### How It Works

1. **Data Validation**: Reindex to complete time range, fill metadata
2. **Feature Engineering**: Add hour-of-day, day-of-week, thermal regime
3. **Template Building**: Build historical templates for similar days
4. **Gap Detection**: Identify contiguous gap regions
5. **Intelligent Router** (TIER 1): Route gaps to appropriate recovery strategy
   - Short gaps (1-4 points) → Local trend
   - Medium gaps (5-288 points in day) → KNN template matching
   - Long gaps (multi-day) → MICE with Kalman smoothing
6. **Recovery**: Apply selected strategy
7. **Post-Processing**: 
   - NaN Guard (TIER 4): Catch any remaining gaps
   - Signal Smoothing (TIER 5): Smooth junctions

### Key Metrics

- **NaN Elimination**: 100% of gaps filled
- **Weekend MAPE Reduction**: 86% → <30% on HEI_13RT
- **Accuracy Preservation**: ~18% on entry points
- **Data Continuity**: 100%

## Output Files

### Recovered Data (`recovered_data.csv`)
Main output with all gaps filled.

### Strategy Log (`strategy_log.csv`)
Audit trail of recovery decisions:
- Site name
- Gap start/end times
- Recovery method applied
- Confidence score

### Low Confidence Flags (`low_confidence_flags.csv`)
Values with confidence < threshold:
- Timestamp
- Site
- Recovered value
- Confidence score
- Reason (optional)

### Diagnostic Log (`deployment.log`)
Detailed execution log with timestamps and diagnostics.

## Performance

### Requirements
- **RAM**: 1-4 GB (depends on data volume)
- **CPU**: Standard multi-core processor
- **Disk**: ~100-500 MB for typical deployments

### Speed
- **1 month of data**: <30 seconds
- **1 year of data**: 2-5 minutes
- **90 days**: ~1 minute

## Troubleshooting

### All NaN values not filled
- Check configuration file loaded correctly
- Verify weather data is available
- Run with `--verbose` flag for detailed logs

### High number of low-confidence flags
- Check data quality
- Consider increasing `window_size` in KNN config
- Verify sufficient historical data (min 14 days recommended)

### Performance issues
- Disable KNN: set `use_knn: false`
- Reduce data volume (split by month)
- Use `use_deep_learning: false` for faster operations

### Missing output files
- Create `output/` directory manually
- Check write permissions
- Verify disk space

## Advanced Usage

### Disable specific methods
```python
algo = DeploymentGapRecovery(
    use_mice=False,   # Disable MICE
    use_knn=True,
    use_kalman=True
)
```

### Access diagnostic information
```python
recovered_df = algo.impute(df, weather_df)

# Strategy decisions
print(algo.strategy_log)

# Low confidence reconstructions
print(algo._low_confidence_flags)

# Peer ratio estimates
print(algo._peer_ratios)
```

### Custom site columns
```python
algo = DeploymentGapRecovery(
    site_cols=['Ptot_HA', 'Ptot_HEI', 'Ptot_RIZOMM']
)
```

## References

- **Paper**: "Data Imputation for Smart Meter Data" (mathematics-12-03004)
- **Decomposition**: Shape-energy factorization (E_i * N_ij)
- **Methods**:
  - KNN: Template matching on historical similar days
  - MICE: Multiple imputation by chained equations
  - Kalman: Recursive linear filter for smoothing
  - Autoencoder: Nonlinear shape encoding (optional)

## License

Internal research use. Not for external distribution.

## Authors

- Developed for deployment of smart meter gap recovery system
- Multiple iterations for production robustness

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/deployment.log`
3. Run test suite to validate installation: `pytest test_deployment.py -v`
4. Check output diagnostics in `output/strategy_log.csv`

---

**Last Updated**: April 2026
**Status**: Production Ready

---

## Enhanced V2 Algorithm (2023-2026)

This version (see `deployment_gap_recovery_extended.py`) supports:
- Multi-week template matching (28-day lookback)
- Adaptive template bias (recent vs historical weighting)
- Smart chunking (gap split by day boundaries and variance)
- External event detection (holidays, events, weather)
- Occupancy/context awareness (if data available)

### How It Works
1. **Template Building**: Scans up to 28 days of historical data to build daily/weekly templates, blending recent and older data.
2. **Gap Detection & Chunking**: Gaps are split into smart chunks (usually by day) to improve recovery accuracy.
3. **Recovery**: Each chunk is reconstructed using the best-matching template, with bias toward recent patterns and context (e.g., holidays).
4. **External Features**: Optionally uses holiday/event/weather data to adjust templates.
5. **Metrics**: Reports MAE, MAPE, RMSE, and R² for each building.

### Deployment Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**
   - Place your historical data in `data/2023_2026_merged.csv` (or update the script to use your file).
   - Data must have columns: `Date`, `Ptot_HA`, `Ptot_HEI`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`.
3. **Run Example Recovery**
   ```bash
   python simulate_5day_gap.py
   ```
   - This will simulate a 5-day gap, run the recovery, and save result plots in `gap_4day/`.
4. **Integrate Into Your Workflow**
   - Import `ExtendedDeploymentAlgorithm` from `deployment_gap_recovery_extended.py` in your own scripts.
   - Pass your data as a pandas DataFrame.
   - Use the `impute()` method to recover gaps.

### Example Usage
```python
from deployment_gap_recovery_extended import ExtendedDeploymentAlgorithm
import pandas as pd

df = pd.read_csv('data/2023_2026_merged.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

algorithm = ExtendedDeploymentAlgorithm(site_cols=[...])
recovered = algorithm.impute(df)
```

---
