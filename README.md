# Project N28 -- Improving Predictive Model Resilience Through Data Quality and Continuity

JUNIA R&D project (LiveTree demonstrator). A neural network predicts daily energy consumption for instrumented buildings. This project adds a resilience layer ("Module de Pilotage") that fills data gaps so predictions keep running during multi-hour outages.

## Repository Structure

```
phase-1/                 Research phase: gap analysis, synthetic hole injection, literature review
phase-2/
  Data/                  CSV time series (consumption, weather, forecasts) -- not in git
  Prediction Model/      Production prediction service (Docker + Keras + Kafka)
demo/                    Single-building demo pipeline (current focus)
.claude/                 Architecture docs, file index, conventions
```

## Prerequisites

**Data and model files** (not tracked in git) must be present at these paths:

```
phase-2/Data/
  2026 historical data.csv
  2026 weather data.csv

phase-2/Prediction Model/docker-previsions-conso/build/previsions_conso/code/
  my_modelCons3HA_Puissances_Ptot.h5
  scalerConsoHA_Puissances_Ptot.save
  scalerxConsoHA_Puissances_Ptot.save
  Holidays.xlsx
```

Obtain these files from the project shared drive or from a team member.

## Demo Pipeline

The demo validates the imputation + prediction pipeline on a single building (`Ptot_HA` -- Hotel Academique). It runs entirely on CSV files, no Kafka or Cassandra needed.

**Pipeline flow:** Load CSV -> extract 7-day window -> inject synthetic gaps -> impute -> predict -> compare against no-gap baseline.

### Option A: Run locally

Requires Python 3.7.

```bash
cd demo
pip install -r requirements.txt

# Single scenario
python run_demo.py --target-date 2026-01-15

# With plots and summary output
python run_demo.py --target-date 2026-01-15 --save-plots --save-summary

# Custom gap configuration
python run_demo.py --target-date 2026-01-15 --block-length 72 --n-blocks 3
python run_demo.py --target-date 2026-01-15 --gap-mode random --n-points 100

# Multi-scenario batch (runs all configs from scenarios.json)
python run_demo.py --target-date 2026-01-15 --scenarios scenarios.json --save-plots --save-summary
```

### Option B: Run with Docker

```bash
cd demo
cp .env.example .env        # edit .env to change target date, gap params, etc.
docker-compose build         # first time only, or when requirements.txt changes
docker-compose up            # uses settings from .env

# Override arguments directly
docker-compose run demo --target-date 2026-02-01 --save-plots --save-summary
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--target-date` | required | Date to predict (YYYY-MM-DD), needs 7 prior days of data |
| `--gap-mode` | `blocks` | `blocks` (contiguous) or `random` (scattered) |
| `--block-length` | `36` | Points per gap block (36 = 6 hours at 10-min intervals) |
| `--n-blocks` | `2` | Number of gap blocks to inject |
| `--n-points` | `50` | Number of random NaN points (when `--gap-mode random`) |
| `--seed` | `42` | Random seed for reproducible gap injection |
| `--save-plots` | off | Generate PNG plots in output directory |
| `--save-summary` | off | Save metrics to `summary.csv` and `summary.json` |
| `--output-dir` | `demo/output/` | Where to write plots and summaries |
| `--scenarios` | none | Path to JSON file for multi-scenario batch runs |

### Output

When `--save-plots` is used, each scenario produces three plots in `output/{date}_{label}/`:

- **predictions.png** -- baseline vs imputed predictions (+ actual if available)
- **history_gaps.png** -- 7-day history window with color-coded imputation regions
- **metrics.png** -- bar chart of MAE, RMSE, MAPE

In multi-scenario mode, an additional **scenario_comparison.png** shows metric degradation across all gap configurations, and `--save-summary` writes a **summary.csv** for programmatic analysis.

### Scenarios File Format

`scenarios.json` defines multiple gap configurations to test in one batch:

```json
[
  {"label": "6h_gap", "gap_mode": "blocks", "block_length": 36, "n_blocks": 1},
  {"label": "24h_gap", "gap_mode": "blocks", "block_length": 144, "n_blocks": 1},
  {"label": "random_50", "gap_mode": "random", "n_points": 50}
]
```

A default `scenarios.json` is included with gap sizes from 1h to 48h.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (watts) |
| RMSE | Root Mean Squared Error (watts) |
| MAPE | Mean Absolute Percentage Error (%) |

Three comparisons are computed:
- **baseline vs imputed** -- how much predictions degrade due to gap filling (primary metric)
- **baseline vs actual** -- baseline model accuracy
- **imputed vs actual** -- imputed model accuracy

## Imputation Strategy

Cascading approach based on gap length:

1. **Short gaps** (<=30 min / 3 points): linear interpolation
2. **Medium gaps** (30 min - 6h / 4-36 points): blend of linear and seasonal profile, weight increasing with gap length
3. **Long gaps** (>6h / >36 points): seasonal backup injection (median by weekday + time-of-day)

Each imputed point receives a quality flag (0=real, 1=linear, 2=seasonal, 3=backup).

## Tech Stack

Python 3.7, TensorFlow 2.2, Keras 2.4, pandas 1.0.5, numpy 1.19, matplotlib 3.3, Docker.
All dependency versions are pinned in `demo/requirements.txt`.
