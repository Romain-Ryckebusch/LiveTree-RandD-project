# LiveTree Imputation Module

Project N28, *"Amélioration de la résilience des modèles prédictifs par la qualité et la continuité des données"* (JUNIA, LiveTree demonstrator).

Three JUNIA buildings (Hôtel Académique, 13 Rue de Toul / HEI_13RT, RIZOMM) report electrical consumption every 10 min. The downstream forecasting model needs the last 7 days of readings (1008 points) to produce a prediction. The acquisition pipeline is unreliable enough that this window rarely arrives complete, so the model can't run on it as-is.

This is the *module de pilotage*. It sits between Cassandra and the forecaster, takes the holed 7-day window, and returns a complete one. Each imputed point is tagged with a quality flag so callers can filter on confidence if they want.

Gap length determines the strategy:

- gaps up to 30 min: linear interpolation
- 30 min to 6 h: weekday-and-hour templates, weighted by outside temperature
- longer than 6 h: donor-day ensemble over the last 56 days

The output always covers all five building columns: `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, and `Ptot_Campus` (the sum of the four, computed internally).

## Quick start

Build the image and run a reconstruction:

```bash
cd Imputation-Module/docker-imputation
docker compose build
docker compose run --rm imputer \
    --source csv \
    --input /io/input.csv \
    --output /io/output.csv
```

`./io/` under `docker-imputation/` is mounted inside the container as `/io`. Put the input CSV there; the output lands in the same folder.

## Usage

`impute_cli.py` is the container's default command. The `--source` flag selects the input mode.

### CSV mode

For offline tests or any environment without Cassandra access.

```bash
docker compose run --rm imputer \
    --source csv \
    --input /io/input.csv \
    --output /io/output.csv \
    --plot /io/output.png \
    --seed 42
```

| Flag | Required | Description |
|------|----------|-------------|
| `--source csv` | yes (or default) | Select CSV input mode. |
| `--input PATH` | yes | Input CSV, exactly 1008 rows (see [I/O contracts](#io-contracts)). |
| `--output PATH` | yes | Output CSV path. |
| `--plot PATH` | no | Writes a reconstruction overlay PNG. |
| `--seed N` | no | Seeds the RNG for deterministic output. |

The `--building` flag is accepted in CSV mode but ignored; the column is always named `value` in the input file.

### Cassandra mode

Pulls history and weather directly from the Cassandra cluster. The window that gets reconstructed ends on the day before `--target-date`.

```bash
CASSANDRA_HOSTS=10.64.253.10 \
docker compose run --rm imputer \
    --source cassandra \
    --target-date 2026-04-10 \
    --building Ptot_HA \
    --output /io/output.csv \
    --plot /io/output.png
```

| Flag | Required | Description |
|------|----------|-------------|
| `--source cassandra` | yes | Select Cassandra input mode. |
| `--target-date YYYY-MM-DD` | yes | Reconstructs the 7-day window ending the day *before* this date. |
| `--building NAME` | no | One of `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, `Ptot_Campus`. Default is `Ptot_HA`. |
| `--output PATH` | yes | Output CSV path. |
| `--plot PATH` | no | Overlay PNG. |
| `--overlay-prior-week` | no | Adds a dashed "copy last week" baseline to the plot. |
| `--overlay-actual` | no | Adds the raw measured values as a solid black line. |

Connection variables: see [Configuration](#configuration).

### Running all buildings at once

`Imputation-Module/docker-imputation/run-all-buildings.sh` loops the Cassandra path over the five target columns for a given date and writes one CSV (and, with `--with-plots`, one PNG) per target into `./io/`.

```bash
cd Imputation-Module/docker-imputation
CASSANDRA_HOSTS=10.64.253.10 ./run-all-buildings.sh 2026-04-10
```

Without a date it uses today (UTC). Forwarded flags: `--test-gap START END` (repeatable), `--overlay-prior-week`, `--overlay-actual`, and `--no-clear` to preserve the previous run's outputs.

## Test harness

Either mode can be run against known-good data to measure reconstruction error. You provide one or more ranges to hide; the imputer reconstructs them; the CLI reports MAE, RMSE, and max error.

```bash
docker compose run --rm imputer \
    --source cassandra \
    --target-date 2026-04-10 \
    --building Ptot_HA \
    --output /io/output.csv \
    --plot /io/output.png \
    --test-gap "2026-04-09 08:00" "2026-04-09 14:00" \
    --test-gap "2026-04-08 18:00" "2026-04-08 22:00" \
    --test-report /io/report.csv \
    --overlay-actual
```

- `--test-gap START END` is repeatable; bounds are inclusive, naive Europe/Paris time.
- `--test-report PATH` writes a per-point CSV (`timestamp, ground_truth, imputed, quality, abs_error`) with `# MAE=...`, `# RMSE=...`, `# max_err=...` as footer lines.
- `--overlay-actual` draws the hidden ground truth onto the plot.
- Masking happens in memory only; neither the source CSV nor the Cassandra cluster is modified.

## I/O contracts

### Input CSV

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | ISO 8601 string | Strictly increasing, on a 10-minute grid within +/- 1 s. Naive or tz-aware both accepted. |
| `value` | float | Power in watts. Empty cells (`NaN`) mark gaps. |

The input must be exactly 1008 rows; the CLI rejects anything else up front.

### Output CSV

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | string | Copied through unchanged from the input. |
| `value` | float | Imputed value. Guaranteed non-`NaN` on successful runs. |
| `quality` | int | Per-point quality flag (see below). |

Same row count as the input, one-to-one.

### Quality flag meanings

| Flag | Meaning | Strategies (from `imputer.py`) | Typical gap length |
|------|---------|--------------------------------|--------------------|
| `0` | Real measurement (no imputation) | n/a | n/a |
| `1` | Linear interpolation | `LINEAR_MICRO`, `LINEAR_SHORT` | <= 30 min |
| `2` | Contextual (templates, peers, safe median) | `THERMAL_TEMPLATE`, `ENHANCED_TEMPLATE`, `WEEKEND_TEMPLATE_*`, `SAFE_LINEAR_MEDIAN`, `PEER_CORRELATION` | 30 min - 6 h |
| `3` | Donor-day ensemble or safe median | `MULTI_WEEK_TEMPLATE`, `SAFE_MEDIAN` | > 6 h |

For high-confidence data, filter on `quality <= 1`; if you trust the templates as well, `quality <= 2`. A `quality` of `3` means the row was reconstructed from similar past days, so it describes a plausible shape rather than an actual measurement.

## Integration

There is no HTTP API. Three ways to wire the module into a larger system, from loosest to tightest coupling.

### 1. File exchange

- Mount a host directory as `/io` in the container. `docker-compose.yml` already does this with `./io`.
- Write a 1008-row input CSV into that directory.
- Trigger `docker compose run --rm imputer ...` from a scheduler (cron, Airflow, systemd timers, etc.).
- Read the output CSV back from the same directory.

Runs are stateless; the `./cache/` directory only holds precomputed hybrid-engine templates and can be discarded. Parallel runs work as long as their input and output paths don't collide.

### 2. Direct Cassandra reads

When the data is already in the LiveTree cluster. The module reads two tables (see `Imputation-Module/src/cassandra_client.py` and `src/config.py`):

- `conso_historiques_clean`, partition key `name = 'Conso_Data'`, columns `Date`, `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`.
- `pv_prev_meteo_clean`, partition key `name = 'Meteorological_Prevision_Data'`. Only `AirTemp` is actually consumed; the other weather columns are ignored.

Keyspace defaults to `previsions_data`; override with `CASSANDRA_KEYSPACE`. Reads only, no writes.

The container joins a Docker network called `cassandra_default` (declared as `cassandra_net` with `external: true` in `docker-compose.yml`). Adjust or drop this if the Cassandra stack runs on a different host.

### 3. Python import

The core can be imported directly:

```python
from imputer import impute

imputed, quality = impute(
    series,          # pd.Series[float], 1008 points, NaN for gaps
    timestamps,      # pd.Series[datetime], 10-min grid aligned to series
    building_column="Ptot_HA",
    random_seed=42,
)
```

Return value: two `pd.Series` of matching length, holding the reconstructed values and the per-point quality flag. Relevant entry points:

- `Imputation-Module/src/imputer.py`, top-level `impute()`. `set_history_source()` lets you hand in a 56-day history from Cassandra.
- `Imputation-Module/src/window.py`, `extract_window()`: slices a 7-day chunk out of a longer history frame.
- `Imputation-Module/src/hybrid_engine.py`, `TemperatureAwareHybridEngine`: the underlying engine for contextual and donor-day strategies.

## Configuration

Configuration is via environment variables. Defaults are in `Imputation-Module/src/config.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CASSANDRA_HOSTS` | `127.0.0.1` | Comma-separated list of contact points. Production: `10.64.253.10,10.64.253.11,10.64.253.12`. |
| `CASSANDRA_USERNAME` | *(empty)* | Username if auth is enabled; empty means anonymous. |
| `CASSANDRA_PASSWORD` | *(empty)* | Password if auth is enabled. |
| `CASSANDRA_KEYSPACE` | `previsions_data` | Keyspace holding the two tables above. |
| `IMPUTER_DATA_DIR` | repo's `data/` | Directory containing the reference CSVs used by CSV mode for the 56-day history. |
| `IMPUTER_RECENT_HA_CSV` | `$IMPUTER_DATA_DIR/Cons_Hotel Academic_*.csv` | Overrides the recent-HA reference CSV path. |
| `IMPUTER_OUTPUT_DIR` | `src/output/` | Template cache location (`hybrid_templates_cache_<building>.pkl`). The shipped `docker-compose.yml` mounts this to `./cache/`. |
