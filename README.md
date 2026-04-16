# LiveTree Imputation Module

Project N28, *"Amélioration de la résilience des modèles prédictifs par la qualité et la continuité des données"* (JUNIA, LiveTree demonstrator).

Three JUNIA buildings (Hôtel Académique, 13 Rue de Toul / HEI_13RT, RIZOMM) push electrical consumption readings every 10 minutes. Anything reading that stream eventually wants a clean 7-day window of 1008 points. Problem is, the acquisition chain drops out for hours at a time, so the window shows up full of holes.

This module (the *module de pilotage*) takes the holed window and fills it in. Short gaps get a linear interpolation. Medium ones fall back on weekday-and-hour templates weighted by outdoor temperature. Long ones get rebuilt from a 56-day history of similar days. Every output point carries a quality flag so you can tell the real measurements apart from the imputed ones.

---

## What it does

- Always returns exactly 1008 points on a 10-minute grid (7 days × 144).
- Reconstructs five columns: `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, and `Ptot_Campus` (which is just the sum of the four buildings, computed on the fly).
- Short gaps (≤30 min) are linearly interpolated. Medium ones use weekday-and-hour templates weighted by outdoor temperature. Long gaps (>6 h) get rebuilt from a donor-day ensemble over the last 56 days.
- Every output row gets a quality flag: `0` real, `1` linear, `2` contextual, `3` donor-day. Consumers can filter or weight on it.
- Reads from a CSV file or directly from Cassandra. Same core either way.
- Ships with a synthetic-gap test harness (`--test-gap`, `--test-report`) that masks points you know are real, reconstructs them, and reports MAE / RMSE / max error.
- A batch script (`run-all-buildings.sh`) runs all five targets for one date in a single go.
- No broker, no HTTP, no hidden state. CSV in, CSV out.

---

## How it fits in

Raw data goes in with holes, a clean 1008-point window comes out. Whatever consumes the window lives outside this repo.

---

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

The `./io/` folder under `docker-imputation/` is the shared directory between host and container. It gets mounted inside as `/io`. Drop your input there, read the output back from the same place.

---

## Usage

Everything goes through `impute_cli.py`, which is the container's default command. The `--source` flag picks between two input modes.

### CSV mode

Use this for offline tests, regression runs, or any setup without Cassandra access.

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
| `--plot PATH` | no | If given, writes a reconstruction overlay PNG. |
| `--seed N` | no | Seeds the imputer's RNG for deterministic output. |

CSV mode always reconstructs a generic `value` column; the `--building` flag is accepted but ignored here.

### Cassandra mode

Pulls history and weather straight from Cassandra. The 7-day window it reconstructs ends the day *before* `--target-date`.

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
| `--plot PATH` | no | Optional reconstruction overlay PNG. |
| `--overlay-prior-week` | no | Adds a dashed "copy last week" baseline to the plot. |
| `--overlay-actual` | no | Adds the raw measured values as a solid black line. |

Connection variables are listed under [Configuration](#configuration).

### Running all buildings at once

`Imputation-Module/docker-imputation/run-all-buildings.sh` runs all five targets for one date and drops a CSV + PNG per target into `./io/`.

```bash
cd Imputation-Module/docker-imputation
CASSANDRA_HOSTS=10.64.253.10 ./run-all-buildings.sh 2026-04-10
```

Without a date, it defaults to today (UTC). You can also pass `--test-gap START END` (repeatable), `--overlay-prior-week`, `--overlay-actual`, and `--no-clear` to keep the previous run's outputs around.

---

## Test harness

Either mode can be run as a test. Mask points you know are real, let the imputer fill them back in, then compare against the ground truth. Handy for regressions and for tuning the engine.

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

- `--test-gap START END` is repeatable. Bounds are inclusive, in naive Europe/Paris time.
- `--test-report PATH` writes a per-point CSV (`timestamp, ground_truth, imputed, quality, abs_error`) with `# MAE=… # RMSE=… # max_err=…` as footer lines.
- `--overlay-actual` draws the hidden ground truth onto the plot.
- The CSV file or Cassandra cluster itself is never touched; masking only happens in memory.

---

## I/O contracts

### Input CSV

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | ISO 8601 string | Strictly increasing, on a 10-minute grid within ±1 s. Naive or tz-aware both work. |
| `value` | float | Power in watts. Empty cells (`NaN`) mark gaps. |

Has to be exactly 1008 rows. Anything else gets rejected up front with an error.

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
| `0` | Real measurement (no imputation) | — | n/a |
| `1` | Linear interpolation | `LINEAR_MICRO`, `LINEAR_SHORT` | ≤ 30 min |
| `2` | Contextual (templates, peers, safe median) | `THERMAL_TEMPLATE`, `ENHANCED_TEMPLATE`, `WEEKEND_TEMPLATE_*`, `SAFE_LINEAR_MEDIAN`, `PEER_CORRELATION` | 30 min – 6 h |
| `3` | Donor-day ensemble or safe median | `MULTI_WEEK_TEMPLATE`, `SAFE_MEDIAN` | > 6 h |

If you only want high-confidence data, keep `quality <= 1`. If you trust the templates too, keep `quality <= 2`. Rows tagged `3` are reconstructed from similar-looking days, so treat them as a plausible shape rather than an actual measurement.

---

## Plugging it into another system

There's no HTTP API. Three ways to wire the module in, roughly from loosest coupling to tightest.

### 1. Talk to it over files (recommended)

Usually the cleanest option:

- Mount a host directory as `/io` in the container. The shipped `docker-compose.yml` already does this for `./io`.
- Drop your 1008-row input CSV there.
- Fire off `docker compose run --rm imputer …` from whatever schedules your jobs (cron, Airflow, systemd timers, an event-driven worker, whatever).
- Read the output CSV back from the same folder.

Runs don't share state (the `./cache/` of hybrid-engine templates is disposable), so you can run multiple reconstructions at the same time as long as their input and output paths don't collide.

### 2. Read Cassandra directly

If the data already lives in the LiveTree Cassandra cluster, skip the CSV step and let the module pull from there.

Tables it reads (see `Imputation-Module/src/cassandra_client.py` and `Imputation-Module/src/config.py`):

- `conso_historiques_clean`, partition key `name = 'Conso_Data'`, columns `Date`, `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`.
- `pv_prev_meteo_clean`, partition key `name = 'Meteorological_Prevision_Data'`, columns `Date`, `AirTemp`, plus other weather fields (only `AirTemp` actually matters).

Keyspace defaults to `previsions_data`; override with `CASSANDRA_KEYSPACE`. The module only reads. It never writes.

The container attaches to a Docker network called `cassandra_default` (shown in `docker-compose.yml` as `cassandra_net`, `external: true`). If your Cassandra stack isn't running on the same Docker host, you'll need to adjust or drop that network.

### 3. Import it as a Python library

The core is plain Python, so you can just import it:

```python
from imputer import impute

imputed, quality = impute(
    series,          # pd.Series[float], 1008 points, NaN for gaps
    timestamps,      # pd.Series[datetime], 10-min grid aligned to series
    building_column="Ptot_HA",
    random_seed=42,
)
```

You get back two `pd.Series` of the same length: the filled values and the quality flags. A few places worth poking at:

- `Imputation-Module/src/imputer.py`: top-level `impute()` call, plus `set_history_source()` if you want to feed it a 56-day history from Cassandra directly.
- `Imputation-Module/src/window.py`: `extract_window()` for slicing a 7-day chunk out of a longer history frame.
- `Imputation-Module/src/hybrid_engine.py`: the `TemperatureAwareHybridEngine` itself. You normally don't need to touch it unless you're tuning.

---

## Configuration

Everything's driven by environment variables; defaults live in `Imputation-Module/src/config.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CASSANDRA_HOSTS` | `127.0.0.1` | Comma-separated list of cluster contact points. In production: `10.64.253.10,10.64.253.11,10.64.253.12`. |
| `CASSANDRA_USERNAME` | *(empty)* | Username if auth is enabled; empty means anonymous. |
| `CASSANDRA_PASSWORD` | *(empty)* | Password if auth is enabled. |
| `CASSANDRA_KEYSPACE` | `previsions_data` | Keyspace holding the two tables above. |
| `IMPUTER_DATA_DIR` | repo's `data/` | Directory holding the reference CSVs the CSV-mode 56-day history reads from. |
| `IMPUTER_RECENT_HA_CSV` | `$IMPUTER_DATA_DIR/Cons_Hotel Academic_…csv` | Explicit path to the recent-HA reference CSV, if you want to override the default under `IMPUTER_DATA_DIR`. |
| `IMPUTER_OUTPUT_DIR` | `src/output/` | Where the hybrid-engine template cache (`hybrid_templates_cache_<building>.pkl`) lives. `docker-compose.yml` mounts this to `./cache/`. |
