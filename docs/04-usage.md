# 4. Usage

This chapter is the operational reference: how to build the image, how to run the daemon, how to run one-off reconstructions, how to use the test harness, how to configure the module, and how to embed it in another Python process.

For the reasoning behind the flags and the design of the CSV formats, see [`03-data-model.md`](03-data-model.md). For the algorithm, see [`06-algorithms.md`](06-algorithms.md).

## 4.1 Prerequisites

Docker with Compose v2 is required (use `docker compose`, not the legacy `docker-compose` binary). A writable `./io/` and `./cache/` directory must exist next to `docker-compose.yml`; the compose file mounts them into the container.

For Cassandra mode, supply reachable contact points (`CASSANDRA_HOSTS`) and create the `conso_historiques_reconstructed` table once per cluster with the DDL in [`../Imputation-Module/cassandra/schema.cql`](../Imputation-Module/cassandra/schema.cql):

```bash
cqlsh -f Imputation-Module/cassandra/schema.cql
```

For CSV mode, place an input CSV at `Imputation-Module/docker-imputation/io/input.csv` matching the contract in [`03-data-model.md`](03-data-model.md#33-cli-csv-io-contract---source-csv) (exactly 1008 rows on a 10-minute grid).

The repo's `data/` folder provides reference CSVs used as the 56-day history seed for `Ptot_HA`. It is mounted read-only inside the container at `/data`.

## 4.2 Build

All commands below assume the current directory is `Imputation-Module/docker-imputation/`:

```bash
cd Imputation-Module/docker-imputation
docker compose build
```

The build context is the project root (`context: ../..` in the compose file), so the `Dockerfile` can `COPY` from `Imputation-Module/src/`. First build takes 3 to 5 minutes (installs `gcc` for pinned native wheels); subsequent builds reuse the pip layer.

## 4.3 Running modes

The container's entry point is `python` (the `ENTRYPOINT` in the Dockerfile) and the default command is `scheduler.py`. That means `docker compose up` starts the nightly daemon, while `docker compose run imputer <script.py> [args]` runs an arbitrary script one-shot.

### 4.3.1 Daemon mode (default)

Starts the long-running scheduler:

```bash
docker compose up -d
docker compose logs -f imputer
```

At `$SCHEDULE_HOUR:$SCHEDULE_MINUTE` Europe/Paris (default `23:50`), the scheduler ticks. On each tick it computes the target date (tomorrow, Europe/Paris), wipes stale `/io/reconstructed_*.csv` and `*.png` files from the previous run, and calls `impute_cli.py --source cassandra --target-date <tomorrow> --building <b> --output /io/reconstructed_<b>_<date>.csv` once per entry in `config.BUILDINGS`. Each per-building call is a subprocess; a crash on one building is caught, logged, and does not abort the rest of the batch. The daemon container itself stays up (`restart: unless-stopped`). Audit JSON logs under `/io/audit_logs/` are kept between runs.

Change the tick time via env vars, either inline or through a `.env` file next to `docker-compose.yml`:

```bash
SCHEDULE_HOUR=2 SCHEDULE_MINUTE=0 docker compose up -d
```

### 4.3.2 Manual trigger (`scheduler.py --run-now`)

Run one full batch (all buildings) immediately without restarting the daemon:

```bash
docker compose run --rm imputer scheduler.py --run-now
```

`--run-now` executes the batch once and exits with a non-zero code if any building failed. The existing daemon (from `docker compose up -d`) is untouched: `docker compose run` creates a separate ephemeral container alongside it. The target date is still "tomorrow in Europe/Paris".

Optional flags (all passed through to each per-building `impute_cli.py` call):

| Flag | Effect |
|---|---|
| `--with-plots` | Emits `reconstructed_<b>_<date>.png` alongside each CSV. Off by default so nightly cron stays CSV-only. |
| `--overlay-prior-week` | Adds a dashed "copy last week" baseline to the plot. Meaningful only with `--with-plots`. |
| `--overlay-actual` | Adds a solid black line for the pre-imputation measured values. Meaningful only with `--with-plots`. |
| `--test-gap START END` | Repeatable. Masks rows in `[START, END]` (inclusive) with NaN in memory for every building and emits a per-building `reconstructed_<b>_<date>_test_report.csv` with MAE/RMSE/max. Cassandra is never modified. |

Example, plotting every building with both overlays:

```bash
docker compose run --rm imputer scheduler.py \
    --run-now --with-plots --overlay-prior-week --overlay-actual
```

For a custom target date or a single building, use `impute_cli.py` directly (next section).

### 4.3.3 CSV mode

No Cassandra, no network. Reconstructs a single 1008-row window from a local CSV.

```bash
docker compose run --rm imputer impute_cli.py \
    --source csv \
    --input /io/input.csv \
    --output /io/output.csv \
    --seed 42
```

Flags:

| Flag | Required | Description |
|---|---|---|
| `--source csv` | yes | Select CSV input mode. |
| `--input PATH` | yes | 1008-row input CSV. See [`03-data-model.md`](03-data-model.md#33-cli-csv-io-contract---source-csv). |
| `--output PATH` | yes | Output CSV path. |
| `--plot PATH` | no | Writes a reconstruction overlay PNG. |
| `--seed N` | no | Seeds `np.random` for reproducible runs. |
| `--overlay-actual` | no | Adds the pre-imputation measured curve to the plot. |
| `--test-gap START END` | no | Synthetic gap mask; see section 4.4. |
| `--test-report PATH` | no | Per-point comparison CSV; requires `--test-gap`. |

The `--building` flag is accepted in CSV mode but ignored; the input column is always named `value`.

Success prints `[OK] Imputed N gap point(s) -> /io/output.csv` and exits 0. Validation or runtime errors go to stderr as `ERROR: ...` with exit code 1.

### 4.3.4 Cassandra mode

Pulls the seven-day window ending the day *before* `--target-date`, plus the full 56-day history and weather, directly from Cassandra. Writes the reconstructed window back to `conso_historiques_reconstructed` and to the output CSV.

```bash
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12 \
docker compose run --rm imputer impute_cli.py \
    --source cassandra \
    --target-date 2026-04-10 \
    --building Ptot_HA \
    --output /io/output.csv \
    --plot /io/output.png
```

Flags specific to Cassandra mode:

| Flag | Required | Description |
|---|---|---|
| `--source cassandra` | yes | Select Cassandra input mode. |
| `--target-date YYYY-MM-DD` | yes | Reconstructs the 7-day window ending the day before this date. |
| `--building NAME` | no | One of `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, `Ptot_RIZOMM`, `Ptot_Campus`. Default `Ptot_HA`. |
| `--output PATH` | yes | Output CSV path. |
| `--plot PATH` | no | Overlay PNG. |
| `--overlay-prior-week` | no | Adds a dashed "copy last week" baseline to the plot. Cassandra mode only. |
| `--overlay-actual` | no | Adds a solid black line for the pre-imputation measured values. |
| `--seed N` | no | RNG seed. |

Connection settings come from environment variables; see section 4.6.

In Cassandra mode, the output CSV is written first, then the rows are upserted to `conso_historiques_reconstructed`. If the Cassandra write fails, the CSV is still on disk for manual ingestion or retry.

### 4.3.5 All-buildings batch (`run-all-buildings.sh`)

A convenience bash wrapper that loops the Cassandra path over all five columns for a given date, writing one CSV and one PNG per building into `./io/`.

```bash
cd Imputation-Module/docker-imputation
CASSANDRA_HOSTS=10.64.253.10 ./run-all-buildings.sh 2026-04-10
```

Without a date argument, defaults to today (UTC). Forwarded flags:

| Flag | Effect |
|---|---|
| `--test-gap START END` | Repeatable. Synthetic masks applied to every building; emits a per-building `test_report_<b>_<date>.csv`. When set, output filenames are suffixed `_test` so clean reconstructions are not overwritten. |
| `--overlay-prior-week` | Dashed "copy last week" baseline on every plot. |
| `--overlay-actual` | Solid pre-imputation curve on every plot. |
| `--no-clear` | Keep previous `reconstructed_*.csv`, `reconstructed_*.png`, and `test_report_*.csv` files in `./io/` instead of wiping them. |

Unlike the scheduler-based `--run-now`, this script does not continue past failures. `set -euo pipefail` at the top of the script means the whole script aborts on the first building that fails, but partial outputs from successful earlier buildings are kept.

## 4.4 Test harness

Either CSV mode or Cassandra mode can be run against known-good data to measure reconstruction error. You supply one or more `--test-gap START END` pairs; the CLI loads the full input as normal, copies the values in each `[START, END]` range into `ground_truth`, replaces those values with NaN in memory only, and runs the imputer on the now-holed series. With `--test-report PATH`, it also writes a per-point CSV with `timestamp, ground_truth, imputed, quality, abs_error` plus footer comments for MAE, RMSE, max absolute error, and the per-gap strategy chosen.

Example (Cassandra mode, two synthetic gaps, both overlays):

```bash
CASSANDRA_HOSTS=10.64.253.10 docker compose run --rm imputer impute_cli.py \
    --source cassandra \
    --target-date 2026-04-16 \
    --building Ptot_HEI_13RT \
    --output /io/output.csv \
    --plot /io/plot.png \
    --test-gap "2026-04-14 08:00" "2026-04-14 14:00" \
    --test-gap "2026-04-15 18:00" "2026-04-15 22:00" \
    --test-report /io/report.csv \
    --overlay-actual
```

A few rules and gotchas. The bound format is naive Europe/Paris, inclusive on both ends: `"2026-04-14 08:00"` to `"2026-04-14 14:00"` is a full six-hour, 37-point gap. A range fully outside the reconstructed 7-day window is a hard error; a partial overlap is clipped with a `[WARN]` on stderr. CSV mode rejects masking the entire 1008-point window (no history to reconstruct from), while Cassandra mode accepts it because the 56-day context is still available, but warns. Masking happens in memory: neither the source CSV nor the Cassandra cluster is modified, and reruns with the same flags are deterministic if `--seed` is also fixed. With `--plot`, each masked range is drawn as a translucent grey band; `--overlay-actual` draws the ground-truth values hidden under the mask as a solid black line, which is the visible diff against the reconstruction.

When supervising the test harness against end-of-data dates, remember to include the naive zero-fill baseline (`naive_impute` in `imputer.py`) as a comparison point so reviewers can see the improvement over a "just fill with zeros" strategy. The `--overlay-prior-week` flag serves the same role visually (dashed "copy last week" curve).

## 4.5 Python import

There is no HTTP API. For embedding the module in another Python process:

```python
from imputer import impute

imputed, quality = impute(
    series,            # pd.Series[float], 1008 points, NaN for gaps
    date_index,        # pd.DatetimeIndex or pd.Series[datetime], 10-min grid
    building_column="Ptot_HA",
    random_seed=42,
)
# imputed : pd.Series[float], no NaN
# quality : pd.Series[int], 0..3 per row
```

Signature (from `imputer.impute`):

```python
def impute(
    series: pd.Series,
    date_index,
    quality: Optional[pd.Series] = None,
    *,
    large_gap_min: int = 144,
    random_seed: Optional[int] = None,
    building_column: Optional[str] = None,
    **_kwargs,
) -> tuple[pd.Series, pd.Series]:
    ...
```

Three helpers alongside `impute()`. `set_history_source(df, building_column)` injects a pre-loaded 56-day history for a given building; call this before `impute()` when the caller has already pulled the Cassandra history once and wants to avoid re-reading reference CSVs (the CLI's Cassandra path does this automatically). `get_last_strategy_log()` returns the `strategy_log` from the most recent `impute()` call, with indices already shifted back to the caller's series, and is useful for building custom test reports. `naive_impute(series, method="linear")` is a zero-fill baseline: it returns the series with NaNs replaced by `0.0` plus a quality vector (`0` for real, `1` for filled), intended as a comparison point in evaluations.

The import has some non-obvious requirements. `Imputation-Module/src/` must be on `PYTHONPATH` (the Docker image sets this; locally, run Python from `Imputation-Module/src/` or `sys.path.insert(0, ...)`). `config.py` reads env vars at import time, so set `CASSANDRA_*`, `IMPUTER_DATA_DIR`, etc. before importing. Numpy 1.19, pandas 1.0.5, scikit-learn 0.23.1, scipy 1.5.4, and `pykalman` must be available (see `requirements.txt`).

## 4.6 Configuration

All runtime configuration is via environment variables. Defaults live in `config.py`, the Dockerfile pre-sets container paths, and `docker-compose.yml` forwards the rest from the host environment or a `.env` file next to it.

### Cassandra

| Variable | Default | Effect |
|---|---|---|
| `CASSANDRA_HOSTS` | `127.0.0.1` (compose: `cassandra-single-node`) | Comma-separated contact points. Production: `10.64.253.10,10.64.253.11,10.64.253.12`. |
| `CASSANDRA_USERNAME` | *(empty)* | If set, switches to `PlainTextAuthProvider`. |
| `CASSANDRA_PASSWORD` | *(empty)* | Paired with username. |
| `CASSANDRA_KEYSPACE` | `previsions_data` | Keyspace holding input tables and the reconstructed output table. |

### Paths (pre-set by the Dockerfile in container, overridable)

| Variable | Container default | Effect |
|---|---|---|
| `IMPUTER_DATA_DIR` | `/data` | Reference CSV directory. |
| `IMPUTER_RECENT_HA_CSV` | `/data/Cons_Hotel Academic_2026-03-22_2026-04-10.csv` | Override the recent-HA CSV path. |
| `IMPUTER_OUTPUT_DIR` | `/app/cache` | Base path for optional outputs. Not actively used by the shipped code; reserved for the template-cache functionality (the engine has `save_templates` / `load_templates` methods but nothing currently calls them). |
| `IMPUTER_AUDIT_LOG_DIR` | `/io/audit_logs` | Per-run audit JSON output. |

### Scheduling

| Variable | Default | Effect |
|---|---|---|
| `SCHEDULE_HOUR` | `23` | Cron hour (0 to 23) for the nightly tick. |
| `SCHEDULE_MINUTE` | `50` | Cron minute (0 to 59) for the nightly tick. |
| `TZ` | `Europe/Paris` | Container OS timezone; should match `config.TIMEZONE`. |

Bad values for `SCHEDULE_HOUR` or `SCHEDULE_MINUTE` (non-integer or out of range) cause the container to exit immediately with a clear error message. The defaults are sensible, so leave them unset unless you need a different tick time.

### `.env` usage

The compose file accepts a `.env` next to `docker-compose.yml` (`env_file: required: false`). A typical production `.env`:

```env
CASSANDRA_HOSTS=10.64.253.10,10.64.253.11,10.64.253.12
CASSANDRA_USERNAME=imputer
CASSANDRA_PASSWORD=********
CASSANDRA_KEYSPACE=previsions_data
SCHEDULE_HOUR=23
SCHEDULE_MINUTE=50
```

`.env` is gitignored. Do not commit credentials.

## 4.7 Volumes and networks

The compose file wires three host-to-container mounts:

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `../../data/` | `/data/` | read-only | Reference CSVs for the 56-day history fallback. |
| `./io/` | `/io/` | read-write | Input CSVs, output CSVs, PNGs, audit logs. |
| `./cache/` | `/app/cache/` | read-write | Template pickle cache. Safely discardable. |

The container joins an external Docker network declared in `docker-compose.yml` as:

```yaml
networks:
  cassandra_net:
    external: true
    name: cassandra_default
```

That `cassandra_default` network must already exist on the host (typically created by a separate Cassandra compose stack). If your Cassandra cluster is reachable at an IP or hostname that does not require joining a specific Docker network, delete the `networks:` block and the `cassandra_net:` attachment on the service. The container will then use the host's default network and you only need `CASSANDRA_HOSTS` set correctly.

## 4.8 Where output lands

| Artefact | Location |
|---|---|
| Reconstructed series CSV | `/io/reconstructed_<building>_<target_date>.csv` |
| Optional overlay PNG | `/io/reconstructed_<building>_<target_date>.png` |
| Test report CSV (if `--test-gap`) | `/io/reconstructed_<building>_<target_date>_test_report.csv` (scheduler) or the path you supplied to `--test-report` (direct CLI) |
| Per-run audit JSON | `/io/audit_logs/<timestamp>.json` |
| Template cache (reserved, not written by shipped code) | `/app/cache/` |

In Cassandra mode, the reconstructed series is also upserted into `conso_historiques_reconstructed`. See [`03-data-model.md`](03-data-model.md#313-conso_historiques_reconstructed-output).
