# 7. Extending the project

This chapter is the dev guide: what to touch when you want to change something. Every entry names the exact files and gives the gotchas that trip up readers who try to change one place and forget the mirror. Pair this with [`05-file-reference.md`](05-file-reference.md) for the *where* and [`06-algorithms.md`](06-algorithms.md) for the *how*.

## 7.1 Add, remove, or rename a building

The building list appears in five places, and all five must stay in lockstep.

The three canonical constants live in `Imputation-Module/src/config.py`. `BUILDINGS` is the canonical list; the CLI `--building` flag validates against it, and the scheduler iterates it. `CAMPUS_COMPONENTS` is the subset of `BUILDINGS` that gets summed into the virtual `Ptot_Campus` channel; if you add a physical meter, decide whether it contributes to the campus total. `BUILDING_TO_RECONSTRUCTED_COLUMNS` is the `{building_id: (value_column, quality_column)}` map used by `cassandra_client.write_reconstructed_window` to build its per-building `INSERT`; add a new entry for every new building.

`Imputation-Module/cassandra/schema.cql` needs the new `"Ptot_<b>"` column and the matching `quality_<b>` column on `conso_historiques_reconstructed`. Running `cqlsh -f schema.cql` against the cluster is a no-op once the table exists, so columns must be added with `ALTER TABLE ... ADD ...` instead:

```cql
ALTER TABLE previsions_data.conso_historiques_reconstructed
    ADD "Ptot_NEWBLDG" double;
ALTER TABLE previsions_data.conso_historiques_reconstructed
    ADD quality_newbldg int;
```

The upstream `conso_historiques_clean` table must also have the new `Ptot_<b>` column. That table is written by the acquisition chain, not by this repo, so coordinate with whoever owns it.

`Imputation-Module/docker-imputation/run-all-buildings.sh` has a hard-coded `BUILDINGS` array at line 81; add the new ID there.

The history seed is the last moving part. `Ptot_HA` is the only building with a reference-CSV history seed (`data/Cons_Hotel Academic_*.csv`); every other building gets its 56-day history injected via `imputer.set_history_source()` from the Cassandra pull. For a brand-new building with no Cassandra history yet, the long-gap donor-day strategy will fall back to `SAFE_MEDIAN` until about 8 weeks of clean data have accumulated.

And one routing decision: set the new meter's tier in `smart_imputation.METER_HIERARCHY` to `entry` (no parent), `sub` (has a parent), or `main` (has children). Tier affects routing: sub-meters get peer-correlation, entries get safe-linear fallbacks.

Renaming is the same list, applied consistently. Any downstream consumer that reads from `conso_historiques_reconstructed` also needs to know about the rename.

Removing a building follows the same list, and in addition requires dropping the columns from Cassandra with `ALTER TABLE ... DROP`. Existing rows retain the dropped-column data until compaction.

## 7.2 Change the window size or sampling frequency

Both are defined in `config.py`. Changing either requires an audit of hard-coded counterparts scattered through the codebase.

### Window size

Change `config.py`: `LOOKBACK_DAYS` (default 7) and `LOOKBACK_POINTS` (= `POINTS_PER_DAY * LOOKBACK_DAYS`).

Change `impute_cli.py`: `EXPECTED_ROWS` (line 16). Match it to `LOOKBACK_POINTS`.

Change `imputer.py`: `_PREPEND_DAYS = 56`. Rule of thumb: keep it at least `4 × template_lookback_days` (default 28) so weekly templates have enough cells. For a 14-day window, consider bumping to 112 days.

Then think about the downstream forecaster. If it was trained on a different window, your reconstructions will not match its input spec. This is the hardest constraint to relax; the forecaster typically cannot accept an arbitrary new window size without retraining.

### Sampling frequency

Change `config.py`: `POINTS_PER_DAY` and `FREQ`.

Change `impute_cli.py`: `EXPECTED_FREQ_SECONDS = 600` (line 17) and `FREQ_TOLERANCE_SECONDS = 1`.

Change `smart_imputation.py`: the `STEPS_PER_DAY` module constant (line 47), plus all the magic numbers in the routing cascade (`<= 3`, `<= 18`, `> 50`, `<= STEPS_PER_DAY`). Those represent time durations at 10-minute resolution; re-derive them for the new frequency.

Change `window.py`: the `freq="10min"` literal in the `pd.date_range` calls.

Change `plot_reconstruction.py`: x-axis tick density (`HourLocator(byhour=range(0, 24, 6))`).

Verify the remaining magic numbers with grep: `grep -rn "'10min'\|\"10min\"\|\"10 min\"\| 600\| 144\| 1008" Imputation-Module/src/`. Every hit is a suspect.

## 7.3 Swap the data source

`cassandra_client.py` is the entire data-source seam. Its three functions are the contract:

```python
def load_historical_data_cassandra() -> pd.DataFrame:
    """Returns: DataFrame with 'Date' (tz-naive or tz-aware) plus one numeric
    column per building in config.BUILDINGS (minus Ptot_Campus, which is
    materialised on the fly)."""

def load_weather_data_cassandra() -> pd.DataFrame:
    """Returns: DataFrame with 'Date' and 'AirTemp' (°C)."""

def write_reconstructed_window(building: str, timestamps, values, quality):
    """Upsert 1008 rows: (partition_key, timestamp, value, quality_flag) for
    the named building. Must leave sibling buildings' columns untouched."""
```

To swap Cassandra for Postgres, InfluxDB, Parquet-on-S3, or anything else: rewrite the three function bodies to use the new driver, keeping the signatures. Rename the file if you want (for example `postgres_client.py`), but then update the two imports in `impute_cli.py` (inside `load_cassandra_window` and the `write_reconstructed_window` call near the end of `main`). Update `config.py` with new connection env vars and drop the unused `CASSANDRA_*` ones. Update `docker-imputation/requirements.txt` with the new driver (pinned). Update `docker-imputation/docker-compose.yml` env vars. And rename `schema.cql` to match the new backend's DDL.

Nothing else in the codebase references the data source. `impute_cli.py`, `scheduler.py`, `window.py`, `imputer.py`, and `smart_imputation.py` all work on in-memory DataFrames.

## 7.4 Change the schedule

### Stay with APScheduler, different time

Two env vars, no code changes:

```bash
SCHEDULE_HOUR=2 SCHEDULE_MINUTE=30 docker compose up -d
```

Both are validated at container startup by `scheduler._env_int` (0 to 23 for hour, 0 to 59 for minute). Invalid values fail fast.

### Use an external scheduler (cron, Airflow, systemd timer)

If you want to run the module from an external scheduler and drop APScheduler entirely, replace the default command in `Dockerfile` (currently `CMD ["scheduler.py"]`) with something trivial like `CMD ["sleep", "infinity"]`, or remove it and expect every invocation to be a `docker compose run`. Have the external scheduler call `docker compose run --rm imputer scheduler.py --run-now` (which reproduces the full batch semantics: per-building subprocess, stale-output wipe, failure tolerance) or `impute_cli.py` one building at a time. You can also remove `APScheduler` from `requirements.txt` if you want a lighter image; the scheduler file itself still needs `APScheduler` to import as-is, so either keep it or delete `scheduler.py` entirely and make the external scheduler responsible for the per-building loop.

The CLI is the real entry point; the scheduler is a convenience loop around it.

### Run more than once a day

`APScheduler.CronTrigger` accepts additional fields. Edit `scheduler.main` to pass `day_of_week`, `minute`, and so on. Example (every hour at minute 50):

```python
trigger = CronTrigger(minute=SCHEDULE_MINUTE, timezone=config.TIMEZONE)
```

Beware: every tick wipes `/io/reconstructed_*.csv` (`_wipe_previous_outputs`). At a sub-daily cadence the wipe would destroy earlier runs of the same day. Either adjust the wipe pattern to include a timestamp suffix, or schedule externally and pass a unique `--output` path per run.

## 7.5 Change gap-length thresholds

The routing cascade in `_intelligent_router` (`smart_imputation.py` line 1379) has four numeric thresholds:

| Threshold | Default | Location | Meaning |
|---|---|---|---|
| ML-cascade cutoff | `gap_size > 50` | line 1411 | Only gaps longer than 50 steps (about 8 h) trigger MICE/KALMAN/KNN. |
| Linear-micro cutoff | `gap_size <= 3` | line 1444 | Up to 30 min goes to `LINEAR_MICRO`. |
| Linear-short cutoff | `gap_size <= 18` | line 1448 | Up to 3 h goes to `LINEAR_SHORT`. |
| Sub-day cutoff | `gap_size <= STEPS_PER_DAY` | line 1457 | Up to 1 day on non-entry goes to `THERMAL_TEMPLATE`, above goes to `ENHANCED_TEMPLATE`. |

Change the constants in `_intelligent_router`, then update the routing decision tree in [`06-algorithms.md`](06-algorithms.md#62-routing-decision-tree) and the "rough mnemonic" in [`03-data-model.md`](03-data-model.md#314-quality-flag-encoding). Re-run the test harness against a known-good night: tighter thresholds route more gaps through templates (flag 2), looser thresholds route more through linear (flag 1). The MAE and RMSE distribution across flags tells you whether the move was an improvement.

## 7.6 Add a new imputation strategy

Implement the fill method on `ExtendedDeploymentAlgorithm` in `smart_imputation.py`. Follow the existing signature: `_fill_my_strategy(self, df, site, gap_start, gap_end)` returning truthy on success, falsy on failure. Mutate `df.loc[gap_start:gap_end-1, site]` in place.

Wire it into the router (`_intelligent_router`). Decide where it sits: as an ML-cascade alternative, a peer-fallback, a new branch in the size-based cascade, or something else. Add a `routing_trace.append('TRY: MY_STRATEGY')` and `_log_and_record('MY_STRATEGY')` block.

Add the flag mapping in `imputer.py`: `_STRATEGY_FLAG_MAP["MY_STRATEGY"] = 2` (or 1, or 3). If you forget, `_flag_for_strategy` defaults to 2 (safe but silently misleading).

Decide on a feature flag. Every existing ML strategy has a boolean toggle in `ExtendedDeploymentAlgorithm.__init__` (`use_knn`, `use_mice`, and so on) and is gated by it in the router. Follow the pattern so operators can disable the new strategy without a code change.

Update the documentation: the flag table in [`03-data-model.md`](03-data-model.md#314-quality-flag-encoding), a subsection in [`06-algorithms.md`](06-algorithms.md#63-fill-strategies), and the routing tree in [`06-algorithms.md`](06-algorithms.md#62-routing-decision-tree) if the cascade order changed.

Test with the harness: add synthetic gaps that the new strategy should win, run with `--test-gap` and `--test-report`, and confirm the `# strategy=MY_STRATEGY` line appears in the report footer.

## 7.7 Change the output destination

The output flow is two writes on success: the output CSV at `--output` (always), then the Cassandra upsert to `conso_historiques_reconstructed` (Cassandra mode only, after the CSV is written).

To write somewhere else: for a different table or keyspace, change `RECONSTRUCTED_TABLE` and `RECONSTRUCTED_PARTITION_KEY` in `config.py` and update `schema.cql` accordingly. For a different Kafka or HTTP sink, rewrite `cassandra_client.write_reconstructed_window`; its contract is "commit 1008 rows for one building, leaving siblings on the same `Date` untouched". For a sink without a partial-update concept (like Kafka), decide whether a message is one building's full series or one timestamp's all-building reading, and match the consumer's expectations. To write to both Cassandra and Kafka, add a second write call in `impute_cli.main` after the existing `write_reconstructed_window` call; guard it behind a CLI flag or env var.

The output CSV is always produced, regardless of whether the Cassandra write succeeds or fails: a failed Cassandra write leaves the CSV on disk for manual ingestion or retry.

## 7.8 Change the Cassandra schema

Any change to `conso_historiques_reconstructed` requires editing `Imputation-Module/cassandra/schema.cql` as the authoritative DDL, then running `ALTER TABLE` on the live cluster (the `CREATE TABLE IF NOT EXISTS` in the file does not re-apply changes; it is a no-op once the table exists), and updating `BUILDING_TO_RECONSTRUCTED_COLUMNS` in `config.py` to name the new columns. If you added columns, `cassandra_client.write_reconstructed_window`'s CQL already uses named columns and no code change is needed. If you renamed columns, `BUILDING_TO_RECONSTRUCTED_COLUMNS` is the only place to update.

Also update [`03-data-model.md`](03-data-model.md#313-conso_historiques_reconstructed-output) with the new DDL, and coordinate with downstream consumers who read from this table.

## 7.9 Invariants to keep in lockstep

A small number of invariants span multiple files. Break any of them and either the validator in `impute_cli.py` rejects the input up front, or something subtler goes wrong at runtime.

`config.BUILDINGS` must match `config.BUILDING_TO_RECONSTRUCTED_COLUMNS` keys, which must match the `Ptot_*` and `quality_*` columns in `cassandra/schema.cql`, which must match the `BUILDINGS` array in `run-all-buildings.sh`.

`config.POINTS_PER_DAY × config.LOOKBACK_DAYS == config.LOOKBACK_POINTS == impute_cli.EXPECTED_ROWS`.

`config.FREQ == "10min"` must match the hard-coded `"10min"` strings inside `window.extract_window` and `ExtendedDeploymentAlgorithm.impute`, and `impute_cli.EXPECTED_FREQ_SECONDS == 600`.

`config.TIMEZONE` must match the `TZ` env var in `docker-compose.yml` and `smart_imputation.SITE_TZ`.

`imputer._STRATEGY_FLAG_MAP` keys must include every strategy name that `smart_imputation._intelligent_router` can emit. Missing keys do not crash (unknown strategies default to flag 2) but the flag becomes inaccurate.

Any time you change one of these, grep for the literal value to find every mirror.

## 7.10 Dependency pinning

Every `requirements.txt` in the repo uses exact `==` version pins, no `>=`, no `~=`, no unpinned names. This is a non-negotiable reproducibility rule for the whole project.

When adding a dependency: pin it with `==`; pin its transitive closure too, using `pip freeze` or `pip-compile` to produce the complete list (unpinned transitive dependencies silently drift on every rebuild); test the new pin-set against the test harness (`--test-gap` with known gaps) before merging; and keep the pins consistent across the image rebuild (if two different CI jobs install different transitive versions, the image is not reproducible).

Python 3.7 is the target interpreter (see `Dockerfile`). Moving to a newer Python typically requires re-pinning everything, because most of the current pins pre-date their Python-3.10+ wheels.
