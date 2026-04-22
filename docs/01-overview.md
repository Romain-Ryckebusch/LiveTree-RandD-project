# 1. Overview

## 1.1 What this project is

This repository is the *module de pilotage*, the steering module of a day-ahead electricity-consumption forecasting system for JUNIA's LiveTree demonstrator campus in Lille, France. It is part of Project N°28, *"Amélioration de la résilience des modèles prédictifs par la qualité et la continuité des données"*.

Four buildings on the campus are instrumented with electrical meters that publish a power reading every ten minutes into a Cassandra cluster. A forecasting service in a separate system reads the last seven days of those readings (exactly 1008 points per building: 7 × 144) and produces a 24-hour-ahead forecast. The forecaster does not tolerate gaps in its input. A single missing row is enough to block the prediction entirely.

The acquisition chain is not reliable enough to guarantee a gap-free seven-day window. Sensors drop out, network links fail, and multi-hour outages are routine. This module sits between the raw Cassandra table and the forecaster: it takes a seven-day window with holes, returns a complete one, and tags every point with a quality flag so downstream consumers can decide how much to trust each value.

The data flow is a straight line, left to right:

```
Building sensors  ->  Cassandra                   ->  This repo          ->  Cassandra                          ->  Downstream forecaster
(10-min readings)     (conso_historiques_clean)       (Imputation Module     (conso_historiques_reconstructed)      (separate system,
                                                       nightly 23:50                                                 out of scope)
                                                       Europe/Paris)
```

The downstream forecaster lives in a different repository and runs as a separate container. Its model, Kafka publisher, and scheduler are not in this repo and are not documented here.

## 1.2 Why it exists

The forecaster's input is rigid: exactly 1008 rows per building, on a 10-minute grid, strictly increasing, covering the seven days ending yesterday (the forecast is for tomorrow), with no NaNs anywhere.

The legacy pipeline solved this by dropping any window with more than about 150 missing rows and doing naive linear interpolation on the rest. So whole days of the forecast were silently skipped whenever the acquisition chain had a bad night, and even on the windows that did make it through, linear interpolation across a multi-hour gap that straddles a weekend, a holiday, or a sharp thermal transition produces a shape that has nothing to do with the building's actual consumption pattern.

This module replaces both behaviours. It reconstructs any gap size from one missing point up to the entire 1008-point window, and it routes each gap to a strategy that matches its context: day of the week, occupancy, outside temperature, which sibling buildings are intact, whether a recent donor day is available.

## 1.3 What it produces

For each building, for each nightly run, the module writes a 1008-row reconstructed window (with `value` and `quality` columns) to Cassandra `conso_historiques_reconstructed` and to a CSV at `/io/reconstructed_<building>_<target_date>.csv`. A matching PNG overlay at `/io/reconstructed_<building>_<target_date>.png` is produced when `--plot` is passed. A per-run audit log lands as JSON under `/io/audit_logs/`.

Each row in the output carries a quality flag describing how its value was obtained:

| Flag | Meaning |
|---|---|
| `0` | Real measurement, not imputed |
| `1` | Linear interpolation (short gap) |
| `2` | Contextual fill (templates, peer correlation, safe-median blend) |
| `3` | Donor-day ensemble or ML fallback (long gap) |

Downstream consumers typically filter on `quality ≤ 1` for high-confidence analysis, relax to `quality ≤ 2` when they trust the contextual templates, and treat `quality = 3` as a plausible shape rather than a measurement. The authoritative mapping from internal strategy names to flags is `_STRATEGY_FLAG_MAP` in `Imputation-Module/src/imputer.py`, reproduced in [`03-data-model.md`](03-data-model.md).

## 1.4 Buildings covered

Five columns go through the pipeline: four physical buildings plus one virtual aggregate.

```
Ptot_Campus  (virtual: sum of the four physical meters below)
│
|--- Ptot_HA         Hôtel Académique
|--- Ptot_HEI_13RT   13 Rue de Toul
|--- Ptot_HEI_5RNS   5 Rue Nicolas Souriau
|--- Ptot_RIZOMM     RIZOMM building
```

The four physical meters are `Ptot_HA`, `Ptot_HEI_13RT`, `Ptot_HEI_5RNS`, and `Ptot_RIZOMM`. Each has a column in `conso_historiques_clean` and a corresponding column in the reconstructed output table.

The virtual aggregate is `Ptot_Campus`. It is not stored in the clean table; the imputer materialises it on the fly as the sum of the four physical columns with `skipna=False`. If any component is missing at a given timestamp, the campus total is also missing at that timestamp and gets imputed like any other gap. This is deliberate: it prevents the campus aggregate from silently under-counting one building.

The building list is defined once in `Imputation-Module/src/config.py` as `BUILDINGS`. Every other part of the codebase reads from there.

## 1.5 Where it runs

In production, the module runs as a Docker container (`rd_imputer`, Python 3.7 slim) alongside the rest of the LiveTree stack. A long-lived APScheduler daemon wakes once per night at 23:50 Europe/Paris, spawns one CLI subprocess per building, reads the raw seven-day window from Cassandra, runs the imputation, and writes the reconstructed window back to `conso_historiques_reconstructed` plus a CSV (and optional PNG) under `/io/`.

The container joins an external Docker network (`cassandra_default`, declared as `cassandra_net` with `external: true` in the compose file) that is assumed to already exist on the host. The Cassandra cluster itself is managed outside this project.

Locally and for one-off reconstructions, the CLI (`impute_cli.py`) can be invoked directly in either CSV mode (no Cassandra needed) or Cassandra mode. The full run-it-yourself guide is in [`04-usage.md`](04-usage.md).

## 1.6 Non-goals

The forecasting model itself is out of scope. This repo only produces the input the forecaster needs; the model lives elsewhere.

Raising sensor-fault tickets upstream is out of scope. Gap detection is reactive: the module reconstructs whatever it is given and records the strategy it used. It does not call out to an alerting system when a sensor dies.

Retraining the forecaster on reconstructed data is out of scope. The forecaster is trained on raw measurements only, to avoid a feedback loop. This module writes to a separate table so the training job can ignore it.

Real-time streaming is out of scope. The pipeline runs once per night, and the unit of work is always a full seven-day window. Sub-minute latency is not a goal.

Multi-site generalisation is out of scope of the current code, which is wired to the five JUNIA building identifiers. See [`07-extending.md`](07-extending.md) for how to adapt it to a different site.
