# LiveTree Imputation Module

Project N°28, JUNIA LiveTree demonstrator. This repository is the *module de pilotage*: a data-imputation service that reconstructs gap-free 7-day windows of 10-minute electricity-consumption readings for the downstream day-ahead forecaster.

Four JUNIA buildings are metered every ten minutes. The forecaster needs exactly 1008 points per building per day (7 × 144) with no NaNs anywhere. The acquisition chain drops rows often enough that the raw window is rarely complete. This module sits between Cassandra and the forecaster, fills the gaps using a gap-length-aware cascade of strategies (linear, templates, peer correlation, donor-day ensemble, MICE/Kalman/KNN), and tags every point with a quality flag.

## Quick start

```bash
cd Imputation-Module/docker-imputation
docker compose build
docker compose up -d
docker compose logs -f imputer
```

## Documentation

All documentation lives under [`docs/`](docs/README.md). Start there.

By audience:

| You want to... | Start at |
|---|---|
| Run the imputer (Docker, CSV, Cassandra, test harness) | [`docs/04-usage.md`](docs/04-usage.md) |
| Understand what the project does and why | [`docs/01-overview.md`](docs/01-overview.md) → [`docs/02-architecture.md`](docs/02-architecture.md) |
| Find what a specific file does | [`docs/05-file-reference.md`](docs/05-file-reference.md) |
| Understand how gaps actually get filled | [`docs/06-algorithms.md`](docs/06-algorithms.md) |
| Modify the project (new building, new strategy, swap data source) | [`docs/07-extending.md`](docs/07-extending.md) |
| Look up a Cassandra column, CSV field, or quality flag | [`docs/03-data-model.md`](docs/03-data-model.md) |
| Look up a term or acronym | [`docs/08-glossary.md`](docs/08-glossary.md) |
