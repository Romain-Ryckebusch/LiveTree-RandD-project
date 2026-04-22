# Documentation

This folder is the canonical reference for the LiveTree Imputation Module. It covers what the project does, how the code is organised, how to run it, and how to modify it. A reader with no prior exposure to the project should be able to start here and get to a mental model of the whole system, then drop into the code.

The root [`../README.md`](../README.md) is intentionally thin: it points at the chapters below. Everything else lives here.

## Reading order

The chapters build on each other. Read them in order for the first pass; after that, use the titles as a reference index.

| # | File | What you learn |
|---|------|----------------|
| 1 | [`01-overview.md`](01-overview.md) | What the project is, why it exists, what it produces, what it does not do. |
| 2 | [`02-architecture.md`](02-architecture.md) | Repository layout, component responsibilities, end-to-end data flow, design decisions, runtime topology. |
| 3 | [`03-data-model.md`](03-data-model.md) | Cassandra table schemas, CSV contracts, quality-flag encoding, timezone conventions. |
| 4 | [`04-usage.md`](04-usage.md) | How to run the imputer: Docker build, daemon mode, CSV mode, Cassandra mode, test harness, configuration, Python import. |
| 5 | [`05-file-reference.md`](05-file-reference.md) | Every file in the repo, one by one: purpose, public surface, key functions, when to edit it. |
| 6 | [`06-algorithms.md`](06-algorithms.md) | How gaps get filled: routing logic, gap-length cascade, templates, donor-day ensemble, DST handling, audit logging. |
| 7 | [`07-extending.md`](07-extending.md) | How to modify the project: add a building, change the schedule, swap the data source, add a strategy, change gap thresholds. |
| 8 | [`08-glossary.md`](08-glossary.md) | Domain terms, building identifiers, naming quirks, acronyms. |

## Shortcut by intent

| You want to... | Start at |
|---|---|
| Run the imputer on a CSV or against Cassandra | [`04-usage.md`](04-usage.md) |
| Understand what the project does and why | [`01-overview.md`](01-overview.md) → [`02-architecture.md`](02-architecture.md) |
| Find which file implements a given function or flag | [`05-file-reference.md`](05-file-reference.md) |
| Understand how a gap is actually filled | [`06-algorithms.md`](06-algorithms.md) |
| Adapt the project to a different site, building list, or data store | [`07-extending.md`](07-extending.md) |
| Look up a Cassandra column, a CSV field, or a quality-flag value | [`03-data-model.md`](03-data-model.md) |
| Look up a term or acronym | [`08-glossary.md`](08-glossary.md) |

## Keeping the docs in sync with the code

When you change a piece of code, update the matching chapter. The mapping is:

| You changed... | Update |
|---|---|
| A Cassandra table, CSV column, or quality-flag value | [`03-data-model.md`](03-data-model.md) |
| `impute_cli.py` argparse definitions | [`04-usage.md`](04-usage.md) flag tables |
| `Imputation-Module/src/*.py` structure (file added, file renamed, public function signature changed) | [`05-file-reference.md`](05-file-reference.md) |
| `smart_imputation.py` routing thresholds, strategies, or `_STRATEGY_FLAG_MAP` in `imputer.py` | [`03-data-model.md`](03-data-model.md) (flag table) and [`06-algorithms.md`](06-algorithms.md) |
| `config.py` constants, env vars, or building list | [`03-data-model.md`](03-data-model.md), [`04-usage.md`](04-usage.md), [`07-extending.md`](07-extending.md) |
| Docker image, `docker-compose.yml`, or the nightly schedule | [`04-usage.md`](04-usage.md) |
| A new domain term or acronym | [`08-glossary.md`](08-glossary.md) |

If you change code and do not know which chapter covers it, grep this folder for a keyword and update wherever it lands.
