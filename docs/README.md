# LiveTree R&D — Documentation

This folder is the canonical reference for **Project N°28 — LiveTree Demonstrator**, covering both the **Imputation Module** (data-quality orchestration layer) and the **Prediction Model** (neural-network day-ahead forecaster) that together form the production pipeline.

It is written so a reader with **no prior exposure to the project** can follow every decision from first principles. No knowledge of the codebase, JUNIA's infrastructure, or time-series machine learning is assumed.

---

## How to read these docs

Read the files in the order below. Each one builds on the previous ones; later sections refer back by filename.

| # | File | What you learn |
|---|------|----------------|
| 1 | [`01-overview.md`](01-overview.md) | Problem statement, actors, what the system produces, where it runs |
| 2 | [`02-architecture.md`](02-architecture.md) | Repo layout, component map, and how data flows between Cassandra, the imputer, the predictor, and Kafka |
| 3 | [`03-data-model.md`](03-data-model.md) | Cassandra table schemas, CSV layouts, Kafka Avro schema, timezone conventions |
| 4 | [`imputation/`](imputation/README.md) | Deep dive into the imputer: files, algorithms, routing rules, quality flags, behaviour scenarios, audit log, deployment |
| 5 | [`prediction/`](prediction/README.md) | Deep dive into the predictor: files, training, inference, DST handling, Kafka publishing, deployment |
| 6 | [`glossary.md`](glossary.md) | Every term, acronym, and internal label used across the system |

Two top-level files already exist at the repository root and are **not duplicated** here:

- [`../README.md`](../README.md) — quick start for running the stack locally.
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — system-level architecture reference maintained by the platform team.

This `docs/` folder goes **deeper** than those two: file-by-file walks, algorithmic derivations, and Q&A-oriented scenarios.

---

## Diagrams

All diagrams in this folder are written as [Mermaid](https://mermaid.js.org/) fenced code blocks and render directly on GitHub, in most Markdown viewers, and in the jury demo notebook. No external image files are used.

---

## When to update these docs

Update the matching file whenever you change:

- A Cassandra table, Kafka schema, or CSV column layout → `03-data-model.md`.
- A routing rule, fill strategy, or confidence formula in `smart_imputation.py` → `imputation/03-algorithm-deep-dive.md` and `imputation/04-strategy-routing.md`.
- A quality-flag mapping in `imputer.py` → `imputation/04-strategy-routing.md`.
- The training schedule, feature list, or model architecture in `ConsoFile.py` → `prediction/03-training.md`.
- The Docker image, environment variables, or nightly schedule → `imputation/07-deployment.md` or `prediction/06-deployment.md`.

Keep the glossary in sync whenever a new term is introduced.
