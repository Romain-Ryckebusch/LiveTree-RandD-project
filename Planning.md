## Week 2 (Mar 23-29) -- Core Development

### Mike & Isaac (AI/Big Data)
- **[A] Improve the imputation algorithm** :
  - Evaluate and improve the seasonal backup profile (better weighting, clustering similar days, etc.)
  - Explore ML-based gap filling for medium/long gaps
  - Benchmark different imputation strategies against the simple baseline
- **[B] Adapt single-building prediction** -- verify the prediction algorithm works end-to-end with the Ptot_HA model and scalers

### Arda & Romain (DevOps/SRE)
- **Arda: Set up Jira** for the project (board, backlog, sprint structure)
- **[C] Set up the demo pipeline** -- make a locally runnable demo
  - Ensure CSV loading works with actual data files
  - Validate gap injection produces realistic test scenarios
- **[D] Build evaluation framework**
  - Implement MAE/RMSE/MAPE comparison
  - Add basic plotting (baseline vs imputed predictions)
- **Set up reproducible environment**

---

## Week 3 (Mar 31 - Apr 5) -- Integration & Iteration

### Mike & Isaac
- **[A] Iterate on the improved imputation** based on evaluation results
  - Test with different gap sizes (1h, 6h, 12h, 24h, 48h)
  - Compare ML-enhanced imputation vs. simple baseline on prediction quality
  - Document how prediction quality degrades with gap size
- **[B] Run systematic experiments**: vary gap patterns, measure impact on prediction MAE
- **Begin writing technical sections** of the deliverable (methodology, algorithm description, ML choices)

### Arda & Romain
- **[C] Dockerize the demo pipeline** -- create `Dockerfile` + `docker-compose.yml`
- **[D] Automate test scenarios** -- script that runs multiple gap configurations and produces a summary table
- **Prepare integration path**: document how the module would plug into the production ConsoFile.py + Kafka pipeline

---

## Week 4 (Apr 7-12) -- Testing & Validation

### All together
- **Run full validation suite**: multiple dates, multiple gap patterns, edge cases
- **Fix bugs and edge cases** found during validation
- **Collect results**: summary tables + plots for the deliverable
- **Mike & Isaac**: write the "Results" section of the deliverable
- **Arda & Romain**: write the "Architecture" and "Methodology" sections, installation guide

---

## Week 5 (Apr 14-19) -- Finalization

### All together
- **Finalize deliverable**: merge all sections, proofread, ensure reproducibility instructions are clear
- **Prepare defense presentation** (20 min, split equally among 4 members):
  - Romain: Project context + problem statement + architecture
  - Arda: Methodology + testing approach + DevOps choices
  - Mike: Imputation algorithm + design decisions
  - Isaac: Results + evaluation + future perspectives

---

## Week 6 (Apr 21-24) -- Submission & Defense

- Submit deliverable
- Final defense (20 min presentation + 15 min questions)