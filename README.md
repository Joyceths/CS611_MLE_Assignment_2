# CS611 Assignment 2 – Loan Default ML Pipeline

Monthly training, inference and monitoring of a loan default model on Airflow, on top of the Assignment 1 datamart.

## Run
```bash
docker-compose build
docker-compose up            # Airflow UI on http://localhost:8080 (admin / admin)
```
The DAG `loan_default_ml_pipeline` is un-paused at start-up and backfills 2023-01 → 2025-07 (31 monthly runs, ~1 h).

## Layout
| Path | Purpose |
|---|---|
| `dags/dag.py` | one monthly run: data layers, training check, inference, monitoring, retrain check |
| `utils/data_processing_*_table.py` | bronze / silver / gold tables from Assignment 1 |
| `utils/task_data_pipeline.py` | runs one data layer for one month (used by the DAG) |
| `utils/model_governance.py` | retrain triggers |
| `utils/model_train.py` | candidates, evaluation, promotion decision |
| `utils/model_inference.py` | scores the month's applications |
| `utils/model_monitor.py` | stability, performance and calibration checks |
| `utils/plot_monitoring.py` | charts and summary (last run only) |
| `utils/config.py` | dates and all thresholds |

## One monthly run
```
bronze -> silver -> gold (label + feature store)
  -> training_check_before_scoring -> train_before_scoring | skip   (initial model or feature outage)
  -> model_inference -> model_monitoring
  -> retrain_check -> train_after_monitoring | no_retrain           (regular degradation, live next month)
  -> monitoring_visualisation (last month)
```
Monthly batch: the month's applications are collected, the feed coverage of the batch is checked, a model is trained
if a trigger fires (matured labels only), then the batch is scored. Scoring each application on arrival would not
allow this order, because the month's coverage is not known yet.

## Outputs
| Output | Location |
|---|---|
| label / feature store | `datamart/gold/label_store`, `datamart/gold/feature_store` |
| predictions | `datamart/gold/model_predictions` |
| monitoring | `datamart/gold/model_monitoring`, `datamart/gold/model_monitoring_csi` |
| models and registry | `model_bank/*.pkl`, `model_bank/registry.json` |
| charts and summary | `monitoring_report/` |

## Leakage controls
* A loan applied in month M is labelled in M+6, so training uses cohorts up to run date − 6 months only (asserted in `model_train.py`).
* Train < validation (2 cohorts) < out-of-time, OOT (2 cohorts) in time order (asserted).
* Scaler and model are fit on training rows only, inside one sklearn Pipeline. The gold layer only fills nulls with 0 and adds `_missing` flags.
* Clickstream features only use months up to the application month.

## How the splits are used
* validation picks the algorithm, parameters and decision threshold; the choice is refit on train + validation
* OOT is used for the final evaluation, the floor checks and the champion / challenger comparison, so the OOT AUC of a promoted model is slightly optimistic
* the AUC from monitoring (cohorts scored before their labels existed) is the forward-looking figure

## Governance
Thresholds are in `utils/config.py`. They were fixed before the final run, but after earlier results had been seen.

| Item | Rule |
|---|---|
| feed outage | batch coverage of a feature group < 90%; ends when coverage is ≥ 90% again |
| champion depends on a group | the group is in the model's input list |
| initial | no model in production (from 2024-04); retried every 2 months if it does not qualify |
| feature outage | champion depends on a group in outage; first month of an outage trains at once, later months after 2 months since the last attempt; live the same month |
| regular degradation | monitoring RED, ≥ 6 new cohorts and ≥ 2 months since the last attempt; live next month |
| attempts | new cohorts and spacing are counted from the last attempt, promoted or not |
| last effective date | a model is never effective after the last application month (2025-01) |
| promotion floor | OOT AUC ≥ 0.75 and calibration gap ≤ 0.05; first model failing it is not live and gets no scores |
| promotion gain | OOT AUC at least 0.005 above the champion; for an outage update the champion is stress-tested (missing group set to 0 and `has_clickstream` = 0) |
| no qualified replacement | champion is kept; while its inputs are degraded, predictions carry `input_status = degraded_review` |

## Monitoring
* **Stability** (same month): score PSI and feature CSI. Alerts use PSI vs the OOT reference; PSI vs the training population is stored but never alerts. Bands 0.10 / 0.25.
* **Performance** (6 months later): AUC, Gini, KS, Brier and calibration gap of the cohort whose labels arrive this month. AUC drop vs the model's OOT AUC: amber 0.03, red 0.06. Calibration gap: amber 0.05, red 0.10.
* Overall status is the worst of the three. PSI is not used to compare model versions; compare them on the same loans (AUC, calibration, Brier).
