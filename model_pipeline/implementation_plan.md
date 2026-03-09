# SavVio Model Pipeline — Implementation Plan (v3)

**Date:** March 2026  
**Status:** In Progress  

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Remaining Work — Priority Order](#2-remaining-work--priority-order)
3. [Task Details](#3-task-details)
4. [Config Additions Needed](#4-config-additions-needed)
5. [Model Storage Strategy](#5-model-storage-strategy)
6. [MLflow Strategy](#6-mlflow-strategy)
7. [CI/CD Pipeline](#7-cicd-pipeline)
8. [Testing Strategy](#8-testing-strategy)
9. [Dependency Chain](#9-dependency-chain)

---

## 1. Current State

### Implemented

| Component | File | Notes |
|-----------|------|-------|
| Pipeline entrypoint | `run_pipeline.py` (v2) | Refactored — 5 functions, 3-way split, bias/LLM placeholders |
| Model training | `train.py` (v2) | Param filtering, eval_set, early stopping, LogReg support |
| Evaluation | `evaluate.py` (v2) | Full visualizations, per-class metrics, MLflow artifacts |
| Hyperparameter tuning | `tuning/optuna_tuner.py` | Optuna + MLflow callback, 3 model types |
| Feature engineering | `feature_engineering.py` | Scenario generation, encoding, scaling |
| Scenario generation | `training_data_generator.py` | Stratified sampling, deterministic labeling |
| Affordability features | `affordability_features.py` | 6 computed financial features |
| Deterministic engine | `decision_logic.py` | 4 RED rules, 5 YELLOW rules, GREEN default |
| Config | `config.py` | Paths, MLflow, feature lists, scenario settings |
| Requirements | `model-requirements.txt` | All dependencies listed |

### In Progress

| Component | Owner |
|-----------|-------|
| Bias detection (`guards/bias_detection.py`) | In progress |
| LLM wrapper (`llm/prompt_engin.py`) | In progress |

### Not Started

| Component | Priority |
|-----------|----------|
| Integrate Optuna into `run_pipeline.py` | P1 |
| Add tuning config to `config.py` | P1 |
| SHAP / LIME explainability | P2 |
| Model registry push to GCP | P2 |
| CI/CD GitHub Actions workflow | P2 |
| Unit tests | P1 |
| Docker volume mount for MLflow | P3 |
| Clean up `experiments/` folder | P3 |

---

## 2. Remaining Work — Priority Order

### Priority 1 — Required for Complete Pipeline

| # | Task | Effort |
|---|------|--------|
| 1 | Add tuning config to `config.py` (`TUNING_BACKEND`, `N_TUNING_TRIALS`, `TUNING_TIMEOUT_SECONDS`) | 10 min |
| 2 | Integrate `tune_best_candidate()` into `run_pipeline.py` after baseline training | 30 min |
| 3 | Remove `linearboost` from `model-requirements.txt` (unused — code uses `XGBClassifier(booster='gblinear')`) | 5 min |
| 4 | Finish bias detection module | In progress |
| 5 | Unit tests for decision engine (100% branch coverage), training, evaluation, tuning | 3–4 hours |

### Priority 2 — Strong Submission

| # | Task | Effort |
|---|------|--------|
| 6 | SHAP / LIME explainability — add to `final_evaluation()` in `run_pipeline.py` | 12–13 hours |
| 7 | Model registry push — `mlflow.register_model()` + GCP Artifact Registry upload script | 2 hours |
| 8 | CI/CD GitHub Actions — test → train → validate gate → bias gate → registry push | 3–4 hours |
| 9 | Finish LLM wrapper | In progress |

### Priority 3 — Polish

| # | Task | Effort |
|---|------|--------|
| 10 | Docker volume mount for MLflow persistence | 30 min |
| 11 | Optuna dashboard setup (`pip install optuna-dashboard`) | 15 min |
| 12 | Clean up `experiments/` folder — use it for notebooks or remove it | 10 min |
| 13 | Add `models/` and `artifacts/` to `.dockerignore` and `.gitignore` | 5 min |

---

## 3. Task Details

### 3.1 Integrate Optuna into run_pipeline.py

Add a tuning step between `train_candidates()` and `select_best_model()` in `main()`. Call `tune_best_candidate()`, train a final model with the returned params, evaluate it on the validation set, and append it to the candidates list. Guard with `Config.TUNING_BACKEND != "none"`.

### 3.2 SHAP / LIME Explainability

Create `explainability/shap_lime.py` with two functions:

- `generate_shap_explanations(model, X_test, label_names, save_dir)` — global summary plot + 3–5 force plots per class.
- `generate_lime_explanations(model, X_test, label_names, save_dir)` — 3–5 local explanations per class.

Call both from `final_evaluation()` in `run_pipeline.py`. Log all plots as MLflow artifacts. Use `shap.TreeExplainer` for tree models (fast), fall back to `shap.KernelExplainer` for Logistic Regression.

### 3.3 Model Registry Push

Two-step process:

1. **MLflow registry:** Call `mlflow.register_model(f"runs:/{run_id}/model", Config.REGISTERED_MODEL_NAME)` after final evaluation. MLflow auto-increments version numbers.

2. **GCP Artifact Registry:** Export model via `joblib.dump()`, upload via `gcloud artifacts generic upload`. Automate in CI/CD — not done manually in production.

Add `REGISTERED_MODEL_NAME = "SavVio_Purchase_Guardrail"` to `config.py`.

### 3.4 CI/CD GitHub Actions

Pipeline flow on push to `model_pipeline/`:

```
Job 1: Unit Tests
    pytest model_pipeline/tests/ --cov
    FAIL → block, notify

Job 2: Train & Validate (needs Job 1)
    dvc pull
    python run_pipeline.py
    check validation gate (F1 > threshold)
    check bias gate (disparity < threshold)
    FAIL → block, notify

Job 3: Registry Push (needs Job 2, main branch only)
    mlflow.register_model()
    gcloud artifacts upload
    notify team
```

Gate thresholds (configurable in config):

| Gate | Metric | Threshold |
|------|--------|-----------|
| Validation | Weighted F1 | > 0.70 |
| Validation | ROC AUC | > 0.75 |
| Bias | Max F1 disparity | < 0.10 |
| Rollback | F1 vs. previous model | No decrease > 0.02 |

---

## 4. Config Additions Needed

Add to `config.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `TUNING_BACKEND` | `"optuna"` | `"optuna"` or `"none"` to skip |
| `N_TUNING_TRIALS` | `50` | Max Optuna trials |
| `TUNING_TIMEOUT_SECONDS` | `600` | Safety timeout for study |
| `REGISTERED_MODEL_NAME` | `"SavVio_Purchase_Guardrail"` | MLflow model registry name |
| `MIN_F1_THRESHOLD` | `0.70` | Validation gate |
| `MIN_AUC_THRESHOLD` | `0.75` | Validation gate |
| `BIAS_DISPARITY_THRESHOLD` | `0.10` | Bias gate |

---

## 5. Model Storage Strategy

| Environment | Where | Format | Versioning |
|-------------|-------|--------|------------|
| Local dev | `model_pipeline/artifacts/` (Config.MODEL_SAVE_DIR) | joblib | Git-ignored, ephemeral |
| MLflow (dev) | MLflow artifact store (Docker volume) | MLflow native | MLflow run ID + version |
| GCP (prod) | GCP Artifact Registry | joblib | Registry version tag |

- `artifacts/` goes in `.gitignore` and `.dockerignore`.
- No DVC for model artifacts — DVC is for data inputs. Models are versioned by MLflow + GCP.
- Preprocessing artifacts (encoder, scaler) are logged to MLflow alongside the model for deployment reproducibility.

---

## 6. MLflow Strategy

### Experiment Organization

```
Experiment: Financial_Wellbeing_Prediction
├── Run: xgboost_baseline         (params, val metrics, plots, model)
├── Run: lightgbm_baseline        (params, val metrics, plots, model)
├── Run: xgb_linear_baseline      (params, val metrics, plots, model)
├── Run: logistic_regression_baseline (params, val metrics, plots, model)
├── Run: xgboost_tuning_trial_001 (auto-logged by Optuna callback)
├── Run: xgboost_tuning_trial_002
├── ...
├── Run: xgboost_tuned            (tuned model, val metrics, plots)
└── Run: FINAL_xgboost            (test metrics, all visualizations, SHAP/LIME)
```

### What Gets Logged Per Run

| Category | Items |
|----------|-------|
| Params | model_type, hyperparams, n_scenarios, random_state, label_type, num_classes |
| Aggregate Metrics | accuracy, f1_score, roc_auc, pr_auc |
| Per-Class Metrics | GREEN_f1, YELLOW_f1, RED_f1, GREEN_precision, etc. |
| Bias Metrics | bias_gate_passed, per-slice disparity values |
| Artifacts | model binary, confusion_matrix.png, roc_curves.png, pr_curves.png, calibration_curves.png, classification_report.txt, encoder.pkl, scaler.pkl, scenarios.csv |

### Local MLflow with Docker Volume

Mount a named volume so MLflow data persists across container rebuilds. The MLflow server command should point `--backend-store-uri` to a SQLite file on the volume and `--default-artifact-root` to a directory on the same volume.

---

## 7. CI/CD Pipeline

### Architecture

```
GitHub Push / PR on model_pipeline/
    ├── Job 1: pytest (unit tests + coverage)
    ├── Job 2: Train → Validate → Bias gate
    ├── Job 3: Rollback check (compare vs. previous registered model)
    └── Job 4: Registry push (main branch only, all gates pass)
```

### Notifications

- Pipeline failure → Slack / email
- Bias gate failure → Slack / email with disparity report
- Successful push → Slack confirmation with model version

---

## 8. Testing Strategy

| Test File | Module | Key Test Cases | Coverage Target |
|-----------|--------|----------------|-----------------|
| `test_decision_logic.py` | Deterministic engine | All 4 RED rules, all 5 YELLOW rules, GREEN default, edge cases (NaN, missing fields, zero income), boundary values | 100% branch |
| `test_feature_engineering.py` | Feature engineering | Missing value imputation, encoding, scaling, build_feature_matrix output shape, label distribution balance | 90%+ |
| `test_training_data_generator.py` | Scenario generation | Stratified vs random sampling, feature computation, label assignment, empty bracket handling | 90%+ |
| `test_train.py` | Model training | All 4 model types, param filtering (invalid params dropped), early stopping triggers, seed reproducibility | 85%+ |
| `test_evaluate.py` | Evaluation | Metric computation, multi-class AUC, plot generation, per-class metric logging, binary edge case | 85%+ |
| `test_optuna_tuner.py` | Tuning | Study creation, objective function, unsupported model error, timeout behavior | 80%+ |
| `test_bias.py` | Bias detection | Slice computation, disparity thresholds, pass/fail logic | 90%+ |
| `conftest.py` | Shared fixtures | Sample DataFrames for financial, product, scenario data; mock MLflow |  |

---

## 9. Dependency Chain

```
Add tuning config to config.py
        ↓
Integrate Optuna into run_pipeline.py
        ↓
Finish bias detection ──────────────────┐
        ↓                               ↓
Unit tests (all modules)        SHAP / LIME explainability
        ↓                               ↓
CI/CD GitHub Actions ←──────────────────┘
        ↓
Model registry push (MLflow + GCP)
        ↓
LLM wrapper (stretch)
        ↓
Docker volume mount, cleanup (polish)
```

Critical path: Config additions → Optuna integration → Bias detection → Unit tests → CI/CD.

---

*End of Implementation Plan v3*
