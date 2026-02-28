# SavVio Model Pipeline Implementation Guide

## MLOps Course: Model Development Phase

**Project:** SavVio - AI-Driven Financial Advocacy Tool  
**Team Members:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

---

## Overview

This guide covers the **Model Development Phase (Phase 3)** for SavVio, following:

- your project scoping plan (financial guardrail + product utility framing),
- and the professor's model pipeline requirements (training, validation, bias slicing, tracking, CI/CD, and registry push).

SavVio's objective is to provide responsible pre-purchase recommendations (Green/Yellow/Red) without financial hallucinations.  
Core principle: **LLM handles context; deterministic logic handles financial math.**

---

## Model Pipeline Execution Order

```
┌──────────────────────────────────────────────────────────────────┐
│                    SavVio Model Development                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load Versioned Data from Data Pipeline (DVC artifacts)       │
│                        ↓                                         │
│  2. Feature/Label Assembly for modeling                          │
│                        ↓                                         │
│  3. Train baseline models (LogReg, XGBoost, etc.)                │
│                        ↓                                         │
│  4. Hyperparameter tuning (grid/random/Bayesian)                 │
│                        ↓                                         │
│  5. Validation on hold-out set                                   │
│     (accuracy, precision, recall, F1, AUC, calibration)          │
│                        ↓                                         │
│  6. Slice-based bias detection                                   │
│     (financial/product/review slices)                            │
│                        ↓                                         │
│  7. Sensitivity/interpretability analysis                        │
│     (feature importance, SHAP/LIME optional)                     │
│                        ↓                                         │
│  8. Model selection (performance + bias constraints)             │
│                        ↓                                         │
│  9. Register best model (Artifact Registry / Vertex Registry)    │
│                        ↓                                         │
│ 10. CI/CD automation + rollback safeguards                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Tools by Phase

| Phase | Primary Tools | Alternatives | CI/CD Gate |
|------|---------------|--------------|------------|
| Data Loading | DVC, Pandas | Polars | Data version + schema check |
| Training | scikit-learn, XGBoost | LightGBM | Reproducible training run |
| Tuning | Grid/Random/Bayesian search | Optuna | Best-run tracking required |
| Validation | sklearn metrics | custom evaluators | Minimum metric thresholds |
| Bias Slicing | Fairlearn, Pandas groupby | AIF360, TFMA | Block on severe disparity |
| Sensitivity | SHAP, LIME | permutation importance | Stability report |
| Tracking | MLflow | Weights & Biases | Run metadata completeness |
| Registry Push | Vertex Model Registry / Artifact Registry | MLflow Registry | Push only on pass |
| CI/CD Automation | GitHub Actions / Cloud Build | Jenkins | Train-validate-bias pipeline |

---

## Table of Contents

1. [Phase 1: Data Loading from Pipeline](#phase-1-data-loading-from-pipeline)  
2. [Phase 2: Model Training & Baselines](#phase-2-model-training--baselines)  
3. [Phase 3: Hyperparameter Tuning](#phase-3-hyperparameter-tuning)  
4. [Phase 4: Validation & Metrics](#phase-4-validation--metrics)  
5. [Phase 5: Bias Detection with Slicing](#phase-5-bias-detection-with-slicing)  
6. [Phase 6: Sensitivity & Explainability](#phase-6-sensitivity--explainability)  
7. [Phase 7: Experiment Tracking & Model Selection](#phase-7-experiment-tracking--model-selection)  
8. [Phase 8: Model Registry Push](#phase-8-model-registry-push)  
9. [Phase 9: CI/CD Automation](#phase-9-cicd-automation)  
10. [Project Structure](#project-structure)  
11. [Runbook (End-to-End)](#runbook-end-to-end)  
12. [Testing](#testing)  
13. [Operational Risks & Guardrails](#operational-risks--guardrails)  

---

## Phase 1: Data Loading from Pipeline

### Objective
Load the latest **versioned and validated** datasets from the data pipeline output and ensure reproducible model inputs.

### Expected input artifacts

- `data-pipeline/dags/data/features/financial_featured.csv`
- `data-pipeline/dags/data/features/product_featured.jsonl`
- `data-pipeline/dags/data/features/review_featured.jsonl`

### Steps

1. Pull DVC artifacts:
   ```bash
   cd data-pipeline/dags/data
   dvc pull
   ```
2. Run schema checks before training.
3. Build model-ready tables (joins, feature selection, label construction).

### Why this matters
Training on non-versioned or partially processed data breaks reproducibility and invalidates comparisons across runs.

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| DVC | Pull versioned artifacts | LakeFS |
| Pandas | Load tabular/JSONL data | Polars |
| Great Expectations/Pandera | Data contract checks | Custom validators |

---

## Phase 2: Model Training & Baselines

### Objective
Train baseline candidates for recommendation/decision support.

### Baseline candidates

- Logistic Regression
- XGBoost (if available)
- Optional calibrated tree models

### Requirements

- Deterministic preprocessing path for model inputs
- Fixed random seeds
- Clear separation of train/validation/test (or scenario-based evaluation where applicable)

### Notes for SavVio

- Deterministic financial rules remain authoritative for affordability math.
- ML components should support ranking/classification confidence, not replace rule-based financial safety logic.

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| scikit-learn | Baseline models | statsmodels |
| XGBoost | Nonlinear baseline | LightGBM |
| NumPy | Numerical ops | JAX (advanced setups) |

---

## Phase 3: Hyperparameter Tuning

### Objective
Improve baseline performance while preserving fairness and robustness.

### Allowed tuning strategies

- Grid Search
- Random Search
- Bayesian Optimization

### Minimum logging per run

- model type + version
- hyperparameters
- split strategy
- metrics on validation set
- artifact path / run ID

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| GridSearchCV/RandomizedSearchCV | Hyperparameter search | Optuna |
| Optuna | Bayesian optimization | Hyperopt |
| MLflow | Search/run logging | Weights & Biases |

---

## Phase 4: Validation & Metrics

### Objective
Validate performance on unseen data with task-relevant metrics.

### Core metrics

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### Recommended additions

- PR-AUC (if class imbalance)
- Calibration error / reliability curves
- Confusion matrix by decision class

### Acceptance pattern

Use threshold gates (example):

- Minimum F1 / AUC
- No severe degradation on key slices
- Bias checks pass configured limits

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| sklearn.metrics | Classification metrics | TorchMetrics |
| Evidently (optional) | Drift/monitor-ready eval summaries | WhyLabs integrations |
| Matplotlib/Seaborn | Eval plots | Plotly |

---

## Phase 5: Bias Detection with Slicing

### Objective
Detect performance disparities and representation risk across meaningful subgroups.

### Required approach (per professor guideline)

1. Define slices
2. Track metrics per slice
3. Compare disparities
4. Apply mitigation when needed
5. Document trade-offs

### Suggested SavVio slices

**Financial**
- income bands, DTI bands, savings-to-income, runway

**Product**
- price bands, rating variance bands, confidence bands (`rating_number`)

**Review**
- rating sentiment buckets, verified purchase, helpfulness tiers, cold-start product bands

### Mitigation options

- re-weighting
- controlled re-sampling
- threshold adjustment
- stratified training/evaluation splits

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Fairlearn | Slice fairness analysis | AIF360 |
| TFMA | TensorFlow-centric fairness/eval | Custom slicing code |
| Pandas groupby | Manual slice metrics | DuckDB SQL analysis |

---

## Phase 6: Sensitivity & Explainability

### Objective
Understand model behavior under feature or parameter variation.

### Methods

- Feature importance (global)
- SHAP / LIME (local + global, if implemented)
- Hyperparameter sensitivity curves

### Expected output

- Top driving features
- Features that create instability across slices
- Stability notes for model selection discussion

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| SHAP | Feature contribution explanations | LIME |
| LIME | Local explanations | Integrated gradients (DL) |
| Permutation importance | Fast sensitivity baseline | Model-specific importances |

---

## Phase 7: Experiment Tracking & Model Selection

### Objective
Track every meaningful experiment and select final model using both performance and fairness criteria.

### Tracking tools

- MLflow (recommended)
- or Weights & Biases

### Required logged artifacts

- run metadata
- metrics
- plots (confusion matrix/ROC/slice charts)
- trained model artifact
- training data/version reference

### Selection rule

Final model is selected **after** bias checks and sensitivity review, not only by aggregate accuracy.

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| MLflow Tracking | Run metadata + artifacts | Weights & Biases |
| MLflow UI | Compare experiments | Custom dashboards |
| DVC tags/commit refs | Data lineage tie-in | Git-only lineage notes |

---

## Phase 8: Model Registry Push

### Objective
Version and store approved model in registry for deployment traceability.

### Options

- GCP Artifact Registry
- Vertex AI Model Registry

### Registry checklist

- model version tag
- commit hash
- training data version (DVC ref)
- validation/bias report reference
- rollback pointer to previous stable model

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Vertex AI Model Registry | Managed model registry | MLflow Model Registry |
| GCP Artifact Registry | Artifact storage | S3 + custom registry metadata |
| Cloud IAM | Access control | GitOps-managed policies |

---

## Phase 9: CI/CD Automation

### Objective
Automate training-validation-bias checks and registry push on code changes.

### Required CI/CD stages

1. Trigger on model code/config change
2. Train model
3. Validate metrics
4. Run bias detection by slices
5. Push model only if gates pass
6. Send notifications/alerts
7. Keep rollback mechanism

### Tooling

- GitHub Actions / Cloud Build
- Optional: Jenkins

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| GitHub Actions | CI orchestration | Jenkins |
| Cloud Build | GCP-native CI/CD | GitLab CI |
| Slack/Email alerts | Failure notifications | PagerDuty |

---

## Phase Summary

| Step | Command (where to run) | Output |
|------|-------------------------|--------|
| **1. Pull data** | `cd data-pipeline/dags/data && dvc pull` | Local versioned datasets |
| **2. Feature prep** | `python model-development/affordability_features.py` | Model-ready feature artifacts |
| **3. Bias checks** | `python model-development/bias_detection.py` | Slice-wise bias diagnostics |
| **4. Train model** | `python model-development/src/train.py --config model-development/config/training_config.yaml` | Candidate model artifacts |
| **5. Registry push** | `python model-development/src/registry.py --model <best-run>` | Versioned model in registry |
| **6. CI/CD gate** | Trigger via PR/push | Automated train-validate-bias decision |

---

## Project Structure

Current `model-development/` files:

- `affordability_features.py`
- `bias_detection.py`
- `README.md`

Recommended target structure:

```
model-development/
├── config/
│   ├── training_config.yaml
│   └── bias_config.yaml
├── src/
│   ├── load_data.py
│   ├── train.py
│   ├── tune.py
│   ├── validate.py
│   ├── bias_slicing.py
│   ├── sensitivity.py
│   └── registry.py
├── experiments/
├── models/
├── tests/
├── affordability_features.py
├── bias_detection.py
└── README.md
```

---

## Runbook (End-to-End)

From repository root:

1. Activate environment:
   ```bash
   source .venv/bin/activate
   ```
2. Ensure data artifacts are available:
   ```bash
   cd data-pipeline/dags/data && dvc pull
   cd ../../../..
   ```
3. Run model preparation / feature logic (example existing scripts):
   ```bash
   python model-development/affordability_features.py
   ```
4. Run slice-based bias checks:
   ```bash
   python model-development/bias_detection.py
   ```
5. Train + validate model (recommended command once `src/train.py` exists):
   ```bash
   python model-development/src/train.py --config model-development/config/training_config.yaml
   ```
6. Push approved model to registry (after gates pass).

---

## Testing

Recommended test coverage:

- unit tests for feature construction
- data contract tests for model inputs
- training smoke tests
- metric-gate tests
- slice-bias regression tests

Run (when tests are present):

```bash
pytest model-development/tests
```

---

## Operational Risks & Guardrails

### Key risks

- Financial hallucination (if model overrides deterministic logic)
- Data/concept drift (income/expense distribution shifts)
- Sparse slice underperformance (e.g., low-income or cold-start products)
- Pipeline/API dependency failures

### Guardrails

- Deterministic financial logic remains final authority
- Bias gates block promotion if disparities exceed threshold
- Registry rollback to last stable model
- Monitoring for latency, drift, and failure alerts

---

## Decision Logic Specification (Green/Yellow/Red)

### Objective
Define the deterministic recommendation policy used by SavVio so model outputs remain interpretable, auditable, and aligned with user financial safety.

### Inputs required at inference

**Financial signals (user-level):**
- `discretionary_income`
- `debt_to_income_ratio` (DTI)
- `monthly_expense_burden_ratio`
- `emergency_fund_months` (runway)
- `savings_to_income_ratio`

**Product/review confidence signals (item-level):**
- `price`
- `average_rating`
- `rating_number`
- `rating_variance`

### Rule priority order (highest to lowest)

1. **Hard-stop safety checks (always evaluated first)**
2. **Caution checks**
3. **Confidence downgrade checks**
4. **Final color assignment**

### 1) Hard-stop safety checks -> `Red`

Return **Red** if **any** condition is true:

- `discretionary_income < 0`
- `debt_to_income_ratio > 0.40`
- `monthly_expense_burden_ratio > 0.80`
- `emergency_fund_months < 1`
- `price > discretionary_income` and `emergency_fund_months < 3`

Why: user is already financially stretched, so purchase risk is high.

### 2) Caution checks -> `Yellow` candidate

If no Red rule is hit, mark as **Yellow candidate** when one or more of:

- `0 <= discretionary_income <= 1000`
- `0.20 <= debt_to_income_ratio <= 0.40`
- `0.50 <= monthly_expense_burden_ratio <= 0.80`
- `1 <= emergency_fund_months <= 3`
- `0.25 <= savings_to_income_ratio <= 1.0`
- `price` falls outside user's normal affordability tier (e.g., premium price with tight buffer)

### 3) Confidence downgrade checks (product uncertainty)

If product signal quality is weak, downgrade one level:

- `rating_number < 10` (low-confidence reviews)
- `rating_variance == 0` with low review count (single/few-review proxy)
- `rating_variance > 1.0` (polarized quality signal)
- `average_rating <= 3.0` (low quality risk)

Downgrade policy:
- Green -> Yellow
- Yellow -> Red
- Red stays Red

### 4) Final assignment

- **Green**: no Red trigger, no Yellow trigger, and no confidence downgrade.
- **Yellow**: no Red trigger, but at least one caution trigger or one downgrade from Green.
- **Red**: any hard-stop trigger, or Yellow downgraded by strong uncertainty/risk signals.

### Tie-breakers and edge cases

- Missing critical financial fields -> default to **Yellow** (insufficient confidence), unless another hard-stop is known.
- If multiple rules conflict, choose the **more conservative** class.
- If user manually overrides risk preferences, keep deterministic thresholds but log override event.

### Example scenarios

| Scenario | Key Signals | Output |
|------|-------------|--------|
| Stable profile, strong runway, moderate item | DTI 0.15, runway 5, positive discretionary | Green |
| Tight buffer, moderate debt, uncertain product | DTI 0.30, runway 2, `rating_number` 7 | Yellow (or Red if downgraded) |
| Negative discretionary income | discretionary `< 0` | Red |
| Risky debt load + polarized reviews | DTI 0.48, variance 1.3 | Red |

### Implementation note

This policy must be implemented in deterministic Python logic (rule engine).  
ML predictions can assist ranking/confidence, but **must not override hard-stop financial safety rules**.

---

## Deliverable Checklist (Model Phase)

- [ ] Data loaded from versioned pipeline outputs
- [ ] Baselines trained and compared
- [ ] Hyperparameter tuning documented
- [ ] Validation metrics on hold-out set
- [ ] Slice-based bias analysis completed
- [ ] Mitigation steps documented if disparities found
- [ ] Experiment runs tracked (MLflow/W&B)
- [ ] Best model pushed to registry
- [ ] CI/CD pipeline runs training + validation + bias checks

