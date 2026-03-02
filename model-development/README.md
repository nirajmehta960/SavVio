# SavVio — Model Development Pipeline

## AI-Driven Financial Advocacy Tool

**Course:** MLOps  
**Phase:** Model Development (Phase 3)  
**Team:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Data Source](#4-data-source)
5. [Phase 1 — Data Loading](#5-phase-1--data-loading)
6. [Phase 2 — Feature Engineering](#6-phase-2--feature-engineering)
7. [Phase 3 — Deterministic Decision Engine](#7-phase-3--deterministic-decision-engine)
8. [Phase 4 — Model Training](#8-phase-4--model-training)
9. [Phase 5 — Hyperparameter Tuning](#9-phase-5--hyperparameter-tuning)
10. [Phase 6 — Validation & Metrics](#10-phase-6--validation--metrics)
11. [Phase 7 — Bias Detection](#11-phase-7--bias-detection)
12. [Phase 8 — Bias Mitigation](#12-phase-8--bias-mitigation)
13. [Phase 9 — Model Selection](#13-phase-9--model-selection)
14. [Phase 10 — Sensitivity & Explainability](#14-phase-10--sensitivity--explainability)
15. [Phase 11 — Experiment Tracking](#15-phase-11--experiment-tracking)
16. [Phase 12 — Model Registry Push](#16-phase-12--model-registry-push)
17. [Phase 13 — CI/CD Automation](#17-phase-13--cicd-automation)
18. [Phase 14 — LLM Wrapping & Guardrails](#18-phase-14--llm-wrapping--guardrails)
19. [Phase 15 — Monitoring & Dashboard](#19-phase-15--monitoring--dashboard)
20. [Runbook](#runbook)
21. [Testing](#testing)
22. [Operational Risks & Guardrails](#operational-risks--guardrails)
23. [Deliverable Checklist](#deliverable-checklist)

---

## Quick Reference: Tools by Phase

| Phase | Primary Tools | Alternatives | CI/CD Gate |
|-------|--------------|--------------|------------|
| Data Loading | DVC, GCS, Pandas | Polars, LakeFS | Data version + schema check |
| Feature Engineering | Pandas, NumPy, scikit-learn | Polars, Feature-engine | Feature schema validation |
| Deterministic Engine | Pure Python | — | Unit tests must pass |
| Model Training | XGBoost, scikit-learn | LightGBM, LinearBoost | Reproducible training run |
| Hyperparameter Tuning | Ray Tune | Optuna, RandomizedSearchCV | Best-run tracking required |
| Validation & Metrics | sklearn.metrics, Matplotlib, Seaborn | TorchMetrics, Plotly | Minimum F1 / AUC threshold |
| Bias Detection | Fairlearn, Pandas groupby | AIF360, TFMA | Block on severe disparity |
| Bias Mitigation | Fairlearn, imbalanced-learn | scikit-learn threshold | Re-evaluate until gates pass |
| Model Selection | MLflow UI | Custom dashboards | Bias mitigation gate must pass |
| Sensitivity & Explainability | SHAP, LIME | Permutation Importance | Stability report required |
| Experiment Tracking | MLflow | Weights & Biases | Run metadata completeness |
| Model Registry Push | GCP Artifact Registry, Vertex AI | MLflow Registry | Push only on all gates pass |
| CI/CD Automation | GitHub Actions, Docker | Cloud Build, Jenkins | src ↔ test ↔ DB ↔ ML pipeline |
| LLM Wrapping | NeMo Guardrails, LangChain | Guardrails AI | Hallucination + safety gate |
| Monitoring & Dashboard | Evidently, Arize | WhyLabs, GCP Monitoring | Drift + latency alerts |

---

---

## 1. Project Overview

SavVio is an AI-driven financial advocacy tool that provides pre-purchase recommendations to users in the form of a **Green / Yellow / Red** signal, based on their financial health and product quality signals.

**Core design principle:**
> LLM handles natural language context. Deterministic logic handles all financial math. ML handles confidence scoring and ranking.

The recommendation pipeline has three layers:

| Layer | Responsibility |
|-------|---------------|
| Deterministic Engine | Authoritative Green/Yellow/Red classification via rule engine |
| ML Model | Confidence scoring and ranking support |
| LLM + Guardrails | Natural language recommendation delivery with safety enforcement |

---

## 2. Architecture

```
User Input
    ↓
┌─────────────────────────────────┐
│     Deterministic Engine        │  ← Hard financial rules
│     Green / Yellow / Red        │  ← Authoritative output
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│        ML Model Layer           │  ← Confidence + ranking support
│   XGBoost / LinearBoost         │  ← Cannot override engine output
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│      LLM Wrapping Layer         │  ← Natural language generation
│      NeMo Guardrails            │  ← Hallucination + safety enforcement
└─────────────────────────────────┘
    ↓
User-facing Recommendation
    ↓
┌─────────────────────────────────┐
│   Monitoring & Dashboard        │  ← Latency, drift, hallucination alerts
└─────────────────────────────────┘
```

### Model Pipeline Execution Order

```
1.  Load Versioned Data (GCS via DVC)
        ↓
2.  Feature Engineering
        ↓
3.  Model Training (XGBoost / LinearBoost baselines)
        ↓
4.  Hyperparameter Tuning (Ray Tune)
        ↓
5.  Validation on Hold-out Set
        ↓
6.  Bias Detection — Post-Training (slice analysis)
        ↓
7.  Model Selection (performance + bias constraints)
        ↓
8.  Sensitivity & Explainability (SHAP / LIME)
        ↓
9.  Experiment Tracking (MLflow)
        ↓
10. Deterministic Engine → Green / Yellow / Red
        ↓
11. LLM Wrapping + NeMo Guardrails
        ↓
12. Model Registry Push (GCP)
        ↓
13. CI/CD Automation (Dockerized)
```

---

## 3. Repository Structure

```
SavVio/
├── data-pipeline/
│   ├── README.md
│   ├── SETUP_AND_RUN.md
│   ├── data-requirements.txt
│   ├── config/                          # Airflow, Token, GCP config
│   ├── logs/                            # Pipeline execution logs
│   ├── tests/
│   │   ├── ingestion/
│   │   ├── preprocess/
│   │   ├── features/
│   │   ├── validation/
│   │   ├── database/
│   │   ├── bias/
│   │   └── test_data_pipeline_airflow.py
│   └── dags/
│       ├── data_pipeline_airflow.py     # Main Airflow DAG
│       ├── data/
│       │   ├── raw/                     # Raw data (financial.csv, product.jsonl, review.jsonl)
│       │   ├── processed/               # Preprocessed outputs
│       │   ├── features/                # Feature-engineered outputs ← model reads from here
│       │   ├── raw.dvc
│       │   ├── processed.dvc
│       │   └── features.dvc
│       └── src/
│           ├── ingestion/               # Data acquisition (GCS, APIs)
│           │   ├── __init__.py
│           │   ├── api_loader.py
│           │   ├── config.py
│           │   ├── gcs_loader.py
│           │   └── run_ingestion.py
│           ├── preprocess/              # Cleaning and transformation
│           ├── features/                # Pipeline-side feature engineering
│           ├── validation/
│           │   └── anomaly/
│           ├── bias/
│           └── database/
│               ├── __init__.py
│               ├── db_connection.py
│               ├── db_schema.py
│               ├── run_database.py
│               ├── upload_to_db.py
│               └── vector_embed.py
│
└── model-development/                   # ← THIS PHASE
    ├── README.md
    ├── config/
    │   ├── training_config.yaml         # Hyperparams, paths, seeds
    │   └── bias_config.yaml             # Slice definitions and thresholds
    ├── src/
    │   ├── load_data.py                 # Phase 1  — Data loading
    │   ├── features.py                  # Phase 2  — Feature assembly
    │   ├── train.py                     # Phase 3  — Training + selection
    │   ├── tune.py                      # Phase 4  — Ray Tune
    │   ├── validate.py                  # Phase 5  — Validation + plots
    │   ├── bias_slicing.py              # Phase 6  — Post-training bias
    │   ├── sensitivity.py               # Phase 8  — SHAP / LIME
    │   ├── registry.py                  # Phase 10 — Registry push
    │   └── llm_wrapper.py              # Phase 12 — LLM + NeMo
    ├── deterministic_engine/
    │   └── decision_logic.py            # Green/Yellow/Red rule engine
    ├── docker/
    │   └── Dockerfile                   # Full pipeline containerization
    ├── tests/
    │   ├── test_load_data.py
    │   ├── test_features.py
    │   ├── test_train.py
    │   ├── test_validate.py
    │   ├── test_bias_slicing.py
    │   ├── test_sensitivity.py
    │   ├── test_registry.py
    │   ├── test_decision_logic.py
    │   └── test_llm_wrapper.py
    ├── experiments/                     # MLflow run artifacts
    ├── models/                          # Local model staging
    ├── affordability_features.py        # Legacy → absorbed into src/features.py
    └── bias_detection.py                # Legacy → absorbed into src/bias_slicing.py
```

---

## 4. Data Source

Model training reads exclusively from **GCS via DVC-versioned artifacts** produced by the data pipeline. The raw database is never accessed directly during model training.

| Source | Role |
|--------|------|
| `GCS data/ bucket (via DVC)` | Primary input for model training — versioned `.csv` and `.jsonl` feature files |
| `Database` | Used upstream in the data pipeline only — not accessed during model development |

**Input artifacts consumed by model pipeline:**

```
data-pipeline/dags/data/features/
├── financial_featured.csv
├── product_featured.jsonl
└── review_featured.jsonl
```

---

## Phase 1 — Data Loading

**Objective:** Load the latest versioned and validated datasets from the data pipeline output and ensure reproducible model inputs.

**Tasks:**
- Configure DVC remote to point to GCS bucket
- Pull versioned feature artifacts:
  ```bash
  cd data-pipeline/dags/data
  dvc pull
  ```
- Validate file existence for all three feature files
- Run schema checks (Pandera / Great Expectations) — verify column names, types, null rates
- Log DVC commit hash and GCS artifact path for reproducibility tracing
- Join financial, product, and review features into a single model-ready table
- Construct Green/Yellow/Red labels from deterministic engine outputs for supervised training

**Tools:**

| Tool | Purpose |
|------|---------|
| DVC + GCS | Pull versioned artifacts |
| Pandas | Load tabular and JSONL data |
| Pandera / Great Expectations | Schema and data contract checks |

---

## Phase 2 — Feature Engineering

**Objective:** Transform raw pipeline artifacts into model-ready features covering financial, product, and review signals.

**Tasks:**
- Extract and normalize financial features from `financial_featured.csv`: `discretionary_income`, `debt_to_income_ratio`, `monthly_expense_burden_ratio`, `emergency_fund_months`, `savings_to_income_ratio`
- Extract product signals from `product_featured.jsonl`: `price`, `average_rating`, `rating_number`, `rating_variance`
- Build review signals from `review_featured.jsonl`: sentiment bucket, verified purchase flag, helpfulness tier, cold-start indicator
- Handle missing values — apply imputation strategy per field; missing financial fields default to Yellow
- Encode categorical features (income bands, sentiment buckets)
- Apply feature scaling and normalization where required
- Run TA feature selection to identify top features before training
- Save final feature matrix and label vector as a versioned artifact

**Reference:** https://github.com/raminmohammadi/MLOps/tree/main/Labs/Model_Development/Feature_Selection

**Tools:**

| Tool | Purpose |
|------|---------|
| Pandas / NumPy | Feature construction |
| scikit-learn Pipelines | Preprocessing and transformation |

---

## Phase 3 — Deterministic Decision Engine

**Location:** `deterministic_engine/decision_logic.py`

The deterministic engine is a pure Python rule system that computes the final Green/Yellow/Red recommendation. It must be built **before** model training because it generates the labels (Green/Yellow/Red) that the ML model trains on. Its output is authoritative — neither the ML model nor the LLM layer can override it.

### Rule Priority Order (Highest to Lowest)

1. Hard-stop safety checks
2. Caution checks
3. Confidence downgrade checks
4. Final color assignment

### Why This Order Matters

Rules are evaluated from most to least conservative. If any hard financial limit is breached, Red is returned immediately with no further evaluation. This ensures safety flags can never be bypassed by product quality or ML confidence signals.

### Inputs at Inference

**Financial (user-level):**

| Field | Description |
|-------|-------------|
| `discretionary_income` | Income remaining after fixed expenses |
| `debt_to_income_ratio` | Total debt payments / gross income |
| `monthly_expense_burden_ratio` | Monthly expenses / monthly income |
| `emergency_fund_months` | Months of expenses covered by savings |
| `savings_to_income_ratio` | Savings / monthly income |

**Product / Review (item-level):**

| Field | Description |
|-------|-------------|
| `price` | Item price |
| `average_rating` | Mean product rating |
| `rating_number` | Total number of ratings |
| `rating_variance` | Variance across ratings |

### 1) Hard-stop Safety Checks → Red

Return **Red** immediately if any condition is true:

- `discretionary_income < 0`
- `debt_to_income_ratio > 0.40`
- `monthly_expense_burden_ratio > 0.80`
- `emergency_fund_months < 1`
- `price > discretionary_income` AND `emergency_fund_months < 3`

### 2) Caution Checks → Yellow Candidate

Mark as Yellow if no Red rule triggered and one or more of:

- `0 <= discretionary_income <= 1000`
- `0.20 <= debt_to_income_ratio <= 0.40`
- `0.50 <= monthly_expense_burden_ratio <= 0.80`
- `1 <= emergency_fund_months <= 3`
- `0.25 <= savings_to_income_ratio <= 1.0`
- `price` falls outside user's normal affordability tier

### 3) Confidence Downgrade Checks

Downgrade one level if product signal quality is weak:

- `rating_number < 10` — insufficient reviews
- `rating_variance == 0` with low review count — artificially uniform signal
- `rating_variance > 1.0` — highly polarized, uncertain quality
- `average_rating <= 3.0` — poor quality signal

Downgrade policy: Green → Yellow → Red (Red stays Red)

### 4) Final Assignment

| Result | Condition |
|--------|-----------|
| **Green** | No Red trigger, no Yellow trigger, no confidence downgrade |
| **Yellow** | No Red trigger, but at least one caution trigger or one downgrade from Green |
| **Red** | Any hard-stop trigger, or Yellow downgraded by strong uncertainty |

### Edge Cases

- Missing financial fields → default **Yellow**
- Conflicting rules → choose **more conservative** class
- User override → retain thresholds, log override event

### Example Scenarios

| Scenario | Key Signals | Output |
|----------|-------------|--------|
| Stable profile, strong runway | DTI 0.15, runway 5 months, positive discretionary | Green |
| Tight buffer, uncertain product | DTI 0.30, runway 2, `rating_number` 7 | Yellow |
| Negative discretionary income | `discretionary_income < 0` | Red |
| Risky debt + polarized reviews | DTI 0.48, variance 1.3 | Red |

### Tasks

- [ ] Implement hard-stop safety checks → Red output
- [ ] Implement caution checks → Yellow candidate logic
- [ ] Implement confidence downgrade checks (product signal quality)
- [ ] Implement final color assignment with tie-breaker rules
- [ ] Use engine output to generate labels for ML model training
- [ ] Write unit tests for all rule conditions and edge cases
- [ ] Verify engine output cannot be overridden by ML or LLM layer

---

## Phase 4 — Model Training

**Objective:** Train baseline candidates for recommendation and decision support.

**Tasks:**
- Set fixed random seeds across NumPy, scikit-learn, and XGBoost for reproducibility
- Create stratified train/validation/test split by label class
- Train **XGBoost (Plan A)** with default hyperparameters as primary baseline
- Train **LinearBoost (Plan B)** as fast fallback (~98% faster than XGBoost)
- Train optional calibrated tree model (CalibratedClassifierCV)
- Log each baseline run to MLflow: model type, params, train/val metrics
- Compare baseline results in MLflow UI and document best pre-tuning candidate

**Note:** ML model output supports confidence scoring and ranking only. It does not override the deterministic financial safety logic.

**Tools:**

| Tool | Purpose |
|------|---------|
| XGBoost | Primary nonlinear baseline (Plan A) |
| LinearBoost | Fast linear fallback (Plan B) |
| scikit-learn | Preprocessing and pipeline |
| MLflow | Baseline run logging |

---

## Phase 5 — Hyperparameter Tuning

**Objective:** Optimize model performance while preserving fairness and robustness.

**Search Strategy:** Ray Tune (Plan A) — distributed, scalable, integrates natively with MLflow. Fallback: RandomizedSearchCV or Optuna.

**Tasks:**
- Define search space: learning rate, max depth, n_estimators, subsample, regularization terms
- Set up Ray Tune with MLflow callback for automatic run logging
- Run distributed hyperparameter search with early stopping
- Fall back to RandomizedSearchCV or Optuna if Ray is unavailable
- Log every trial: model type, hyperparameters, split strategy, validation metrics, artifact path
- Identify and tag best trial in MLflow
- Document search space and tuning strategy for submission

**Minimum logging per run:** model type, hyperparameters, split strategy, validation metrics, artifact path / run ID.

**Tools:**

| Tool | Purpose |
|------|---------|
| Ray Tune | Distributed hyperparameter search (Plan A) |
| Optuna | Bayesian optimization fallback |
| GridSearchCV / RandomizedSearchCV | Simple search fallback |
| MLflow | Trial logging |

---

## Phase 6 — Validation & Metrics

**Objective:** Validate model performance on unseen data using task-relevant metrics and required visualizations.

**Tasks:**
- Run trained model on held-out test set (not used during training or tuning)
- Compute: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Compute PR-AUC if class imbalance is present
- Generate calibration / reliability curve
- Generate confusion matrix per model candidate (Green / Yellow / Red classes)
- Generate ROC curve comparison across all model candidates
- Generate bar plots comparing F1 / AUC across model runs
- Log all metrics and plots to MLflow
- Apply acceptance gates — block promotion if below minimum thresholds
- Document which models pass or fail the gate and why

**Tools:**

| Tool | Purpose |
|------|---------|
| sklearn.metrics | Classification metrics |
| Matplotlib / Seaborn | Required visualizations |
| Evidently | Drift-ready eval summaries |

---

## Phase 7 — Bias Detection

**Objective:** Detect performance disparities across meaningful data subgroups after model training. Bias detection is performed **post-training** — run on validation/test set predictions after model fitting is complete.

**Tasks:**
- Define all slices in `config/bias_config.yaml`
- Collect model predictions and ground truth per slice on validation/test set
- Compute per-slice metrics: Accuracy, F1, AUC for each subgroup
- Compare per-slice vs. aggregate metrics — flag disparities above configured threshold
- Generate bias report: F1 bar chart per slice, disparity summary table
- Log bias report and visualizations to MLflow
- Document all detected disparities before moving to mitigation

**Slice Definitions for SavVio:**

| Slice Type | Subgroups |
|------------|-----------|
| Financial | Income bands, DTI bands, savings-to-income, emergency fund runway |
| Product | Price bands, rating variance bands, review confidence bands (`rating_number`) |
| Review | Sentiment buckets, verified purchase, helpfulness tiers, cold-start bands |

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Slice fairness analysis |
| AIF360 | Alternative fairness toolkit |
| TFMA | TensorFlow-centric slice evaluation |
| Pandas groupby | Manual slice metric computation |

---

## Phase 8 — Bias Mitigation

**Objective:** Apply mitigation strategies to address detected bias and re-evaluate until disparities fall within acceptable thresholds.

**Tasks:**
- Review bias report from Phase 7 — identify which slices exceed disparity threshold
- Apply one or more mitigation strategies:
  - **Re-weighting** — assign higher loss weights to underrepresented groups
  - **Controlled re-sampling** — oversample sparse slices in training data
  - **Decision threshold adjustment** — set different classification thresholds per slice
  - **Stratified re-training** — retrain with stratified splits enforcing slice balance
- Re-run bias detection (Phase 7) after mitigation
- Compare pre- and post-mitigation disparity metrics
- Document trade-offs made (e.g., slight drop in aggregate accuracy for fairness gain)
- If disparity persists beyond threshold → block model promotion via CI/CD gate

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Fairness constraints and threshold optimization |
| imbalanced-learn | Re-sampling strategies |
| scikit-learn | Threshold adjustment per class |

---



## Phase 9 — Model Selection

**Objective:** Select the final model only after both validation metrics and bias mitigation are satisfactory.

**Tasks:**
- Collect all candidates that passed the validation gate (Phase 6)
- Filter out any candidate that failed the bias gate (Phase 8)
- Rank remaining candidates by aggregate F1 / AUC
- Review sensitivity and stability notes from Phase 10 for final candidates
- Select best model and document: metrics, bias results, trade-offs made
- Tag selected run in MLflow as `best-model`
- Log selection rationale as an MLflow run note or artifact

**Selection rule:** Final model is never selected on aggregate accuracy alone. The bias mitigation gate must pass first.

---

## Phase 10 — Sensitivity & Explainability

**Objective:** Understand how model behavior changes with respect to input features and hyperparameter variation.

**Tasks:**
- Compute global feature importance from XGBoost built-in scores
- Run SHAP on selected model — generate global summary plot and local force plots
- Run LIME on 3–5 representative predictions per class (Green / Yellow / Red)
- Generate hyperparameter sensitivity curves (F1 vs. key hyperparameters)
- Identify features that cause instability across financial and product slices
- Document top 5–10 driving features for Green/Yellow/Red classification
- Log all SHAP plots, LIME explanations, and sensitivity charts to MLflow
- Write stability notes flagging inconsistent features across slices

**Tools:**

| Tool | Purpose |
|------|---------|
| SHAP | Global and local feature contribution explanations |
| LIME | Local interpretable explanations |
| Permutation Importance | Fast sensitivity baseline |

---

## Phase 11 — Experiment Tracking

**Objective:** Track every meaningful experiment and maintain full lineage from data version to model artifact.

**Tasks:**
- Set up MLflow tracking server (local or GCP-hosted)
- Create MLflow experiment named `savvio-model-development`
- Instrument `train.py` to auto-log: params, metrics, model artifact, data version reference
- Log bias reports and slice charts as MLflow artifacts per run
- Log confusion matrix, ROC curve, and bar plots per run
- Use MLflow UI to compare all runs — capture comparison screenshot for submission
- Tag winning run as `best-model` with version label
- Tie DVC commit hash and GCS data reference to each MLflow run for full lineage

**Tools:**

| Tool | Purpose |
|------|---------|
| MLflow Tracking | Run metadata and artifacts |
| MLflow UI | Experiment comparison and visualization |
| DVC tags / commit refs | Data lineage tie-in |

---

## Phase 12 — Model Registry Push

**Objective:** Version and store the approved model in the registry for deployment traceability and rollback capability.

**Tasks:**
- Confirm model passed all gates: validation ✅ and bias ✅
- Serialize model artifact (pickle / joblib / ONNX)
- Tag artifact with: model version, commit hash, DVC data ref, MLflow run ID
- Push to GCP Artifact Registry or Vertex AI Model Registry
- Record rollback pointer — store previous stable model version tag
- Verify push succeeded and artifact is retrievable from registry
- Update `config/training_config.yaml` with latest registered model version

**Registry checklist:**

- [ ] Model version tag
- [ ] Commit hash
- [ ] Training data version (DVC ref)
- [ ] Validation metric report reference
- [ ] Bias analysis report reference
- [ ] Rollback pointer to previous stable model

**Tools:**

| Tool | Purpose |
|------|---------|
| GCP Artifact Registry | Artifact storage and versioning |
| Vertex AI Model Registry | Managed model registry |
| Cloud IAM | Access control |

---

## Phase 13 — CI/CD Automation

**Objective:** Automate the full training → validation → bias → registry pipeline on every code change, containerized in Docker, connecting `src ↔ test ↔ DB ↔ ML`.

### Pipeline Architecture

```
GitHub Push / PR on model-development/
        ↓
GitHub Actions / Cloud Build  [Dockerized]
        ├── 1. src unit tests
        ├── 2. DB connection check
        ├── 3. DVC pull (versioned data)
        ├── 4. ML training (inside Docker container)
        ├── 5. Automated validation gate
        │       └── below threshold? → BLOCK + alert
        ├── 6. Automated bias detection gate
        │       └── severe disparity? → BLOCK + alert
        ├── 7. Rollback check
        │       └── worse than previous? → BLOCK + alert
        ├── 8. Registry push (only if all gates pass)
        └── 9. Slack / email notification
```

**Tasks:**
- Write `docker/Dockerfile` to containerize full training and validation environment
- Test Docker build locally: `docker build -t savvio-model ./docker`
- Configure GitHub Actions workflow (`.github/workflows/model_ci.yml`)
- Implement automated validation gate — fail CI if metrics below threshold
- Implement automated bias gate — fail CI and send alert if disparity exceeds limit
- Implement rollback mechanism — compare new vs. previous registry model; block if worse
- Connect pipeline: src unit tests → DB connection check → ML training → registry push
- Set up Slack/email notifications for pipeline failure, completion, and gate failures
- Test full end-to-end pipeline in CI environment
- Document pipeline YAML and Docker setup for reproducibility

### Sample Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model-development/ ./model-development/
COPY data-pipeline/ ./data-pipeline/
CMD ["python", "model-development/src/train.py", \
     "--config", "model-development/config/training_config.yaml"]
```

**Tools:**

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI orchestration |
| Docker | Full pipeline containerization |
| Cloud Build | GCP-native CI/CD alternative |
| Slack / Email | Failure and completion notifications |

---

## Phase 14 — LLM Wrapping & Guardrails

**Objective:** Wrap ML and deterministic outputs with an LLM for natural language delivery, enforced by safety guardrails and prompt engineering.

**Tasks:**
- Design system prompt template including: financial signals, decision class (Green/Yellow/Red), product context
- Implement `llm_wrapper.py` — takes deterministic engine output + ML confidence → generates natural language recommendation
- Enforce that LLM output never contradicts the deterministic engine color class
- Integrate NeMo Guardrails — define rails for: hallucinated financial figures, out-of-scope advice, unsafe outputs
- Test guardrails with adversarial prompts — confirm rails block unsafe completions
- Version-control all prompt templates alongside model artifacts
- Set up monitoring dashboard: track latency, refusal rate, hallucination flags per request
- Set up drift detection on LLM recommendation distribution over time
- Configure alerts for: hallucination spike, latency threshold breach, safety rail trigger volume
- Document all prompt engineering decisions and guardrail configuration

**Tools:**

| Tool | Purpose |
|------|---------|
| NeMo Guardrails | LLM safety and hallucination prevention |
| LangChain / LlamaIndex | Prompt orchestration |
| Evidently / Arize | Output monitoring and drift detection |



---

## Phase 15 — Monitoring & Dashboard

**Objective:** Monitor the live system for latency, drift, hallucination flags, and recommendation quality after deployment.

**Tasks:**
- Set up monitoring dashboard tracking: latency per request, refusal rate, hallucination flag rate
- Monitor ML model for data/concept drift — flag if input feature distributions shift significantly
- Monitor LLM recommendation distribution over time — detect drift in Green/Yellow/Red output ratios
- Set up alerts for:
  - Hallucination spike above threshold
  - Latency breach (response time > SLA)
  - Safety rail trigger volume increase
  - Model performance degradation vs. baseline
- Log all monitoring metrics to dashboard (Evidently / Arize)
- Trigger re-training pipeline if drift exceeds configured threshold
- Document monitoring setup and alert thresholds

**Tools:**

| Tool | Purpose |
|------|---------|
| Evidently | Data and model drift detection |
| Arize | ML observability and monitoring |
| WhyLabs | Alternative monitoring platform |
| GCP Cloud Monitoring | Infrastructure and latency alerts |

---

## 20. Runbook

```bash
# 1. Activate environment
source .venv/bin/activate
# OR use Docker:
docker build -t savvio-model ./docker
docker run savvio-model

# 2. Pull versioned data from GCS via DVC
cd data-pipeline/dags/data
dvc pull
cd ../../..

# 3. Run feature engineering
python model-development/affordability_features.py

# 4. Train model
python model-development/src/train.py \
  --config model-development/config/training_config.yaml

# 5. Run post-training bias detection
python model-development/bias_detection.py

# 6. Push approved model to registry
python model-development/src/registry.py --model <best-run-id>
```

---

## 21. Testing

```bash
pytest model-development/tests
```

**Test coverage:**

| Test File | What It Tests |
|-----------|--------------|
| `test_load_data.py` | DVC pull, schema checks, join logic |
| `test_features.py` | Feature construction, encoding, scaling |
| `test_train.py` | Training smoke test, seed reproducibility |
| `test_validate.py` | Metric computation, acceptance gates |
| `test_bias_slicing.py` | Slice metric computation, disparity detection |
| `test_sensitivity.py` | SHAP output shape, LIME stability |
| `test_registry.py` | Registry push, rollback pointer |
| `test_decision_logic.py` | All rule conditions and edge cases |
| `test_llm_wrapper.py` | Guardrail enforcement, prompt output validation |

---

## 22. Operational Risks & Guardrails

### Risks

| Risk | Description |
|------|-------------|
| Financial hallucination | LLM contradicts or overrides deterministic engine output |
| Data/concept drift | Income or expense distributions shift over time |
| Sparse slice underperformance | Low-income or cold-start users receive lower quality recommendations |
| Pipeline/API failures | DVC pull errors, GCS connectivity issues, registry downtime |

### Guardrails

| Guardrail | Mechanism |
|-----------|-----------|
| Deterministic engine authority | Rule engine output is final — ML and LLM cannot override it |
| NeMo Guardrails | Safety rails enforced at LLM output boundary |
| Bias promotion gate | CI/CD blocks model push if slice disparities exceed threshold |
| Registry rollback | Previous stable model version retained; auto-revert on underperformance |
| Monitoring dashboard | Real-time alerts for latency, drift, and hallucination flags |

---

## 23. Deliverable Checklist

### Professor Guidelines

- [ ] Data loaded from versioned pipeline outputs (GCS via DVC)
- [ ] Baseline models trained and compared
- [ ] Hyperparameter tuning documented (Ray Tune — Plan A)
- [ ] Validation metrics computed on hold-out set
- [ ] Visualizations produced: confusion matrix, ROC curve, F1/AUC bar plots
- [ ] Experiments tracked in MLflow with full artifact logging
- [ ] Sensitivity and explainability analysis completed (SHAP / LIME)
- [ ] Post-training slice-based bias analysis completed
- [ ] Bias mitigation steps documented where disparities found
- [ ] Model selection performed after bias checking
- [ ] Best model pushed to GCP Artifact Registry or Vertex AI
- [ ] CI/CD pipeline: trigger → train → validate → bias → push
- [ ] Automated validation gate implemented
- [ ] Automated bias detection gate implemented
- [ ] Notifications and alerts configured
- [ ] Rollback mechanism implemented
- [ ] Full pipeline containerized in Docker

### SavVio-Specific

- [ ] Data source confirmed: GCS via DVC (not raw database)
- [ ] Deterministic engine implemented for Green/Yellow/Red logic
- [ ] ML model confirmed as confidence layer only — does not override engine
- [ ] Ray Tune configured as Plan A for hyperparameter search
- [ ] Bias detection confirmed as post-training
- [ ] TA feature selection reference incorporated
- [ ] MLflow experiment tracking fully implemented
- [ ] CI/CD connects src ↔ test ↔ DB ↔ ML (Dockerized)
- [ ] LLM wrapping implemented with NeMo Guardrails
- [ ] Prompt templates version-controlled
- [ ] Monitoring and dashboard deployed
