# SavVio Data Pipeline Setup & Execution Guide

This guide is the current team runbook for local execution of ingestion, preprocessing,
feature engineering, and the validation/anomaly flow.

> Note: Keep this file aligned with `dags/src/*` entrypoints. Merge into `README.md` when stable.

---

## 1. Environment Setup

### Prerequisites

- Python 3.10+
- `pip` (or `uv`)
- Google Cloud credentials (if using GCS source)

### Installation

1. Go to the pipeline folder:

   ```bash
   cd data-pipeline
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r data-requirements.txt
   ```

4. Use a single `.env` file at the repository root (`Purchase-Guardrail-Agent/.env`):

   ```env
   GCP_PROJECT_ID=your-project-id
   GCS_BUCKET_NAME=savvio-data-bucket
   GCP_CREDENTIALS_PATH=/path/to/service-account.json

   DATA_DIR=data
   LOG_LEVEL=INFO
   ```

---

## 2. Execution Order (Current)

Run all commands from `data-pipeline/`.

### 2.1 Ingestion

```bash
python3 dags/src/ingestion/run_ingestion.py
```

Output:

- `data/raw/financial_data.csv`
- `data/raw/product_data.jsonl`
- `data/raw/review_data.jsonl`

### 2.2 Preprocessing

```bash
python3 dags/src/preprocess/run_preprocessing.py
```

Output:

- `data/processed/financial_preprocessed.csv`
- `data/processed/product_preprocessed.jsonl`
- `data/processed/review_preprocessed.jsonl`

### 2.3 Feature Engineering

```bash
python3 dags/src/features/run_features.py
```

Output:

- `data/features/financial_featured.csv`
- `data/features/product_rating_variance.csv`

Notes:

- Product data currently stays at `data/processed/product_preprocessed.jsonl` for DB import.
- Feature engineering is currently applied to financial and review datasets.

### 2.4 Validation + Anomaly Detection

```bash
python3 dags/src/validation/run_validation.py all
```

Stage order:

1. `raw` (schema/rule checks)
2. `raw_anomalies` (Tier 1, INFO-only monitoring)
3. `processed` (post-transform checks)
4. `features` (feature checks + formula spot checks)
5. `anomalies` (Tier 2, pre-DB anomaly checks)

To run a single stage:

```bash
python3 dags/src/validation/run_validation.py raw
python3 dags/src/validation/run_validation.py raw_anomalies
python3 dags/src/validation/run_validation.py processed
python3 dags/src/validation/run_validation.py features
python3 dags/src/validation/run_validation.py anomalies
```

---

## 3. Validation Outputs

### Reports

Validation reports are written to:

- `logs/validation/`

Examples:

- `raw_validation_*.json`
- `raw_anomaly_validation_*.json`
- `processed_validation_*.json`
- `features_validation_*.json`
- `anomaly_validation_*.json`

### Quarantine

Detected anomaly rows are written to:

- `data/quarantine/`

---

## 4. Two-Tier Anomaly Strategy

### Tier 1 (`raw_anomalies`)

- Input: `data/raw/financial_data.csv`
- Scope: financial data only (income, savings, expenses)
- Purpose: early visibility and trend monitoring
- Severity behavior: INFO-only, does not block pipeline

### Tier 2 (`anomalies`)

- Input: `data/features/financial_featured.csv`
- Scope: financial data only (core columns + engineered features)
- Purpose: anomaly checks on featured financial data before DB load
- Severity behavior: WARNING/CRITICAL drive `ALERT`/`HALT` actions

> Product and review data quality is covered by raw/processed validators.
> Anomaly detection is scoped to financial data where outliers have the
> highest impact on the agent's purchase guardrail decisions.

---

## 5. DVC Workflow (Raw / Processed / Features)

After data changes:

```bash
dvc add data/raw
dvc add data/processed
dvc add data/features
```

Commit metadata and push data:

```bash
git add .
git commit -m "Update data artifacts"
dvc push
```

For teammates:

```bash
git pull origin main
dvc pull
```

---

## 6. Quick Command List

```bash
# 1) Ingestion
python3 dags/src/ingestion/run_ingestion.py

# 2) Preprocessing
python3 dags/src/preprocess/run_preprocessing.py

# 3) Feature engineering
python3 dags/src/features/run_features.py

# 4) Full validation pipeline
python3 dags/src/validation/run_validation.py all
```
