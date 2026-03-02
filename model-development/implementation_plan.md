# Machine Learning Pipeline Implementation Plan

This document outlines a detailed blueprint for the ML Pipeline, treating it as a first-class citizen of the product (equal to the Data Pipeline) and preparing it for seamless integration with downstream deployment tools (Terraform, GitHub Actions, FastAPI backend, React frontend).

## 1. High-Level Architecture Overview

The ultimate product lifecycle will be split into three decoupled but interconnected phases:
1. **Data Pipeline (Airflow)**: Ingests raw data, processes it, and dumps prepared features into a staging environment / Data Warehouse.
2. **ML Pipeline (MLflow + ML Dev)**: Consumes the prepared data to train models, orchestrate feature engineering pipelines, evaluate biases, track experiments/hyperparameters in MLflow, apply LLM-driven prompt engineering, and register the winning model artifact.
3. **Deployment (CI/CD + Backend/Frontend)**: A CI/CD pipeline (e.g., GitHub Actions) triggers Terraform to provision infrastructure, pulling the "Registered/Production" model from MLflow, wrapping it in a backend (FastAPI), and serving it to the frontend.

## 2. Infrastructure Consolidation (Docker)

You currently have two docker-compose files in `model-development/`:
- [docker-compose.yml](file:///Users/bridgeinformatics/temp/NEU/MLOps/Purchase-Guardrail-Agent/model-development/docker-compose.yml) (MLflow server, Postgres metadata backend, RustFS/S3 artifact storage).
- [docker-compose_.yml](file:///Users/bridgeinformatics/temp/NEU/MLOps/Purchase-Guardrail-Agent/model-development/docker-compose_.yml) (The ML developer GPU environment I created earlier).

**Step 1 Action Plan**:
We must **merge** these two files into a single [docker-compose.yml](file:///Users/bridgeinformatics/temp/NEU/MLOps/Purchase-Guardrail-Agent/model-development/docker-compose.yml) to ensure that the `ml-dev` container (running your XGBoost/LGBM code) is natively attached to the `mlflow-network`.

- **`postgres`**: Stores MLflow experiment metadata.
- **`storage` (RustFS)**: Local S3-compatible object storage for heavy artifacts (the actual `.pkl` models and LLM prompt templates).
- **`create-bucket`**: Initializes the S3 bucket.
- **`mlflow`**: The UI and API for model tracking.
- **`ml-dev`**: Built using our `nvidia/cuda` Dockerfile. This is where your code executes. It will mount `../data`, `../temp_data`, and wait for commands.

*Why this matters?* Connecting these networks means your Python code in `ml-dev` can simply call `mlflow.set_tracking_uri("http://mlflow:5000")` instead of complex host-networking workarounds.

## 3. The ML Pipeline Code Structure

Currently, you have a Jupyter Notebook ([xgboost.ipynb](file:///Users/bridgeinformatics/temp/NEU/MLOps/Purchase-Guardrail-Agent/model-development/src/xgboost.ipynb)). To make this a production-grade ML pipeline, we must modularize the code inside `model-development/src/`.

### Proposed Folder Structure:
```text
model-development/
├── Dockerfile                  # GPU-enabled base image
├── docker-compose.yml          # Unified MLflow + ML-Dev orchestration
├── model-requirements.txt
├── src/
│   ├── run_pipeline.py         # Entrypoint to run the whole ML pipeline end-to-end
│   ├── config.py               # Defines MLflow URIs, Hyperparameters, File paths
│   ├── data_loader.py          # Reads from temp_data or Airflow DB
│   ├── features/
│   │   └── engineering.py      # OrdinalEncoding, Scaling, Target definitions
│   ├── core_models/
│   │   ├── train.py            # Logic to train XGBoost, LightGBM, LinearBoost
│   │   └── evaluate.py         # Accuracy, F1, ROC-AUC calculations
│   ├── guards/
│   │   └── bias_detection.py   # Fairlearn integration (Demographic Parity)
│   └── llm/
│       └── prompt_engin.py     # Wrappers for prompting/language explanations
```

## 4. Pipeline Execution Flow

When `python src/run_pipeline.py` is executed (either manually by you in the `ml-dev` container, or automatically by a CI/CD runner):

### Phase A: Data & Feature Engineering
- Load `financial_featured.csv`.
- Create and fit `OrdinalEncoder` and `StandardScaler`.
- **CRITICAL**: Log the fitted `encoder.pkl` and `scaler.pkl` as artifacts to MLflow. The future FastAPI backend *must* download these exact files to transform live user input identically to the training data.

### Phase B: Training & Experiment Tracking
- Define hyperparameter grids for XGBoost, LightGBM, and LinearBoost.
- Start an MLflow Run.
- Iterate through combinations using cross-validation.
- Use `mlflow.log_params()` and `mlflow.log_metrics()` for every run.
- Log the actual trained model to the MLflow artifact store.

### Phase C: Bias Detection & Constraints
- After training, the best model is passed to `guards/bias_detection.py`.
- Using `fairlearn`, we evaluate metrics like Disparate Impact or Equalized Odds against sensitive features (Region, Gender).
- If the model violates the bias thresholds, the pipeline fails early, or flags the model in MLflow as "Unsafe".

### Phase D: LLM Wrap & Prompts
- The LLM logic generates explanation templates or guardrail constraints.
- We log these specific prompt version strings into MLflow as parameters, locking the LLM behavior to the specific version of the boosting model.

### Phase E: Model Registration
- The pipeline script queries MLflow for the "Best" run based on a composite score (Accuracy - Bias Penalty).
- It registers this model in the MLflow Model Registry and transitions its stage to `Staging`.

## 5. Deployment Handoff Strategy (The Future)

By building the ML Pipeline this way, we make Deployment trivial:

1. **Trigger**: When you are happy with the model marked `Staging` in MLflow, you manually click transition to `Production` in the MLflow UI (or push a git tag).
2. **GitHub Actions / CI/CD**: Detects this change.
3. **Terraform**: Spins up AWS/GCP infrastructure for your Backend (FastAPI).
4. **Backend Boot**: Upon starting, the FastAPI backend queries MLflow (which will now be hosted in the cloud, e.g., an AWS EC2 or managed Databricks tracking server):
   ```python
   # Inside future FastAPI backend
   model = mlflow.pyfunc.load_model("models:/financial_wellness_model/Production")
   encoder = mlflow.artifacts.download_artifacts("models:/financial_wellness_model/Production/encoder.pkl")
   ```
5. **App is Live**: The backend parses React frontend JSON schemas (enforced by Pydantic), applies the downloaded `encoder`, predicts using `model`, and returns the results.

## Approvals & Next Steps

If this overarching technical strategy aligns with your vision for the product:
1. Please confirm this Implementation Plan.
2. I will immediately execute Step 1: Merging the `docker-compose.yml` to set up the unified infrastructure.
3. I will then begin scaffolding the `model-development/src/` architecture and modifying your original Notebook logic into production-grade MLflow-tracked Python modules.
