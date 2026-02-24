# SavVio Data Pipeline Setup & Execution Guide

This guide provides step-by-step instructions to **set up the environment and run the pipeline** so that it can be **reproduced on another machine** without errors. It supports the evaluation criteria for **Proper Documentation**, **Reproducibility**, and **Data Version Control (DVC)**.

**See also:** `README.md` for full pipeline design, evaluation-criteria mapping, and phase-by-phase documentation.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+
- `pip` or `uv` (recommended for speed)
- Google Cloud SDK (optional; for direct GCS access)

### GCP credentials and config folder

1. **Create the config folder** in the data-pipeline directory (if it does not exist):
   ```bash
   mkdir -p data-pipeline/config
   ```

2. **Store the GCP service account key** in that folder:
   - Obtain the service account JSON key from your GCP project (IAM & Admin → Service Accounts → Keys).
   - Save it as `data-pipeline/config/savvio-gcp-key.json`.
   - **Do not commit this file to Git.** The folder `data-pipeline/config/` is already in the repo `.gitignore`.

3. **Reference this path** in your `.env` (see below). Use an absolute path in `.env` so ingestion and DVC can find it from any working directory (e.g. `/path/to/SavVio/data-pipeline/config/savvio-gcp-key.json`).

### Installation

1. From the **repository root** (e.g. `SavVio/`), create and activate a virtual environment (you can use repo root or data-pipeline):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies. Use the repo root `requirements.txt` or `data-pipeline/data-requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Or from the data-pipeline directory:
   ```bash
   cd data-pipeline && pip install -r data-requirements.txt
   ```
   Ensure pandas, google-cloud-storage, dvc[gs], great-expectations, python-dotenv, pytest, and other pipeline dependencies are installed.

3. **Configure environment variables**
   - Create a `.env` file at the **repository root** (e.g. `SavVio/.env`).
   - Add the following (use the path to the key you saved in the config folder):
     ```env
     # Google Cloud Configuration
     GCP_PROJECT_ID=your-project-id
     GCS_BUCKET_NAME=savvio-data-bucket
     GCP_CREDENTIALS_PATH=/absolute/path/to/SavVio/data-pipeline/config/savvio-gcp-key.json
     
     # Pipeline Config
     DATA_DIR=data
     LOG_LEVEL=INFO
     ```

### DVC remote credential path

DVC needs the same GCP key to push/pull from GCS. Set the credential path once (from repo root):

```bash
dvc remote modify gcs credentialpath /absolute/path/to/SavVio/data-pipeline/config/savvio-gcp-key.json
```

Use the same path as `GCP_CREDENTIALS_PATH` in your `.env`. After this, `dvc push` and `dvc pull` will use the key without extra env vars. If you get a **401** when running `dvc pull`, run the command above with your actual key path.

---

## 2. Data Ingestion

The ingestion script downloads raw data from Google Cloud Storage (or API in prod) to `data-pipeline/dags/data/raw/`.

**Run ingestion** (from repository root or from `data-pipeline/dags`):
```bash
cd data-pipeline/dags/src/ingestion
python3 run_ingestion.py
```
Or from repo root:
```bash
cd data-pipeline/dags && python3 src/ingestion/run_ingestion.py
```

**What this does:**
- Creates `data/raw/` under `data-pipeline/dags/data/` if needed.
- Downloads `financial_data.csv`, `product_data.jsonl`, and `review_data.jsonl` from GCS.
- Overwrites existing files if re-run.

---

## 3. Data Preprocessing

A single orchestrator runs all preprocessing steps in order. Run it from the **dags** directory so paths resolve correctly.

**Run all preprocessing:**
```bash
cd data-pipeline/dags
python3 src/preprocess/run_preprocessing.py
```

**Individual steps (for debugging):**
- From `data-pipeline/dags`: run the modules under `src/preprocess/` (financial, product, review).

**Outputs:**
- Processed files are written to `data-pipeline/dags/data/processed/`.
- Logs go to the console (and `logs/` if configured).

---

## 4. Data Validation (Schema, Statistics, and Anomaly Detection)

The validation script runs **schema and data quality checks** (Great Expectations) and **anomaly detection** against the raw, processed, and featured datasets. It ensures the data meets expected structure and quality standards before it flows further into the pipeline or database. Anomalies (e.g. missing values, outliers) are detected and reported; the pipeline can be configured to alert on failures.

**Run all validation stages:**
```bash
cd data-pipeline/dags
python3 src/validation/run_validation.py all
```

**Run a specific stage (for debugging):**
- Valid stages: `raw`, `processed`, `features`, `raw_anomalies`, `anomalies`.
```bash
cd data-pipeline/dags
python3 src/validation/run_validation.py processed
```

**Outputs:**
- A summary of PASSED/WARNING/CRITICAL checks per stage is logged to the console.
- The pipeline will halt if any CRITICAL failures are detected.

---

## 5. Data Versioning (DVC)

Data is versioned with **DVC** using three pointer files: **`raw.dvc`**, **`processed.dvc`**, and **`features.dvc`** in `data-pipeline/dags/data/`. Git tracks these `.dvc` files so that data history is maintained alongside code. The remote is GCS: `gs://savvio-data-bucket/dvcstore`. Ensure you have set the DVC credential path (see **1. Environment Setup → DVC remote credential path**).

### Basic workflow

All DVC commands below are run from **`data-pipeline/dags/data`** (where the `.dvc` files live).

1. **Track new or updated data**
   After ingestion, preprocessing, or feature engineering:
   ```bash
   cd data-pipeline/dags/data
   dvc add raw
   dvc add processed
   dvc add features
   ```

2. **Commit DVC pointer files in Git**
   ```bash
   git add raw.dvc processed.dvc features.dvc
   git commit -m "Update data artifacts"
   ```

3. **Push data to GCS**
   ```bash
   dvc push
   ```
   (Uses the credential path set in `.dvc/config`.)

4. **Pull data (for other team members)**
   After cloning or pulling Git:
   ```bash
   cd data-pipeline/dags/data
   dvc pull
   ```
   If you get **401**, set the credential path:  
   `dvc remote modify gcs credentialpath /path/to/data-pipeline/config/savvio-gcp-key.json`  
   If you get **missing cache files**, the data was never pushed; run ingestion/preprocess/features as needed, then `dvc add` and `dvc push`.

### Troubleshooting DVC
- **Lock file error:** `rm -f .dvc/tmp/lock` (from repo root).
- **Cache / 401:** Ensure `credentialpath` in `.dvc/config` points to your `savvio-gcp-key.json` (see **1. Environment Setup**).
- **Missing cache:** Run `dvc doctor`; ensure remote URL is `gs://savvio-data-bucket/dvcstore`.

---

## 6. Bias Detection

The bias detection scripts analyze the financial, product, and review datasets to identify underrepresented groups, data missingness bias, and potential fairness risks that could impact downstream models.

**Run all bias detection checks:**
```bash
cd data-pipeline/dags
python3 src/bias/run_bias.py
```

**Outputs:**
- Terminal logs containing the bias reports for financial, product, and review data. No files are written.
- Flags and warnings for any metrics that do not meet the MVP data representation thresholds.

---

## 7. Airflow Orchestration

The entire data pipeline is orchestrated using Apache Airflow. The DAG `savvio_data_pipeline` runs all steps sequentially and concurrently where possible (from ingestion and preprocessing to bias detection).

**Start the Airflow environment:**
Ensure your local PostgreSQL database is running, then run:
```bash
# From the repository root
docker-compose up -d
```

**Accessing Airflow:**
- **URL:** `http://localhost:8080`
- **Username:** `airflow`
- **Password:** `airflow`

In the Airflow UI, trigger the `savvio_data_pipeline` DAG manually to run the entire pipeline end-to-end.

**Stopping Airflow:**
```bash
docker-compose down
```

---

## 8. Testing

The project includes a comprehensive suite of unit tests using `pytest`. Testing ensures that ingestion, validation, preprocessing, anomaly detection, features, bias analysis, and Airflow DAGs execute correctly.

**Run all tests:**
Make sure you are in your active virtual environment, then:
```bash
# From the data-pipeline directory
pytest tests/ -v
```

**Run specific test modules:**
```bash
pytest tests/ingestion/ -v
pytest tests/features/ -v
pytest tests/test_data_pipeline_airflow.py -v
```

---

## 9. Reproducibility (Full Pipeline on Another Machine)

To replicate the pipeline on a new machine:

1. **Clone the repository** and ensure Git and Python 3.10+ are installed.
2. **Create and activate a virtual environment** (see **1. Environment Setup**).
3. **Install dependencies** (`pip install -r requirements.txt` or `data-requirements.txt`).
4. **Configure environment**: create `.env` with `GCP_CREDENTIALS_PATH`, `GCS_BUCKET_NAME`, etc.; add `data-pipeline/config/savvio-gcp-key.json` (do not commit).
5. **Set DVC remote credential path** once: `dvc remote modify gcs credentialpath <path-to-key>`.
6. **Obtain data** either by running ingestion (`python3 src/ingestion/run_ingestion.py` from `data-pipeline/dags`) or by running `dvc pull` from `data-pipeline/dags/data` if data is already versioned.
7. **Run the pipeline** manually (Steps 2–4 in the Pipeline Summary table) or start Airflow and trigger the DAG.

Following these steps should allow anyone to clone, install, and run the pipeline without errors.

---

## 10. Error Handling and Logging

- **Pipeline failures:** The Airflow DAG uses branching and `EmailOperator` tasks so that when a task fails (e.g. validation, preprocessing), an email alert is sent and the run does not silently succeed. Check the Airflow UI for task status and logs.
- **Validation failures:** Running `python3 src/validation/run_validation.py all` reports PASSED/WARNING/CRITICAL per check. CRITICAL failures should be resolved before proceeding; logs and the validation report indicate which checks failed.
- **Logs:** Each module uses Python `logging` with a consistent format (`[timestamp] [level] [module] message`). Airflow stores per-task logs in the Airflow logs directory. Use these logs to troubleshoot ingestion, preprocessing, feature, or database errors.
- **Common issues:** Missing GCP key or wrong `GCP_CREDENTIALS_PATH` → ingestion fails. Missing DVC credential path → `dvc push`/`dvc pull` return 401. Missing PostgreSQL or wrong DB config → database load tasks fail.

---

## 11. Pipeline Summary

| Step | Command (where to run) | Output |
|------|-------------------------|--------|
| **1. Ingest** | `cd data-pipeline/dags && python3 src/ingestion/run_ingestion.py` | `data-pipeline/dags/data/raw/*` |
| **2. Preprocess** | `cd data-pipeline/dags && python3 src/preprocess/run_preprocessing.py` | `data-pipeline/dags/data/processed/*` |
| **3. Features** | `cd data-pipeline/dags && python3 src/features/run_features.py` | `data-pipeline/dags/data/features/*` |
| **4. Validate** | `cd data-pipeline/dags && python3 src/validation/run_validation.py all` | _Validation Report_ |
| **5. Version** | `cd data-pipeline/dags/data && dvc add raw processed features && dvc push` | `gs://savvio-data-bucket/dvcstore` |
| **6. Bias Check** | `cd data-pipeline/dags && python3 src/bias/run_bias.py` | _Bias Terminal Report_ |
| **7. Airflow** | `docker-compose up -d` (from repo root) | _Pipeline orchestrated in Airflow UI_ |
| **8. Tests** | `pytest tests/ -v` (from data-pipeline dir) | _Pytest pass/fail output_ |

**Team checklist:** Create `data-pipeline/config/`, add `savvio-gcp-key.json` (do not commit), set `GCP_CREDENTIALS_PATH` in `.env`, and run `dvc remote modify gcs credentialpath <path-to-key>` once.

**Code structure:** Pipeline code lives under `data-pipeline/dags/src/` (ingestion, preprocess, features, validation, database, bias). The Airflow DAG is `data-pipeline/dags/data_pipeline_airflow.py`. Data is under `data-pipeline/dags/data/` with DVC pointer files `raw.dvc`, `processed.dvc`, and `features.dvc` for versioning.

---
