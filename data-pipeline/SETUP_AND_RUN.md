# SavVio Data Pipeline Setup & Execution Guide

This guide provides step-by-step instructions for setting up the environment, running data ingestion, executing preprocessing pipelines, and managing data versioning with DVC.

> **Note:** This is a temporary guide for the development team. These instructions will be merged into the main `README.md` upon pipeline completion.

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

2. Install dependencies (from repo root or from `data-pipeline/dags` if there is a local `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure pandas, google-cloud-storage, dvc[gs], python-dotenv are installed.)*

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

## 4. Data Validation

The validation script runs data quality and anomaly checks against the raw, processed, and featured datasets. It ensures the data meets schemas and quality standards before it flows further into the pipeline or database.

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

DVC tracks raw, processed, and features under `data-pipeline/dags/data/`. The remote is GCS: `gs://savvio-data-bucket/dvcstore`. Ensure you have set the DVC credential path (see **1. Environment Setup → DVC remote credential path**).

### Basic workflow

All DVC commands below are run from **`data-pipeline/dags/data`** (where the `.dvc` files live).

1. **Track new or updated data**
   After ingestion or preprocessing:
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
   If you get **missing cache files**, the data was never pushed; run preprocessing then `dvc add processed` and `dvc push` (or use `./scripts/backfill-processed-dvc.sh` from repo root).

### Troubleshooting DVC
- **Lock file error:** `rm -f .dvc/tmp/lock` (from repo root).
- **Cache / 401:** Ensure `credentialpath` in `.dvc/config` points to your `savvio-gcp-key.json` (see **1. Environment Setup**).
- **Missing cache:** Run `dvc doctor`; ensure remote URL is `gs://savvio-data-bucket/dvcstore`.

---

## 6. Pipeline Summary

| Step | Command (where to run) | Output |
|------|-------------------------|--------|
| **1. Ingest** | `cd data-pipeline/dags && python3 src/ingestion/run_ingestion.py` | `data-pipeline/dags/data/raw/*` |
| **2. Preprocess** | `cd data-pipeline/dags && python3 src/preprocess/run_preprocessing.py` | `data-pipeline/dags/data/processed/*` |
| **3. Features** | `cd data-pipeline/dags && python3 src/features/run_features.py` | `data-pipeline/dags/data/features/*` |
| **4. Validate** | `cd data-pipeline/dags && python3 src/validation/run_validation.py all` | _Validation Report_ |
| **5. Version** | `cd data-pipeline/dags/data && dvc add raw processed features && dvc push` | `gs://savvio-data-bucket/dvcstore` |

**Team checklist:** Create `data-pipeline/config/`, add `savvio-gcp-key.json` (do not commit), set `GCP_CREDENTIALS_PATH` in `.env`, and run `dvc remote modify gcs credentialpath <path-to-key>` once.

---
