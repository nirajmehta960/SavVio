# SavVio Data Pipeline Setup & Execution Guide

This guide provides step-by-step instructions for setting up the environment, running data ingestion, executing preprocessing pipelines, and managing data versioning with DVC.

> **Note:** This is a temporary guide for the development team. These instructions will be merged into the main `README.md` upon pipeline completion.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+
- `pip` or `uv` (recommended for speed)
- Google Cloud SDK (if interacting with GCS directly)

### Installation
1. Navigate to the pipeline directory:
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
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, ensure you have pandas, google-cloud-storage, dvc, dvc-gcs, python-dotenv installed)*

4. Configure Environment Variables:
   - Create a `.env` file in the project root folder
   - Add the following configurations:
     ```env
     # Google Cloud Configuration
     GCP_PROJECT_ID=your-project-id
     GCS_BUCKET_NAME=savvio-data-bucket
     GCP_CREDENTIALS_PATH=/path/to/your/service-account-key.json
     
     # Pipeline Config
     DATA_DIR=data
     LOG_LEVEL=INFO
     ```

---

## 2. Data Ingestion

The ingestion script downloads raw data from Google Cloud Storage (or API in prod) to `data/raw/`.

**Run Ingestion:**
```bash
# Ensure you are in data-pipeline directory
python3 ingestion-scripts/run_ingestion.py
```

**What this does:**
- Checks `data/raw/` for existing files.
- Downloads `financial_data.csv`, `product_data.jsonl`, and `review_data.jsonl`.
- Verifies file integrity.

---

## 3. Data Preprocessing

We have a unified orchestrator script that runs all preprocessing tasks in the correct order.

**Run All Preprocessing:**
```bash
python3 scripts/run_preprocessing.py
```

**Individual Steps (if needed for debugging):**
- **Financial Data:** `python3 -m scripts.preprocess.financial`
- **Product Data:** `python3 -m scripts.preprocess.product`
- **Review Data:** `python3 -m scripts.preprocess.review`

**Outputs:**
- Processed files are saved to `data/processed/`.
- Logs are printed to the console (and saved to `logs/` if configured).

---

## 4. Data Versioning (DVC)

We use DVC (Data Version Control) to version our raw and processed datasets.

### Basic Workflow

1. **Track New/Modified Data:**
   After running ingestion or preprocessing, if data has changed:
   ```bash
   # Add processed data folder to DVC
   dvc add data/processed
   
   # Add raw data (only if you ingested new raw data)
   dvc add data/raw
   ```

2. **Commit DVC Changes to Git:**
   DVC creates `.dvc` files that point to the actual data. These must be committed to Git.
   ```bash
   git add data/processed.dvc data/raw.dvc
   git commit -m "Update processed data artifacts"
   ```

3. **Push Data to Remote Storage (GCS):**
   This uploads the actual large data files to the shared GCS bucket.
   ```bash
   dvc push
   ```

4. **Pull Data (for other team members):**
   To get the latest data versions committed by teammates:
   ```bash
   git pull origin main
   dvc pull
   ```

### Troubleshooting DVC
- **Lock file error?**
  Run `rm -f .dvc/tmp/lock` if DVC complains about a lock file after a crash.
- **Cache error?**
  Run `dvc doctor` to diagnose issues.

---

## 5. Pipeline Summary

| Step | Script / Command | output |
|------|------------------|--------|
| **1. Ingest** | `python3 ingestion-scripts/run_ingestion.py` | `data/raw/*` |
| **2. Preprocess** | `python3 scripts/run_preprocessing.py` | `data/processed/*` |
| **3. Version** | `dvc add data/processed && dvc push` | `gs://savvio-data-bucket/dvc-store` |

---
