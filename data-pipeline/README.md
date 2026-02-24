# SavVio Data Pipeline Implementation Guide

## MLOps Course: Data Pipeline Phase

**Project:** SavVio - AI-Driven Financial Advocacy Tool  
**Team Members:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

---

## Evaluation Criteria Mapping

This pipeline is designed to meet the course evaluation criteria:

| Criterion                                    | How It Is Addressed                                                                                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Proper Documentation**                  | This README, `SETUP_AND_RUN.md`, well-commented code, and the folder structure below.                                                                |
| **2. Modular Syntax and Code**               | Each component lives in `dags/src/` as reusable modules (ingestion, preprocess, features, validation, database, bias).                               |
| **3. Pipeline Orchestration (Airflow DAGs)** | `data_pipeline_airflow.py` implements the full DAG with logical task flow and branching for error handling.                                          |
| **4. Tracking and Logging**                  | Python logging in every module; Airflow task logs; error alerts via `EmailOperator` on failure.                                                      |
| **5. Data Version Control (DVC)**            | Data is versioned with DVC using `raw.dvc`, `processed.dvc`, and `features.dvc` in `dags/data/`; history is maintained in Git. See SETUP_AND_RUN.md. |
| **6. Pipeline Flow Optimization**            | Parallel tasks (ingestion, preprocessing); optimization documented in Phase 19; Airflow Gantt charts available in the UI.                            |
| **7. Schema and Statistics Generation**      | Great Expectations used in validation (raw, processed, features); schema and quality validated over time.                                            |
| **8. Anomaly Detection and Alerts**          | Anomaly validators in `src/validation/anomaly/`; raw and featured anomaly checks; email alerts on validation failure.                                |
| **9. Bias Detection and Mitigation**         | Data slicing in `dags/src/bias/` (financial, product, review); slice analysis and training-time mitigation recommendations.                          |
| **10. Test Modules**                         | Unit tests under `tests/` for ingestion, preprocess, features, validation, database, and DAG; pytest; edge cases covered.                            |
| **11. Reproducibility**                      | `SETUP_AND_RUN.md` provides setup and run instructions so the pipeline can be replicated on another machine.                                         |
| **12. Error Handling and Logging**           | Branching and error-email tasks in the DAG; try/except and validation reports; logs sufficient for troubleshooting.                                  |

---

## Pipeline Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│                     SavVio Data Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SETUP                                                       │
│     └── Environment, Docker, Git, DVC init                      │
│                        ↓                                        │
│  2. DATA COLLECTION (Planning)                                  │
│     └── Identify sources, document requirements, privacy        │
│                        ↓                                        │
│  3. DATA INGESTION                                              │
│     ├── Load Financial data                                     │
│     ├── Load Product data                    [parallel]         │
│     └── Load Review data                     [parallel]         │
│                        ↓                                        │
│  4. VERSION RAW DATA (DVC Checkpoint #1)                        │
│     └── dvc add data/raw/                                       │
│                        ↓                                        │
│  5. SCHEMA & STATISTICS GENERATION                              │
│     └── Define expected structure, compute baseline stats       │
│                        ↓                                        │
│  6. RAW DATA VALIDATION                                         │
│     └── Validate raw data against schema                        │
│                        ↓                                        │
│  7. ANOMALY DETECTION & ALERTS                                  │
│     └── Detect outliers, missing values, trigger alerts         │
│                        ↓                                        │
│  8. DATA PREPROCESSING                                          │
│     └── Clean, transform, standardize                           │
│                        ↓                                        │
│  9. PROCESSED DATA VALIDATION                                   │
│     └── Validate preprocessing didn't break data                │
│                        ↓                                        │
│  10. VERSION PROCESSED DATA (DVC Checkpoint #2)                 │
│      └── dvc add data/processed/                                │
│                        ↓                                        │
│  11. FEATURE ENGINEERING                                        │
│      └── Create derived features (RUS, affordability, sentiment)│
│                        ↓                                        │
│  12. FEATURE VALIDATION                                         │
│      └── Validate feature calculations and ranges               │
│                        ↓                                        │
│  13. VERSION FEATURES (DVC Checkpoint #3)                       │
│      └── dvc add data/features/ (features.dvc)                 │
│                        ↓                                        │
│  14. LOAD TO DATABASE                                           │
│      ├── PostgreSQL (financial, product, review data)           │
│      └── pgvector (product embeddings for RAG)                  │
│                        ↓                                        │
│  15. BIAS DETECTION & MITIGATION                                │
│      └── Slice analysis on features, fairness checks            │
│                        ↓                                        │
│  16. PIPELINE ORCHESTRATION (Airflow DAG)                       │
│      └── Connect all tasks, set dependencies                    │
│                        ↓                                        │
│  17. TESTING                                                    │
│      └── Unit tests for each component                          │
│                        ↓                                        │
│  18. TRACKING, LOGGING & MONITORING                             │
│      └── Logging, metrics, dashboards                           │
│                        ↓                                        │
│  19. PIPELINE OPTIMIZATION                                      │
│      └── Gantt analysis, parallelization                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Tools by Phase

| Phase                | Primary Tools                 | Alternatives                            | Airflow Integration            |
| -------------------- | ----------------------------- | --------------------------------------- | ------------------------------ |
| Setup                | Docker, Git, DVC              | —                                       | —                              |
| Data Collection      | Documentation                 | —                                       | —                              |
| Ingestion            | Pandas, API clients           | Polars                                  | PythonOperator                 |
| Schema/Stats         | Great Expectations            | Pandera, ydata-profiling, custom Python | PythonOperator                 |
| Raw Validation       | Great Expectations            | Pandera, Pydantic, custom validators    | PythonOperator                 |
| Anomaly Detection    | Great Expectations, Evidently | Custom Python (IQR/z-score)             | PythonOperator + EmailOperator |
| Preprocessing        | Pandas                        | Polars                                  | PythonOperator                 |
| Processed Validation | Great Expectations            | Pandera, custom validators              | PythonOperator                 |
| Features             | Pandas, NumPy                 | Polars                                  | PythonOperator                 |
| Feature Validation   | Great Expectations            | Pandera, custom validators              | PythonOperator                 |
| Versioning           | DVC, GCP Cloud Storage        | —                                       | BashOperator                   |
| Load to Database     | SQLAlchemy, psycopg2          | pandas.to_sql                           | PythonOperator                 |
| Embeddings           | Sentence-Transformers, OpenAI | LangChain embeddings                    | PythonOperator                 |
| Bias Detection       | Fairlearn                     | Custom Pandas slicing, AIF360           | PythonOperator                 |
| Orchestration        | Apache Airflow                | —                                       | Native                         |
| Testing              | pytest                        | unittest                                | —                              |
| Monitoring           | Python logging, Airflow UI    | GCP Cloud Logging, Grafana              | Native                         |
| Optimization         | Airflow Gantt                 | cProfile                                | Native                         |

---

## Table of Contents

1. [Phase 1: Project Setup & Environment Configuration](#phase-1-project-setup--environment-configuration)
2. [Phase 2: Data Collection & Planning](#phase-2-data-collection--planning)
3. [Phase 3: Data Ingestion](#phase-3-data-ingestion)
4. [Phase 4: Version Raw Data (DVC Checkpoint #1)](#phase-4-version-raw-data-dvc-checkpoint-1)
5. [Phase 5: Schema & Statistics Generation](#phase-5-schema--statistics-generation)
6. [Phase 6: Raw Data Validation](#phase-6-raw-data-validation)
7. [Phase 7: Anomaly Detection & Alerts](#phase-7-anomaly-detection--alerts)
8. [Phase 8: Data Preprocessing & Transformation](#phase-8-data-preprocessing--transformation)
9. [Phase 9: Processed Data Validation](#phase-9-processed-data-validation)
10. [Phase 10: Version Processed Data (DVC Checkpoint #2)](#phase-10-version-processed-data-dvc-checkpoint-2)
11. [Phase 11: Feature Engineering](#phase-11-feature-engineering)
12. [Phase 12: Feature Validation](#phase-12-feature-validation)
13. [Phase 13: Version Features (DVC Checkpoint #3)](#phase-13-version-features-dvc-checkpoint-3)
14. [Phase 14: Load to Database](#phase-14-load-to-database)
15. [Phase 15: Bias Detection & Mitigation](#phase-15-bias-detection--mitigation)
16. [Phase 16: Pipeline Orchestration (Airflow DAGs)](#phase-16-pipeline-orchestration-airflow-dags)
17. [Phase 17: Testing](#phase-17-testing)
18. [Phase 18: Tracking, Logging & Monitoring](#phase-18-tracking-logging--monitoring)
19. [Phase 19: Pipeline Optimization](#phase-19-pipeline-optimization)

---

## Phase 1: Project Setup & Environment Configuration

### Objective

Establish the foundational project structure, dependencies, and development environment before any data work begins.

### Steps

1. **Create folder structure** following the required format (actual implementation layout):

   ```
   SavVio/
   ├── data-pipeline/
   │   ├── README.md              # This file
   │   ├── SETUP_AND_RUN.md       # Setup and run instructions (reproducibility)
   │   ├── data-requirements.txt  # Python dependencies (or use repo root requirements.txt)
   │   ├── config/                # Configuration (e.g. database, GCP key path)
   │   ├── logs/                  # Pipeline execution logs
   │   ├── tests/                 # Unit tests (pytest)
   │   │   ├── ingestion/
   │   │   ├── preprocess/
   │   │   ├── features/
   │   │   ├── validation/
   │   │   ├── database/
   │   │   ├── bias/
   │   │   └── test_data_pipeline_airflow.py
   │   └── dags/                  # Airflow DAG and pipeline code
   │       ├── data_pipeline_airflow.py   # Main DAG definition
   │       ├── data/
   │       │   ├── raw/           # Raw data (financial_data.csv, product_data.jsonl, review_data.jsonl)
   │       │   ├── processed/     # Preprocessed outputs
   │       │   ├── features/      # Feature-engineered outputs
   │       │   ├── raw.dvc        # DVC pointer for raw (versioned in Git)
   │       │   ├── processed.dvc  # DVC pointer for processed
   │       │   └── features.dvc   # DVC pointer for features
   │       └── src/
   │           ├── ingestion/     # Data acquisition (GCS, APIs)
   │           │   ├── run_ingestion.py
   │           │   ├── gcs_loader.py
   │           │   └── ...
   │           ├── preprocess/    # Cleaning and transformation
   │           │   ├── run_preprocessing.py
   │           │   ├── financial.py
   │           │   ├── product.py
   │           │   ├── review.py
   │           │   └── ...
   │           ├── features/      # Feature engineering
   │           │   ├── run_features.py
   │           │   ├── financial_features.py
   │           │   ├── product_review_features.py
   │           │   └── ...
   │           ├── validation/   # Schema, stats, anomaly checks (Great Expectations)
   │           │   ├── run_validation.py
   │           │   ├── validate/
   │           │   ├── anomaly/
   │           │   └── ...
   │           ├── database/      # PostgreSQL and pgvector load
   │           │   ├── run_database.py
   │           │   ├── upload_to_db.py
   │           │   ├── vector_embed.py
   │           │   └── ...
   │           └── bias/          # Bias detection (data slicing)
   │               ├── run_bias.py
   │               ├── financial_bias.py
   │               ├── product_bias.py
   │               ├── review_bias.py
   │               └── utils.py
   ```

2. **Set up Python environment**
   - Create `requirements.txt` or `environment.yml`
   - Include: pandas, polars, apache-airflow, dvc, great-expectations, pytest, evidently, fairlearn, sqlalchemy, psycopg2-binary, sentence-transformers

3. **Configure Docker environment for Airflow**
   - Use official Apache Airflow Docker Compose setup
   - Customize for SavVio (mount data directories, install dependencies)
   - Ensure all team members can replicate the environment

4. **Initialize version control**
   - Git repository setup
   - Create `.gitignore` (exclude data files, logs, credentials, `__pycache__`)
   - Initialize DVC: `dvc init`
   - Configure DVC remote: `dvc remote add -d gcs gs://savvio-data-bucket`

5. **Configure database connections**
   - Create `config/database.yaml` with connection settings
   - Local (Development): Local PostgreSQL instance
   - Cloud (Production): GCP Cloud SQL for PostgreSQL

### Tools & Alternatives

| Tool                    | Purpose               | Alternative              |
| ----------------------- | --------------------- | ------------------------ |
| Docker + Docker Compose | Containerized Airflow | Podman                   |
| Git                     | Code version control  | —                        |
| DVC                     | Data version control  | LakeFS                   |
| Python 3.12+            | Runtime               | —                        |
| PostgreSQL              | Local database        | SQLite (lighter for dev) |

---

## Phase 2: Data Collection & Planning

### Objective

Identify, understand, and document the data sources needed for SavVio's purchase guardrail functionality. This phase is about planning and documentation before actual data ingestion.

### Steps

1. **Translate user needs into data needs**
   - Users: Consumers making purchase decisions
   - User Need: Make informed, responsible purchase decisions
   - System Need: Financial health data + Product information + Product reviews

2. **Document data sources for SavVio**

   **Financial Data:**
   | Attribute | Details |
   |-----------|---------|
   | Source | Personal Finance dataset |
   | Format | CSV |
   | Size | ~X records (to be confirmed) |
   | Update Frequency | Static (for academic project) |

   **Product Data:**
   | Attribute | Details |
   |-----------|---------|
   | Source | Product dataset |
   | Format | JSONL |
   | Size | ~X records (to be confirmed) |
   | Update Frequency | Static (for academic project) |

   **Review Data:**
   | Attribute | Details |
   |-----------|---------|
   | Source | Product review dataset |
   | Format | JSONL |
   | Size | ~X records (to be confirmed) |
   | Update Frequency | Static or event-driven |

3. **Document expected data fields**

   **Financial Data Fields:**
   | Field | Type | Description | Required |
   |-------|------|-------------|----------|
   | monthly_income | float | User's monthly income | Yes |
   | rent | float | Monthly rent payment | No |
   | recurring_bills | float | Subscriptions, utilities, etc. | No |
   | savings_balance | float | Current savings amount | No |
   | debt_obligations | float | Monthly debt payments | No |

   **Product Data Fields:**
   | Field | Type | Description | Required |
   |-------|------|-------------|----------|
   | product_name | string | Name of the product | Yes |
   | category | string | Product category | Yes |
   | price | float | Product price | Yes |
   | specifications | string | Product details/specs | No |
   | description | string | Product description (for embeddings) | No |

   **Review Data Fields:**
   | Field | Type | Description | Required |
   |-------|------|-------------|----------|
   | product_id | integer/string | Foreign key to product | Yes |
   | reviewer_id | string | Anonymized reviewer identifier | Yes |
   | rating | float | Product rating (1-5) | Yes |
   | text | string | Review text/feedback | No |
   | helpful_count | integer | Number of "helpful" votes | No |
   | date | timestamp | Review submission date | No |

4. **Document data privacy measures**
   - Masking/hashing user identifiers
   - Encryption for financial snapshots
   - Read-only access (no transactional capabilities)
   - Compliance with data privacy principles
   - Anonymized review data (no PII)

5. **Create Data Card documentation**

   | Attribute             | Description                                                |
   | --------------------- | ---------------------------------------------------------- |
   | Data Type             | Medium-sized datasets (thousands of records)               |
   | Financial Data Fields | Monthly income, rent, recurring bills, savings, debt       |
   | Product Data Fields   | Product name, category, price, specifications, description |
   | Review Data Fields    | Product ID, rating, text, helpfulness metrics              |
   | Estimated Size        | Small to medium (hundreds to thousands of records each)    |
   | Update Frequency      | Periodic or event-driven (simulated)                       |
   | Sensitive Fields      | Income, savings, expenses (masked and encrypted)           |

### Outputs

- `docs/data_sources.md` — Documented data sources and fields
- `docs/data_card.md` — Formal data card
- `docs/privacy_measures.md` — Privacy and security documentation

---

## Phase 3: Data Ingestion

### Objective

Download data from external sources and load it into the pipeline system in a consistent, reproducible manner.

### Steps

1. **Create modular ingestion package structure**

   ```
   scripts/ingest/
   ├── __init__.py
   ├── api_loader.py         # API utilities
   ├── gcs_loader.py         # GCS utilities
   ├── config.py             # Configuration
   ├── financial.py          # Financial data ingestion
   ├── product.py            # Product data ingestion
   ├── review.py             # Review data ingestion
   ├── utils.py              # Shared utilities
   └── run_ingestion.py      # Main entry point
   ```

2. **Implement shared utilities** (`ingest/utils.py`)
   - Logging configuration
   - Error handling helpers
   - File path management
   - API wrapper functions

3. **Implement financial data ingestion** (`ingest/financial.py`)
   - `download_financial_data()` — Fetches from source
   - `load_financial_data()` — Reads CSV into DataFrame
   - `save_raw_financial_data()` — Saves to `data/raw/`

4. **Implement product data ingestion** (`ingest/product.py`)
   - `download_product_data()` — Fetches from source
   - `load_product_data()` — Reads JSONL into DataFrame
   - `save_raw_product_data()` — Saves to `data/raw/`

5. **Implement review data ingestion** (`ingest/review.py`)
   - `download_review_data()` — Fetches from source
   - `load_review_data()` — Reads JSONL into DataFrame
   - `save_raw_review_data()` — Saves to `data/raw/`

6. **Store raw data in original format**
   - Save to `data/raw/financial.csv`
   - Save to `data/raw/products.jsonl`
   - Save to `data/raw/reviews.jsonl`

7. **Log ingestion metadata**
   - Timestamp, record counts, checksums, errors

### Tools & Alternatives

| Tool        | Purpose       | Alternative          | When to Use Alternative                |
| ----------- | ------------- | -------------------- | -------------------------------------- |
| Pandas      | Data loading  | Polars               | If performance issues with large files |
| API clients | Data fetching | Direct HTTP requests | Custom control needed                  |

### Airflow Integration

```python
ingest_financial = PythonOperator(
    task_id='ingest_financial_data',
    python_callable=financial.load_financial_data,
    dag=dag
)

ingest_products = PythonOperator(
    task_id='ingest_product_data',
    python_callable=product.load_product_data,
    dag=dag
)

ingest_reviews = PythonOperator(
    task_id='ingest_review_data',
    python_callable=review.load_review_data,
    dag=dag
)

# Parallel execution
[ingest_financial, ingest_products, ingest_reviews] >> next_task
```

---

## Phase 4: Version Raw Data (DVC Checkpoint #1)

### Objective

Version control the raw ingested data to ensure reproducibility and enable rollback if needed. Data is versioned using DVC with separate `.dvc` pointer files for raw, processed, and features (history maintained in Git).

### Steps

1. **Add raw data to DVC tracking** (run from `data-pipeline/dags/data`)

   ```bash
   cd data-pipeline/dags/data
   dvc add raw
   ```

   This creates `raw.dvc` (and updates `.gitignore` for the raw directory).

2. **Commit .dvc files to Git**

   ```bash
   git add raw.dvc
   git commit -m "Add raw data v1.0"
   ```

3. **Push data to remote storage**

   ```bash
   dvc push
   ```

4. **Tag the version**
   ```bash
   git tag -a "data-raw-v1.0" -m "Initial raw data ingestion"
   ```

### Tools & Alternatives

| Tool              | Purpose         | Alternative        |
| ----------------- | --------------- | ------------------ |
| DVC               | Data versioning | LakeFS, Git LFS    |
| GCP Cloud Storage | Remote storage  | AWS S3, Azure Blob |

### Why Version Here?

- Captures original data before any modifications
- Enables rollback if preprocessing introduces errors
- Provides "source of truth" for debugging

---

## Phase 5: Schema & Statistics Generation

### Objective

Define the expected structure (data schema) of your datasets and compute baseline statistics. This is NOT about database tables — it's about defining what valid data looks like.

> **Note:** "Data Schema" = expected structure/rules for a dataset (columns, types, constraints).  
> "Database Schema" = tables, columns, relationships in a database (covered in Phase 14).

### Steps

1. **Define data schema (expected structure)**

   **Financial Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | monthly_income | float | No | >= 0 |
   | rent | float | Yes | >= 0 |
   | recurring_bills | float | Yes | >= 0 |
   | savings_balance | float | Yes | >= 0 |
   | debt_obligations | float | Yes | >= 0 |

   **Product Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | product_name | string | No | non-empty |
   | category | string | No | valid category list |
   | price | float | No | > 0 |
   | specifications | string | Yes | — |
   | description | string | Yes | — |

   **Review Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | product_id | integer/string | No | foreign key valid |
   | reviewer_id | string | No | non-empty |
   | rating | float | No | 1-5 range |
   | text | string | Yes | — |
   | helpful_count | integer | Yes | >= 0 |
   | date | timestamp | Yes | valid date |

2. **Compute baseline statistics**
   - Numeric: min, max, mean, median, std, percentiles
   - Categorical: unique values, frequency distribution
   - All columns: null percentage, data type distribution

3. **Create expectation suites (Great Expectations)**
   - `expect_column_to_exist`
   - `expect_column_values_to_be_of_type`
   - `expect_column_values_to_not_be_null`
   - `expect_column_values_to_be_between`
   - `expect_column_values_to_be_in_set`

4. **Store artifacts**
   - `config/schemas/financial_schema.json`
   - `config/schemas/product_schema.json`
   - `config/schemas/review_schema.json`
   - `config/statistics/baseline_stats.json`

### Tools & Alternatives

| Tool               | Purpose               | Alternative       | When to Use Alternative  |
| ------------------ | --------------------- | ----------------- | ------------------------ |
| Great Expectations | Schema & expectations | Pandera           | Lighter weight, Pythonic |
| ydata-profiling    | Auto statistics       | Pandas describe() | Quick exploration        |
| Custom Python      | Full control          | —                 | Simple schemas           |

---

## Phase 6: Raw Data Validation

### Objective

Validate raw ingested data against the schema defined in Phase 5 to catch ingestion errors early.

### Steps

1. **Run validation checkpoint on raw data**
   - Load expectation suite
   - Execute against `data/raw/` files
   - Generate validation results

2. **Validation rules**

   **Financial Data:**
   - All expected columns exist
   - Data types are correct
   - Required fields not null
   - Values within expected ranges

   **Product Data:**
   - All expected columns exist
   - Price is positive
   - Product name not empty

   **Review Data:**
   - All expected columns exist
   - Rating between 1-5
   - Product ID references valid product
   - Reviewer ID not empty

3. **Handle validation failures**
   - Log errors with details
   - Quarantine invalid records to `data/quarantine/`
   - Generate HTML report (Great Expectations Data Docs)
   - Critical failures halt pipeline

4. **Store results**
   - `logs/validation/raw_validation_YYYYMMDD.json`

### Tools & Alternatives

| Tool               | Purpose                | Alternative       | When to Use Alternative      |
| ------------------ | ---------------------- | ----------------- | ---------------------------- |
| Great Expectations | Validation checkpoints | Pandera           | Simpler DataFrame validation |
| —                  | —                      | Pydantic          | Row-level/JSON validation    |
| —                  | —                      | Custom validators | Maximum flexibility          |

---

## Phase 7: Anomaly Detection & Alerts

### Objective

Detect data anomalies (outliers, suspicious patterns) and trigger alerts when issues are found.

### Steps

1. **Define anomaly rules**

   **Financial Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Income = 0 | Rule-based |
   | Expenses > 2x income | Rule-based |
   | Negative monetary values | Rule-based |
   | Extreme outliers | IQR or z-score |

   **Product Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Price <= 0 | Rule-based |
   | Price > $100,000 | Threshold |
   | Missing name with valid price | Rule-based |

   **Review Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Rating outside 1-5 | Rule-based |
   | Product ID doesn't exist | Referential check |
   | Duplicate reviews | Deduplication |
   | Extreme helpful counts | Outlier detection |

2. **Implement detection**
   - Statistical: IQR method, z-score
   - Rule-based: business logic checks

3. **Configure alerts**

   | Severity | Condition      | Action       |
   | -------- | -------------- | ------------ |
   | INFO     | Minor outliers | Log only     |
   | WARNING  | >5% issues     | Email alert  |
   | CRITICAL | Data corrupted | Email + halt |

4. **Quarantine suspicious records**

### Tools & Alternatives

| Tool                         | Purpose               | Alternative    | When to Use Alternative |
| ---------------------------- | --------------------- | -------------- | ----------------------- |
| Great Expectations           | Rule-based detection  | Custom Python  | More control            |
| Evidently AI                 | Statistical detection | PyOD           | More algorithms         |
| Airflow EmailOperator        | Email alerts          | SendGrid, SMTP | External email service  |
| Airflow SlackWebhookOperator | Slack alerts          | Custom webhook | Other chat platforms    |

---

## Phase 8: Data Preprocessing & Transformation

### Objective

Clean, transform, and standardize validated data into a consistent format ready for feature engineering.

### Steps

1. **Create modular preprocessing package**

   ```
   scripts/preprocess/
   ├── __init__.py
   ├── financial.py
   ├── product.py
   ├── review.py
   ├── utils.py
   └── run_preprocessing.py
   ```

2. **Data cleaning**

   **Financial Data:**
   - Handle missing values (impute median or flag)
   - Standardize transaction categories
   - Remove duplicates

   **Product Data:**
   - Flatten nested JSONL objects if present
   - Remove duplicates
   - Standardize price format
   - Clean product names

   **Review Data:**
   - Flatten nested JSONL objects if present
   - Remove duplicate reviews
   - Clean text (trim whitespace, handle special chars)
   - Validate ratings are in 1-5 range
   - Handle missing helpful_count

3. **Data transformation**

   **Financial Data:**
   - Convert to monthly format (annual ÷ 12, weekly × 4.33)
   - Categorize expenses
   - Calculate total fixed expenses

   **Product Data:**
   - Standardize category names
   - Clean descriptions for embedding

   **Review Data:**
   - Normalize text for processing
   - Aggregate ratings by product

4. **Save processed data**
   - `data/processed/financial_processed.csv`
   - `data/processed/products_processed.csv`
   - `data/processed/reviews_processed.csv`

### Tools & Alternatives

| Tool   | Purpose              | Alternative | When to Use Alternative       |
| ------ | -------------------- | ----------- | ----------------------------- |
| Pandas | Data manipulation    | Polars      | 10-100x faster for large data |
| NumPy  | Numerical operations | —           | —                             |

---

## Phase 9: Processed Data Validation

### Objective

Validate that preprocessing transformations didn't break the data or introduce errors.

### Steps

1. **Define processed data expectations**
   - All original records accounted for (minus quarantined)
   - No new null values introduced
   - Transformations applied correctly (e.g., monthly format)
   - No duplicate records

2. **Run validation checkpoint**
   - Validate `data/processed/` files
   - Compare record counts: raw vs processed
   - Verify transformation logic

3. **Specific checks for SavVio**
   - Financial values in monthly format
   - Expense categories standardized
   - Product prices positive and reasonable
   - Review ratings normalized and valid

4. **Handle failures**
   - Log transformation errors
   - Alert if significant data loss
   - Option to rollback to raw data

### Tools & Alternatives

| Tool               | Purpose               | Alternative |
| ------------------ | --------------------- | ----------- |
| Great Expectations | Validation            | Pandera     |
| Custom Python      | Transformation checks | —           |

---

## Phase 10: Version Processed Data (DVC Checkpoint #2)

### Objective

Version control the processed data to track transformations.

### Steps

1. **Add processed data to DVC** (run from `data-pipeline/dags/data`)

   ```bash
   cd data-pipeline/dags/data
   dvc add processed
   ```

   This creates/updates `processed.dvc`.

2. **Commit and push**

   ```bash
   git add processed.dvc
   git commit -m "Add processed data v1.0"
   dvc push
   ```

3. **Tag the version**
   ```bash
   git tag -a "data-processed-v1.0" -m "Processed data"
   ```

### Why Version Here?

- Captures cleaned data before feature engineering
- If features have bugs, no need to re-preprocess
- Enables raw vs processed comparison

---

## Phase 11: Feature Engineering

### Objective

Create meaningful features within each data track — financial health features per user and quality features per product. Affordability metrics (which require both user and product data) are computed at inference time by the Deterministic Financial Logic Engine, not pre-computed in the pipeline.

### Steps

1. **Create modular feature engineering package**

   ```
   scripts/features/
   ├── __init__.py
   ├── financial_features.py
   ├── review_features.py
   ├── utils.py
   └── run_features.py
   ```

2. **Financial health features** (`financial_features.py`)

   Input: `data/processed/financial_preprocessed.csv`
   Output: `data/features/financial_featured.csv`

   | Feature                        | Formula                    | Purpose                           |
   | ------------------------------ | -------------------------- | --------------------------------- |
   | `discretionary_income`         | income - (expenses + emi)  | Available money after obligations |
   | `debt_to_income_ratio`         | emi / income               | Debt burden indicator             |
   | `savings_to_income_ratio`      | savings / income           | Savings health indicator          |
   | `monthly_expense_burden_ratio` | (expenses + emi) / income  | Spending pattern                  |
   | `emergency_fund_months`        | savings / (expenses + emi) | Safety buffer in months           |

3. **Product quality features** (`review_features.py`)

   Input: `data/processed/review_preprocessed.jsonl`
   Output: `data/features/product_rating_variance.csv` (one row per product)

   | Feature           | Formula                 | Purpose                 |
   | ----------------- | ----------------------- | ----------------------- |
   | `rating_variance` | std(rating) per product | Rating consensus signal |

   > **Note:** `average_rating` and `rating_number` (equivalent to `num_reviews`) already exist in the product metadata. The only feature requiring individual review data is `rating_variance`, which measures how polarized opinions are about a product.

4. **Handle edge cases**
   - Zero income: ratios set to NaN (XGBoost handles natively)
   - Division by zero: safe handling with NaN defaults
   - Single-review products: rating_variance defaults to 0.0

5. **Outputs**
   - `data/features/financial_featured.csv` — Financial profiles enriched with health metrics
   - `data/features/product_rating_variance.csv` — Product-level rating variance
   - `config/feature_definitions.json` — Feature metadata documentation

### Tools & Alternatives

| Tool   | Purpose          | Alternative | When to Use Alternative |
| ------ | ---------------- | ----------- | ----------------------- |
| Pandas | Feature creation | Polars      | Large datasets          |
| NumPy  | Numerical ops    | —           | —                       |

---

## Phase 12: Feature Validation

### Objective

Validate that engineered features are correctly calculated and within expected ranges.

### Steps

1. **Define feature expectations**

   **Financial Features:**
   | Feature | Expected Range | Validation |
   |---------|---------------|------------|
   | `discretionary_income` | Can be negative | Check not NaN where income > 0 |
   | `debt_to_income_ratio` | 0 to ~2 | Flag if > 2; NaN only when income = 0 |
   | `savings_to_income_ratio` | 0 to ~1 | Flag if > 1; NaN only when income = 0 |
   | `monthly_expense_burden_ratio` | 0 to ~1 | Flag if > 1 |
   | `emergency_fund_months` | >= 0 | NaN only when obligations = 0 |

   **Product Quality Features:**
   | Feature | Expected Range | Validation |
   |---------|---------------|------------|
   | `rating_variance` | >= 0 | Check not negative; 0.0 for single-review products |

2. **Run validation**
   - No unexpected NaN or Inf values
   - Ratios within reasonable bounds
   - All expected features present
   - Product count in variance output matches product count in reviews

3. **Cross-validate calculations**
   - Sample records: manually verify formulas
   - Edge cases: zero income, single review products

4. **Handle failures**
   - Log calculation errors
   - Identify problematic source records

### Tools & Alternatives

| Tool               | Purpose              | Alternative |
| ------------------ | -------------------- | ----------- |
| Great Expectations | Feature validation   | Pandera     |
| Custom Python      | Formula verification | —           |

---

## Phase 13: Version Features (DVC Checkpoint #3)

### Objective

Version control the feature-engineered data for reproducibility.

### Steps

1. **Add features to DVC** (run from `data-pipeline/dags/data`)

   ```bash
   cd data-pipeline/dags/data
   dvc add features
   ```

   This creates/updates `features.dvc`.

2. **Commit and push**

   ```bash
   git add features.dvc
   git commit -m "Add features v1.0"
   dvc push
   ```

3. **Tag the version** (optional)
   ```bash
   git tag -a "data-features-v1.0" -m "Feature-engineered data"
   ```

### Why Version Here?

- Ties financial health features to model training versions
- Enables comparison of feature distributions across pipeline runs
- Experiment reproducibility

---

## Phase 14: Load to Database

### Objective

Load processed data, engineered features, and product embeddings into PostgreSQL (relational) and pgvector (vector) databases for the SavVio application.

### Environment Configuration

| Environment | Database                           | Connection      |
| ----------- | ---------------------------------- | --------------- |
| Development | Local PostgreSQL + pgvector        | localhost:5432  |
| Production  | GCP Cloud SQL + pgvector extension | Cloud SQL proxy |

### Steps

1. **Identify datasets to load**
   - `data/features/financial_featured.csv` — Financial profiles with health metrics
   - `data/processed/product_preprocessed.jsonl` — Product catalog
   - `data/features/product_rating_variance.csv` — Rating variance per product (merged onto products during load)
   - `data/processed/review_preprocessed.jsonl` — Individual reviews

2. **Create database loading package**

   ```
   scripts/database/
   ├── __init__.py
   ├── postgres_loader.py    # Relational data
   ├── vector_loader.py      # Embeddings
   ├── models.py             # SQLAlchemy models
   └── utils.py              # Connection helpers
   ```

3. **Define database schema (tables)**

   **PostgreSQL Tables:**

   ```sql
   -- Financial profiles with pre-computed features
   CREATE TABLE financial_profiles (
       id SERIAL PRIMARY KEY,
       user_id VARCHAR(255) UNIQUE,
       monthly_income DECIMAL(12,2),
       monthly_expenses DECIMAL(12,2),
       monthly_emi DECIMAL(12,2),
       savings_balance DECIMAL(12,2),
       discretionary_income DECIMAL(12,2),
       debt_to_income_ratio DECIMAL(5,4),
       savings_to_income_ratio DECIMAL(5,4),
       monthly_expense_burden_ratio DECIMAL(5,4),
       emergency_fund_months DECIMAL(8,2),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Products with quality signal from reviews
   CREATE TABLE products (
       id SERIAL PRIMARY KEY,
       product_id VARCHAR(255) UNIQUE,
       product_name VARCHAR(500),
       category VARCHAR(100),
       price DECIMAL(12,2),
       average_rating DECIMAL(2,1),
       rating_number INTEGER,
       rating_variance DECIMAL(5,4),     -- Merged from review features
       specifications TEXT,
       description TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Individual reviews (for RAG context retrieval)
   CREATE TABLE reviews (
       id SERIAL PRIMARY KEY,
       product_id VARCHAR(255) REFERENCES products(product_id),
       reviewer_id VARCHAR(255),
       rating DECIMAL(2,1),
       text TEXT,
       helpful_vote INTEGER DEFAULT 0,
       review_date TIMESTAMP,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

   **pgvector Table (for embeddings):**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;

   CREATE TABLE product_embeddings (
       id SERIAL PRIMARY KEY,
       product_id VARCHAR(255) REFERENCES products(product_id),
       embedding vector(384),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE INDEX ON product_embeddings
   USING ivfflat (embedding vector_cosine_ops);
   ```

4. **Merge rating_variance onto products during load**

   This is where `product_rating_variance.csv` gets attached to products:

   ```python
   # In postgres_loader.py
   def load_products(products_path, rating_variance_path, engine):
       """Load products with rating_variance merged from review features."""
       products_df = pd.read_json(products_path, lines=True)
       variance_df = pd.read_csv(rating_variance_path)

       # Left join: keep all products, attach variance where available.
       products_df = pd.merge(
           products_df,
           variance_df,
           on="product_id",
           how="left"
       )

       # Products with no reviews get rating_variance = 0.0.
       products_df["rating_variance"] = products_df["rating_variance"].fillna(0.0)

       products_df.to_sql("products", engine, if_exists="replace", index=False)
   ```

5. **Implement data loaders**

   **postgres_loader.py:**
   - `load_financial_profiles(df, engine)` — Load financial profiles with features
   - `load_products(products_path, rating_variance_path, engine)` — Load products with rating_variance merged
   - `load_reviews(df, engine)` — Load individual reviews
   - `get_engine(env)` — Get connection based on environment

   **vector_loader.py:**
   - `generate_embeddings(texts, model)` — Create embeddings from product descriptions
   - `load_embeddings(product_ids, embeddings, engine)` — Store in pgvector

6. **Environment-based configuration**

   ```yaml
   # config/database.yaml
   development:
     host: localhost
     port: 5432
     database: savvio_dev
     user: ${DB_USER}
     password: ${DB_PASSWORD}

   production:
     host: /cloudsql/project:region:instance
     port: 5432
     database: savvio_prod
     user: ${DB_USER}
     password: ${DB_PASSWORD}
   ```

7. **Load data**
   - Truncate existing data (or upsert strategy)
   - Load financial profiles (with pre-computed features)
   - Load products (with rating_variance merged)
   - Load reviews
   - Generate and load embeddings

### Tools & Alternatives

| Tool                  | Purpose           | Alternative                       | When to Use Alternative |
| --------------------- | ----------------- | --------------------------------- | ----------------------- |
| SQLAlchemy            | ORM & connection  | psycopg2 direct                   | Simpler queries         |
| psycopg2              | PostgreSQL driver | asyncpg                           | Async operations        |
| Sentence-Transformers | Embeddings        | OpenAI API                        | Higher quality, cost    |
| pgvector              | Vector storage    | Pinecone, Vertex AI Vector Search | Managed service         |

### Airflow Integration

```python
load_financial = PythonOperator(
    task_id='load_financial_profiles',
    python_callable=postgres_loader.load_financial_profiles,
    dag=dag
)

load_products = PythonOperator(
    task_id='load_products',
    python_callable=postgres_loader.load_products,
    op_kwargs={'rating_variance_path': 'data/features/product_rating_variance.csv'},
    dag=dag
)

load_reviews = PythonOperator(
    task_id='load_reviews',
    python_callable=postgres_loader.load_reviews,
    dag=dag
)

generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=vector_loader.generate_and_load,
    dag=dag
)

[load_financial, load_products, load_reviews] >> generate_embeddings
```

---

## Phase 15: Bias Detection & Mitigation

### Objective

Detect and mitigate data representation bias across financial profiles and product/review data. The goal is to ensure the pipeline produces balanced, representative data so downstream models and the decision engine don't systematically disadvantage any subgroup.

### Why Two Separate Tracks?

Each data track is analyzed independently because they represent fundamentally different populations (users vs. products) and carry different bias risks:

| Track          | Population         | Bias Risk                                                                                         |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| Financial      | User profiles      | Underrepresentation of financially vulnerable users — the people SavVio is designed to help most  |
| Product/Review | Products & reviews | Skewed category coverage, price range gaps, or unreliable quality signals for low-review products |

> **Note:** Decision outcome bias (e.g., does the Green/Yellow/Red recommendation system unfairly penalize low-income users?) is tested in Phase 3: Model Development, once the Deterministic Financial Logic Engine and affordability calculations exist. The data pipeline focuses on **data representation bias** only.

### Steps

1. **Financial data — Slice analysis**

   | Slice Dimension       | Groups                                    | What to Check                           |
   | --------------------- | ----------------------------------------- | --------------------------------------- |
   | Income bracket        | Low (<$3k), Medium ($3k-$7k), High (>$7k) | Sufficient low-income representation?   |
   | Debt-to-income ratio  | Low (<0.2), Medium (0.2-0.4), High (>0.4) | Balanced coverage across debt levels?   |
   | Expense burden        | Low (<0.5), Medium (0.5-0.8), High (>0.8) | Are high-burden users underrepresented? |
   | Emergency fund months | Critical (<1), Low (1-3), Healthy (3+)    | Enough financially stressed profiles?   |

   **Analysis:**
   - Count records per slice — flag slices with <10% of total records
   - Compare feature distributions across slices (mean, median, std of each financial metric)
   - Identify if any slice has significantly different feature distributions that could bias model training

   **Why this matters for SavVio:**
   If the financial data skews toward high-income, healthy profiles, the model won't learn effective decision boundaries for users who are financially vulnerable. These are the users most likely to benefit from a "Red Light" recommendation, and the system must work well for them.

2. **Product/review data — Slice analysis**

   | Slice Dimension               | Groups                                         | What to Check                                         |
   | ----------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
   | Product category              | Electronics, Clothing, Home, etc.              | Any category with <5% of products?                    |
   | Price range                   | Budget (<$25), Mid ($25-$200), Premium (>$200) | Balanced price representation?                        |
   | Average rating                | Low (<3), Medium (3-4), High (>4)              | Are low-rated products underrepresented?              |
   | Review volume (rating_number) | Few (<10), Some (10-100), Many (>100)          | Do low-review products lack reliable quality signals? |
   | Rating variance               | Low (<0.5), Medium (0.5-1.0), High (>1.0)      | Are polarizing products represented?                  |

   **Analysis:**
   - Count products per slice — flag underrepresented groups
   - Check if `rating_variance` is meaningful for low-review products (variance from 2 reviews is unreliable)
   - Verify price distribution covers the range users are likely to query about

   **Why this matters for SavVio:**
   If the product data is dominated by one category (e.g., electronics), the RAG retrieval and quality signals will perform poorly for other categories. If budget products are underrepresented, the system may lack good alternatives to recommend when giving a "Yellow Light."

3. **Evaluate for bias**
   - Are any critical slices underrepresented (<10% of records)?
   - Would the data gaps cause the system to perform worse for specific user groups?
   - Are there product categories where quality signals (rating_variance, avg_rating) are unreliable due to low review counts?

4. **Implement mitigation if needed**

   **Financial data:**
   - Oversample underrepresented income brackets or debt levels
   - Generate synthetic profiles for underrepresented slices
   - Document which slices are underrepresented and the expected impact

   **Product/review data:**
   - Flag products with fewer than N reviews as having low-confidence quality signals
   - Ensure category distribution covers common purchase types
   - Document category gaps and their impact on recommendations

5. **Document analysis**
   - The following subsection summarizes slices used, bias found, and mitigation strategies (aligned with the team’s Bias Detection document). Raw datasets are not modified; mitigation is applied at **model training time only**.

---

### Bias Detection Report (Phase 15 Summary)

This section describes the **slices used**, **bias found**, and **mitigation strategies** from representation bias analysis across Financial (17 columns), Product (10 columns), and Review (8 columns) datasets. The objective is to identify underrepresented high-risk financial groups and high-uncertainty product/review slices that could skew downstream affordability and recommendation models.

#### Slices Used for Bias Detection

**Financial (domain-informed risk bands)**

| Dimension                                                        | Bands / Logic                                                   | Threshold (flagged if) |
| ---------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------- |
| Discretionary income                                             | Negative (&lt;0), Tight (0–1000), Comfortable (&gt;1000)        | —                      |
| Debt-to-income (DTI)                                             | Safe (&lt;0.2), Warning (0.2–0.4), Risky (&gt;0.4)              | Warning &lt;10%        |
| Savings-to-income                                                | Fragile (&lt;0.25), Moderate (0.25–1.0), Strong (&gt;1.0)       | Fragile &lt;10%        |
| Monthly expense burden                                           | Comfortable (&lt;0.5), Tight (0.5–0.8), Overstretched (&gt;0.8) | —                      |
| Emergency fund months                                            | Quantile bands (Q1–Q4) + outliers                               | —                      |
| Income / expenses / loan amount / interest / term / credit score | Low, Medium, High (quantiles or domain bins)                    | High-risk band &lt;10% |
| Employment status                                                | Employed, Self-employed, Unemployed, Student                    | Category &lt;10%       |
| Region                                                           | Geographic categories                                           | —                      |
| user_id                                                          | Uniqueness check                                                | Uniqueness &lt;95%     |
| savings_balance                                                  | Near-zero, Low, Moderate, High                                  | Near-zero &lt;10%      |

**Product (uncertainty and coverage bands)**

| Dimension              | Bands / Logic                                          | Threshold (flagged if) |
| ---------------------- | ------------------------------------------------------ | ---------------------- |
| Price                  | Budget, Mid-range, Premium                             | —                      |
| Average rating         | Low, Medium, High                                      | —                      |
| rating_number          | Low / Medium / High confidence (review count)          | —                      |
| rating_variance        | Consensus, Mixed, Polarized, Single-review proxy (0.0) | —                      |
| Description / features | Length or count bands (0, 1–2, 3–5, 6+)                | —                      |
| details (Brand)        | Rare-brand detection                                   | Rare brands &lt;5%     |
| category               | Long-tail category coverage                            | Category &lt;5%        |
| product_id             | Uniqueness                                             | —                      |

**Review (signal and coverage bands)**

| Dimension                  | Bands / Logic                  | Threshold (flagged if)     |
| -------------------------- | ------------------------------ | -------------------------- |
| rating                     | Negative, Neutral, Positive    | Neutral or minority &lt;5% |
| verified_purchase          | True, False                    | Minority class &lt;5%      |
| helpful_vote               | None, Low, Medium, High        | High &lt;5%                |
| review_title / review_text | Short, Medium, Long, Empty     | —                          |
| user_id                    | Uniqueness                     | Uniqueness &lt;95%         |
| Reviews per product        | 1, 2–5, 6–20, 21+ (cold-start) | —                          |

---

#### Bias Found (Phase 15 Outputs)

**Financial (flagged)**

| Column / slice              | Finding                                                                 | Risk                                                                                    |
| --------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **savings_balance**         | Near-zero savings severely underrepresented (~0%)                       | Financial vulnerability under-captured; model may miss users most in need of Red Light. |
| **employment_status**       | Unemployed (9.93%) and Student (9.91%) slightly below 10%               | Financially vulnerable groups underrepresented.                                         |
| **debt_to_income_ratio**    | Warning band (0.2–0.4) = 3.48%                                          | Mid-risk users underrepresented; model may learn binary Safe vs Risky.                  |
| **savings_to_income_ratio** | Fragile (&lt;0.25) ~1.5%                                                | Long-term vulnerability underrepresented.                                               |
| **emergency_fund_months**   | Critical (&lt;1 month) and Fragile (1–3 months) highly underrepresented | Emergency-risk users rare; classifier may rarely predict Red in real distress cases.    |

**Review (flagged)**

| Column / slice        | Finding         | Risk                                   |
| --------------------- | --------------- | -------------------------------------- |
| **user_id**           | Uniqueness ~83% | Repeat reviewers may dominate signals. |
| **rating**            | Neutral = 4.89% | Middle sentiment underrepresented.     |
| **verified_purchase** | False ~4.16%    | Non-verified reviews underrepresented. |
| **helpful_vote**      | High = 1.5%     | High-signal reviews scarce.            |

**Product (flagged)**

| Column / slice              | Finding                                                                   | Risk                                                    |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| **category**                | Many long-tail categories &lt;5%                                          | Coverage skew toward popular categories; category bias. |
| **details (Brand)**         | Many rare brands &lt;5%                                                   | Model may overfit to dominant brands.                   |
| **Low-confidence products** | High share of single-review / low-count items (e.g. rating_variance == 0) | Cold-start risk; uncertainty at serving time.           |

---

#### Mitigation Strategies (Training-Time Only)

Mitigation is applied **only at model training time**. The raw dataset is not modified.

**Financial**

- **DTI Warning band:** Oversample Warning (0.2–0.4) during training; ensure moderate-debt users are sufficiently represented in the training split.
- **Savings-to-income (Fragile):** Oversample Fragile users; ensure minimum exposure of low-savings profiles; if needed, controlled synthetic low-savings profiles (clearly labeled).
- **Emergency fund (Critical / Fragile):** Oversample Critical and Fragile runway users; stress-test classifier on &lt;3 month runway users; optionally controlled synthetic emergency profiles. **Highest priority mitigation.**
- **Employment (Unemployed / Student):** Stratified sampling by employment_status so vulnerable groups are represented.

**Review**

- **Rating (Neutral):** Stratified sampling by sentiment class (Negative / Neutral / Positive).
- **verified_purchase (False):** Oversample non-verified reviews during training.
- **helpful_vote (High):** Weight high-helpfulness reviews more during training.
- **user_id (repeat reviewers):** User-level deduplication or per-user weighting to avoid repeat reviewers dominating.

**Product**

- **Category (long-tail):** Stratified sampling or category grouping to reduce bias toward popular categories.
- **Brand (rare):** Group rare brands or apply brand smoothing to avoid overfitting to major brands.
- **Low-confidence (single-review):** Flag as low-confidence at serving time; down-weight in recommendation logic; optionally require minimum review count.

**Cross-cutting**

- Stratified sampling across all flagged slices where applicable.
- Controlled oversampling of underrepresented high-risk slices (with caps to avoid distortion).
- Evaluation stress tests on vulnerable slices (low savings, DTI Warning, cold-start products).
- No synthetic data in raw pipeline unless model performance on vulnerable slices remains poor after oversampling.

---

#### Trade-offs and Design Rationale

| Choice                               | Rationale                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **No raw data mutation**             | Preserves real-world distributions and avoids synthetic artifacts in source-of-truth datasets.           |
| **Mitigation at training time only** | Keeps the data pipeline deterministic and auditable; fairness controls are applied where models are fit. |
| **Oversampling with caps**           | Reduces underrepresentation while limiting overfitting to rare slices; validated via cross-validation.   |
| **Long-tail / category grouping**    | Improves fairness across categories but may increase variance; addressed with grouping and smoothing.    |

**Limitations and next steps**

- Incorporate fairness-aware sampling directly into model training pipelines (Phase 3).
- Add per-slice performance metrics (e.g. recall on vulnerable users).
- Consider post-training calibration and threshold tuning for high-risk (Red) decisions.

---

### Tools & Alternatives

| Tool               | Purpose                    | Alternative   | When to Use Alternative             |
| ------------------ | -------------------------- | ------------- | ----------------------------------- |
| Fairlearn          | Bias metrics & analysis    | AIF360        | More comprehensive fairness toolkit |
| Pandas groupby     | Manual slice analysis      | Custom Python | Full control                        |
| SliceFinder        | Automatic slice discovery  | —             | Exploratory analysis                |
| Matplotlib/Seaborn | Distribution visualization | Plotly        | Interactive charts                  |

### Airflow Integration

```python
bias_financial = PythonOperator(
    task_id='bias_analysis_financial',
    python_callable=bias.analyze_financial_bias,
    dag=dag
)

bias_products = PythonOperator(
    task_id='bias_analysis_products',
    python_callable=bias.analyze_product_bias,
    dag=dag
)

# Run in parallel since they analyze independent tracks.
[bias_financial, bias_products] >> complete
```

### Future: Decision Outcome Bias (Phase 3)

Once the Deterministic Financial Logic Engine is built, a separate bias analysis should test:

- Do Green/Yellow/Red recommendations distribute fairly across income brackets?
- Does the affordability score systematically penalize certain financial profiles?
- Are certain product categories more likely to receive Red Light recommendations regardless of user finances?

This analysis requires the full decision pipeline and belongs in the model development phase, not the data pipeline.

---

## Phase 16: Pipeline Orchestration (Airflow DAGs)

### Objective

Structure the entire pipeline using Apache Airflow DAGs with conditional branching to manage complex error-handling logic, parallel processing, and dependencies.

### Implementation Setup

1. **Architecture**
   - We run a containerized Airflow environment via `docker-compose.yaml` (including PostgreSQL and Redis backends).
   - Core DAG file: `dags/data_pipeline_airflow.py`

2. **Implemented DAG Structure**
   The DAG achieves maximum concurrency while respecting data dependencies and executing specific validation branches to catch failures. Here is the implemented structure:

   ```
   [ingest_financial] ───────┐
   [ingest_product]   ───────┼──> [check_ingestion] ──(failed?)──> [email_error_at_ingestion]
   [ingest_review]    ───────┘          │ (success)
                                        ▼
   [validate_raw_financial] ─┐
   [validate_raw_product]   ─┼──> [check_raw_validation] ──(failed?)──> [email_error_error_raw_validation]
   [validate_raw_review]    ─┘          │
                                        ▼
   [detect_anomalies_*]     ────> [check_anomalies] ──(failed?)──> [email_error_at_anomalies]
                                        │
                                        ▼
   [preprocess_financial] ───       (parallel)
   [preprocess_product]   ───> [preprocess_review]
                                        │
                                        ▼
   [check_preprocessing] ──(failed?)──> [email_error_at_preprocessing]
                                        │
                                        ▼
   [validate_processed_*] ──> [check_processed_validation] ──(failed?)──> [email_error_at_processed_validation]
                                        │
                                        ▼
   [feature_financial]    ───┐
   [feature_reviews]      ───┴──> [check_feature_engineering] ──(failed?)──> [email_error_at_feature_engineering]
                                        │
                                        ▼
   [validate_featured]    ──────> [check_featured_validation] ──(failed?)──> [email_error_at_featured_validation]
                                        │
                                        ▼
   [setup_database] ──> [load_financial, load_product] ──> [load_review] ──> [generate_load_embeddings]
                                        │
                                        ▼
   [check_db_loading] ──(failed?)──> [email_error_at_DB_loading]
                                        │(success)
                                        ▼
   [email_pipeline_success] ──> [pipeline_sentinel]
   ```

3. **Task Implementation**
   - The DAG tasks primarily map directly to `src/` modules using `PythonOperator`.
   - Error detection is implemented via `BranchPythonOperator` blocks (e.g., `make_branch_check(...)`) that evaluate upstream task states and trigger `EmailOperator` tasks on failure.

---

## Phase 17: Testing

### Objective

Provide comprehensive coverage of all data modules and tasks before pipeline deployment using parameterized test files and robust service mocks.

### Implementation

1. **Test Structure**
   The `pytest` test suite is fully modularized and located under `tests/`. Tests have been separated into logic boundaries:

   ```
   tests/
   ├── ingestion/
   │   ├── __init__.py
   │   └── test_gcs_loader.py       # Testing GCP blob mocks
   ├── validation/
   │   ├── test_raw_validator.py
   │   └── ...
   ├── bias/
   ├── test_data_pipeline_airflow.py # DAG-level testing and operator mocking
   └── ...
   ```

2. **Testing Standards Applied**
   - **Mocking Extraneous Services**: Stubs are aggressively used (e.g., `_stub` in `test_data_pipeline_airflow.py` and `_stub_google` in `test_gcs_loader.py`) to effectively isolate module tests from requiring active GCP or Airflow metadata connections.
   - **Format Standard:** All test files include module docstrings, section dividers, and structural comments.
   - **Bias Detection Validation:** The bias detection modules (`financial_bias.py`, `product_bias.py`, `review_bias.py`, `run_bias.py`) are exercised end-to-end by running `python3 src/bias/run_bias.py` against the engineered datasets. The logged outputs and the \"Bias Detection Report\" section above form the test oracle for Phase 15.

3. **Running the Tests**
   - `pytest tests/ -v`

---

## Phase 18: Tracking, Logging & Monitoring

### Objective

Ensure data pipeline execution observability.

### Implementation

1. **Module-wide Logging**
   - `src/utils.py` contains shared `logging` config that formats logs by `[timestamp] [level] [module_name]`.
   - Every individual script explicitly initializes its logger instance globally using `logger = logging.getLogger(__name__)`.
2. **Airflow Alerts**
   - If any core task fails, branching conditions inside Airflow dynamically evaluate the task context state and trigger an `EmailOperator` (like `email_error_at_preprocessing`) notifying pipeline maintainers precisely which phase failed.

---

## Phase 19: Pipeline Optimization

### Objective

Diagnose runtime bottlenecks and pipeline faults.

### Implementation

1. **DAG Race Condition Mitigation**
   - _Problem:_ A race condition dropped merged data due to overlapping run times in `preprocess_product` and `preprocess_review`.
   - _Solution:_ Airflow dependencies were explicitly enforced for these two segments `preprocess_product >> preprocess_review` to ensure product mapping succeeds before dependency resolution.

2. **DVC (Data Version Control) Checkpoints**
   - Intermediary DVC markers allow the pipeline data to be cached at raw, processed, and featured states without repeating slow upstream execution times. Data commits are persisted in GCS.

3. **Orchestration Concurrency**
   - Tasks like `ingest_financial`, `ingest_product`, and `ingest_review` execute totally asynchronously without blocking execution threads since they exist independently of one another.
