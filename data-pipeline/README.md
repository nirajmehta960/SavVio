# SavVio Data Pipeline Implementation Guide

## MLOps Course: Data Pipeline Phase

**Project:** SavVio - AI-Driven Financial Advocacy Tool  
**Team Members:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

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
│      └── dvc add data/validated/                                │
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

| Phase | Primary Tools | Alternatives | Airflow Integration |
|-------|---------------|--------------|---------------------|
| Setup | Docker, Git, DVC | — | — |
| Data Collection | Documentation | — | — |
| Ingestion | Pandas, API clients | Polars | PythonOperator |
| Schema/Stats | Great Expectations | Pandera, ydata-profiling, custom Python | PythonOperator |
| Raw Validation | Great Expectations | Pandera, Pydantic, custom validators | PythonOperator |
| Anomaly Detection | Great Expectations, Evidently | Custom Python (IQR/z-score) | PythonOperator + EmailOperator |
| Preprocessing | Pandas | Polars | PythonOperator |
| Processed Validation | Great Expectations | Pandera, custom validators | PythonOperator |
| Features | Pandas, NumPy | Polars | PythonOperator |
| Feature Validation | Great Expectations | Pandera, custom validators | PythonOperator |
| Versioning | DVC, GCP Cloud Storage | — | BashOperator |
| Load to Database | SQLAlchemy, psycopg2 | pandas.to_sql | PythonOperator |
| Embeddings | Sentence-Transformers, OpenAI | LangChain embeddings | PythonOperator |
| Bias Detection | Fairlearn | Custom Pandas slicing, AIF360 | PythonOperator |
| Orchestration | Apache Airflow | — | Native |
| Testing | pytest | unittest | — |
| Monitoring | Python logging, Airflow UI | GCP Cloud Logging, Grafana | Native |
| Optimization | Airflow Gantt | cProfile | Native |

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

1. **Create folder structure** following the required format:
   ```
   SavVio/
   ├── data-pipeline/
   │   ├── dags/              # Airflow DAG definitions
   │   ├── data/
   │   │   ├── raw/           # Raw financial, product, & review data
   │   │   │   ├── financial.csv
   │   │   │   ├── products.jsonl
   │   │   │   ├── reviews.jsonl
   │   │   │   └── .gitignore
   │   │   ├── processed/     # Cleaned & transformed data
   │   │   │   ├── financial_processed.csv
   │   │   │   ├── products_processed.csv
   │   │   │   ├── reviews_processed.csv
   │   │   │   └── .gitignore
   │   │   └── validated/     # Feature-engineered data
   │   │       ├── features.csv
   │   │       └── .gitignore
   │   ├── scripts/
   │   │   ├── ingest/        # Data ingestion modules
   │   │   │   ├── __init__.py
   │   │   │   ├── api_loader.py
   │   │   │   ├── gcs_loader.py
   │   │   │   ├── config.py
   │   │   │   ├── financial.py
   │   │   │   ├── product.py
   │   │   │   ├── review.py
   │   │   │   ├── utils.py
   │   │   │   └── run_ingestion.py
   │   │   ├── validate/      # Validation modules
   │   │   │   ├── __init__.py
   │   │   │   ├── raw_validator.py
   │   │   │   ├── processed_validator.py
   │   │   │   ├── feature_validator.py
   │   │   │   ├── schemas/
   │   │   │   │   ├── financial_schema.json
   │   │   │   │   ├── product_schema.json
   │   │   │   │   └── review_schema.json
   │   │   │   └── run_validation.py
   │   │   ├── preprocess/    # Preprocessing modules
   │   │   │   ├── __init__.py
   │   │   │   ├── financial.py
   │   │   │   ├── product.py
   │   │   │   ├── review.py
   │   │   │   ├── utils.py
   │   │   │   └── run_preprocessing.py
   │   │   ├── anomaly/       # Anomaly detection modules
   │   │   │   ├── __init__.py
   │   │   │   ├── detectors.py
   │   │   │   ├── config.yaml
   │   │   │   └── run_anomaly_detection.py
   │   │   ├── features/      # Feature engineering modules
   │   │   │   ├── __init__.py
   │   │   │   ├── financial_features.py
   │   │   │   ├── affordability_features.py
   │   │   │   ├── review_features.py
   │   │   │   ├── utils.py
   │   │   │   └── run_features.py
   │   │   ├── bias/          # Bias detection modules
   │   │   │   ├── __init__.py
   │   │   │   ├── slicing.py
   │   │   │   ├── analysis.py
   │   │   │   └── run_bias_analysis.py
   │   │   └── database/      # Database loading modules
   │   │       ├── __init__.py
   │   │       ├── postgres_loader.py
   │   │       ├── vector_loader.py
   │   │       ├── models.py
   │   │       ├── utils.py
   │   │       └── run_db_load.py
   │   ├── tests/             # Unit tests
   │   │   ├── test_ingestion.py
   │   │   ├── test_validation.py
   │   │   ├── test_preprocessing.py
   │   │   ├── test_features.py
   │   │   ├── test_anomaly.py
   │   │   ├── test_bias.py
   │   │   ├── test_database.py
   │   │   └── conftest.py
   │   ├── logs/              # Pipeline execution logs
   │   ├── config/            # Configuration files
   │   ├── docs/              # Documentation
   │   ├── dvc.yaml           # DVC pipeline definition
   │   └── README.md          # Pipeline documentation
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

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Docker + Docker Compose | Containerized Airflow | Podman |
| Git | Code version control | — |
| DVC | Data version control | LakeFS |
| Python 3.12+ | Runtime | — |
| PostgreSQL | Local database | SQLite (lighter for dev) |

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
   
   | Attribute | Description |
   |-----------|-------------|
   | Data Type | Medium-sized datasets (thousands of records) |
   | Financial Data Fields | Monthly income, rent, recurring bills, savings, debt |
   | Product Data Fields | Product name, category, price, specifications, description |
   | Review Data Fields | Product ID, rating, text, helpfulness metrics |
   | Estimated Size | Small to medium (hundreds to thousands of records each) |
   | Update Frequency | Periodic or event-driven (simulated) |
   | Sensitive Fields | Income, savings, expenses (masked and encrypted) |

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

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Pandas | Data loading | Polars | If performance issues with large files |
| API clients | Data fetching | Direct HTTP requests | Custom control needed |

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
Version control the raw ingested data to ensure reproducibility and enable rollback if needed.

### Steps

1. **Add raw data to DVC tracking**
   ```bash
   dvc add data/raw/financial.csv
   dvc add data/raw/products.jsonl
   dvc add data/raw/reviews.jsonl
   ```

2. **Commit .dvc files to Git**
   ```bash
   git add data/raw/*.dvc data/raw/.gitignore
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

| Tool | Purpose | Alternative |
|------|---------|-------------|
| DVC | Data versioning | LakeFS, Git LFS |
| GCP Cloud Storage | Remote storage | AWS S3, Azure Blob |

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

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Great Expectations | Schema & expectations | Pandera | Lighter weight, Pythonic |
| ydata-profiling | Auto statistics | Pandas describe() | Quick exploration |
| Custom Python | Full control | — | Simple schemas |

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

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Great Expectations | Validation checkpoints | Pandera | Simpler DataFrame validation |
| — | — | Pydantic | Row-level/JSON validation |
| — | — | Custom validators | Maximum flexibility |

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
   
   | Severity | Condition | Action |
   |----------|-----------|--------|
   | INFO | Minor outliers | Log only |
   | WARNING | >5% issues | Email alert |
   | CRITICAL | Data corrupted | Email + halt |

4. **Quarantine suspicious records**

### Tools & Alternatives

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Great Expectations | Rule-based detection | Custom Python | More control |
| Evidently AI | Statistical detection | PyOD | More algorithms |
| Airflow EmailOperator | Email alerts | SendGrid, SMTP | External email service |
| Airflow SlackWebhookOperator | Slack alerts | Custom webhook | Other chat platforms |

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

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Pandas | Data manipulation | Polars | 10-100x faster for large data |
| NumPy | Numerical operations | — | — |

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

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Great Expectations | Validation | Pandera |
| Custom Python | Transformation checks | — |

---

## Phase 10: Version Processed Data (DVC Checkpoint #2)

### Objective
Version control the processed data to track transformations.

### Steps

1. **Add processed data to DVC**
   ```bash
   dvc add data/processed/financial_processed.csv
   dvc add data/processed/products_processed.csv
   dvc add data/processed/reviews_processed.csv
   ```

2. **Commit and push**
   ```bash
   git add data/processed/*.dvc
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
Create meaningful features that maximize predictive signal for SavVio's decision engine.

### Steps

1. **Create modular feature engineering package**
   ```
   scripts/features/
   ├── __init__.py
   ├── financial_features.py
   ├── affordability_features.py
   ├── review_features.py
   ├── utils.py
   └── run_features.py
   ```

2. **Financial health features**

   | Feature | Formula | Purpose |
   |---------|---------|---------|
   | `discretionary_income` | income - (expenses + emi) | Available money |
   | `debt_to_income_ratio` | emi / income | Burden indicator |
   | `savings_rate` | savings / income | Health indicator |
   | `monthly_expense_burden_ratio` | (expenses + emi) / income | Spending pattern |
   | `financial_runway` | savings / (expenses + emi) | Safety buffer |

3. **Affordability features**

   | Feature | Formula | Purpose |
   |---------|---------|---------|
   | `price_to_income_ratio` | price / income | Relative cost |
   | `affordability_score` | discretionary - price | Can afford? |
   | `residual_utility_score` | (savings - price) / (expenses + emi) | Emergency fund impact |

4. **Review-based features**

   | Feature | Formula | Purpose |
   |---------|---------|---------|
   | `review_length` | char_count(review_text) | Detail-depth signal |
   | `rating_variance` | std(rating) per product | Rating consistency |

5. **Handle edge cases**
   - Zero income: flag, don't compute ratios
   - Division by zero: safe division with defaults
   - Missing review data: fill with defaults

   - `data/features/features.csv`
      - **Description:** This is the master consolidated dataset containing all engineered features (affordability metrics, review signals, and financial ratios) linked to user-product interactions.
      - **Purpose:** This file is the primary input for training the Purchase Guardrail model and will be deployed to the PostgreSQL database for serving.
   - `config/feature_definitions.json`

### Tools & Alternatives

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Pandas | Feature creation | Polars | Large datasets |
| NumPy | Numerical ops | — | — |
| Scikit-learn | Encoding | Category Encoders | More encoding options |

---

## Phase 12: Feature Validation

### Objective
Validate that engineered features are correctly calculated and within expected ranges.

### Steps

1. **Define feature expectations**

   | Feature | Expected Range | Validation |
   |---------|---------------|------------|
   | `discretionary_income` | Can be negative | Check not NaN |
   | `debt_to_income_ratio` | 0 to ~2 | Flag if > 2 |
   | `savings_rate` | 0 to ~1 | Flag if > 1 |
   | `affordability_score` | Any | Check not NaN/Inf |
   | `residual_utility_score` | Any | Check not NaN/Inf |
   | `avg_product_rating` | 1-5 | Check in range |
   | `num_reviews` | >= 0 | Check not negative |
   | `rating_variance` | >= 0 | Check not negative |

2. **Run validation**
   - No NaN or Inf values
   - Ratios within reasonable bounds
   - All expected features present
   - Review aggregations correct

3. **Cross-validate calculations**
   - Sample records: manually verify formula
   - Edge cases: zero income, negative values

4. **Handle failures**
   - Log calculation errors
   - Identify problematic source records

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Great Expectations | Feature validation | Pandera |
| Custom Python | Formula verification | — |

---

## Phase 13: Version Features (DVC Checkpoint #3)

### Objective
Version control the final feature set for model reproducibility.

### Steps

1. **Add features to DVC**
   ```bash
   dvc add data/features/features.csv
   dvc add data/features/financial_featured.csv
   dvc add data/features/reviews_featured.jsonl
   ```

2. **Commit and push**
   ```bash
   git add data/features/*.dvc
   git commit -m "Add features v1.0"
   dvc push
   ```

3. **Tag the version**
   ```bash
   git tag -a "data-features-v1.0" -m "Feature-engineered data"
   ```

4. **Update dvc.yaml**
   - Define complete pipeline stages
   - Enable `dvc repro` for reproduction

### Why Version Here?
- Final model-ready data
- Ties to model versions
- Experiment reproducibility

---

## Phase 14: Load to Database

### Objective
Load processed data and feature embeddings into PostgreSQL (relational) and pgvector (vector) databases for the SavVio application.

### Environment Configuration

| Environment | Database | Connection |
|-------------|----------|------------|
| Development | Local PostgreSQL + pgvector | localhost:5432 |
| Production | GCP Cloud SQL + pgvector extension | Cloud SQL proxy |

### Steps

1. **Identify datasets to load**
   We will load the following processed and featured datasets into the database:
   - `data/features/financial_featured.csv`
   - `data/features/reviews_featured.jsonl`
   - `data/processed/product_preprocessed.jsonl`
   - `data/features/features.csv`

2. **Create database loading package**
   ```
   scripts/database/
   ├── __init__.py
   ├── postgres_loader.py    # Relational data
   ├── vector_loader.py      # Embeddings
   ├── models.py             # SQLAlchemy models
   └── utils.py              # Connection helpers
   ```

2. **Define database schema (tables)**

   **PostgreSQL Tables:**
   ```sql
   -- Financial data table
   CREATE TABLE financial_profiles (
       id SERIAL PRIMARY KEY,
       monthly_income DECIMAL(12,2),
       rent DECIMAL(12,2),
       recurring_bills DECIMAL(12,2),
       savings_balance DECIMAL(12,2),
       debt_obligations DECIMAL(12,2),
       discretionary_income DECIMAL(12,2),
       debt_to_income_ratio DECIMAL(5,4),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Product data table
   CREATE TABLE products (
       id SERIAL PRIMARY KEY,
       product_name VARCHAR(500),
       category VARCHAR(100),
       price DECIMAL(12,2),
       specifications TEXT,
       description TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Review data table
   CREATE TABLE reviews (
       id SERIAL PRIMARY KEY,
       product_id INTEGER REFERENCES products(id),
       reviewer_id VARCHAR(255),
       rating DECIMAL(2,1),
       text TEXT,
       helpful_count INTEGER DEFAULT 0,
       date TIMESTAMP,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

   **pgvector Table (for embeddings):**
   ```sql
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;

   -- Product embeddings for RAG
   CREATE TABLE product_embeddings (
       id SERIAL PRIMARY KEY,
       product_id INTEGER REFERENCES products(id),
       embedding vector(384),  -- Dimension depends on model
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   -- Create index for similarity search
   CREATE INDEX ON product_embeddings 
   USING ivfflat (embedding vector_cosine_ops);
   ```

3. **Generate product embeddings**
   - Use product description + specifications
   - Generate embeddings using Sentence-Transformers or OpenAI
   - Store in pgvector for RAG retrieval

4. **Implement data loaders**

   **postgres_loader.py:**
   - `load_financial_data(df, engine)` — Load financial profiles
   - `load_product_data(df, engine)` — Load products
   - `load_review_data(df, engine)` — Load reviews
   - `get_engine(env)` — Get connection based on environment

   **vector_loader.py:**
   - `generate_embeddings(texts, model)` — Create embeddings
   - `load_embeddings(product_ids, embeddings, engine)` — Store in pgvector

5. **Environment-based configuration**
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

6. **Load data**
   - Truncate existing data (or upsert strategy)
   - Load financial profiles
   - Load products
   - Load reviews
   - Generate and load embeddings

### Tools & Alternatives

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| SQLAlchemy | ORM & connection | psycopg2 direct | Simpler queries |
| psycopg2 | PostgreSQL driver | asyncpg | Async operations |
| Sentence-Transformers | Embeddings | OpenAI API | Higher quality, cost |
| pgvector | Vector storage | Pinecone, Vertex AI Vector Search | Managed service |

### Airflow Integration
```python
load_to_postgres = PythonOperator(
    task_id='load_to_postgres',
    python_callable=postgres_loader.load_all,
    op_kwargs={'env': '{{ var.value.environment }}'},
    dag=dag
)

generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=vector_loader.generate_and_load,
    dag=dag
)

load_to_postgres >> generate_embeddings
```

---

## Phase 15: Bias Detection & Mitigation

### Objective
Detect and mitigate bias in the FEATURE data through slicing and analysis across subgroups. This happens on features because features are what the model uses for decisions.

### Why After Feature Engineering?

| Check Point | What You'd Find | Why Features is Better |
|-------------|-----------------|----------------------|
| Raw Data | Demographic imbalance | Good to know, but... |
| Features | **Bias in decision inputs** | This is what affects recommendations |

Example: Raw data might have equal income representation, but `affordability_score` feature might systematically disadvantage certain groups.

### Steps

1. **Identify slicing dimensions**

   | Slice Dimension | Groups | Why It Matters |
   |-----------------|--------|----------------|
   | Income bracket | Low (<$3k), Medium ($3k-$7k), High (>$7k) | Fair across income levels |
   | Debt-to-income | Low (<0.2), Medium (0.2-0.4), High (>0.4) | Don't penalize debt unfairly |
   | Expense burden | Low (<0.5), Medium (0.5-0.8), High (>0.8) | Fair regardless of spending |
   | Product category | Electronics, Clothing, Home, etc. | No category bias |
   | Rating distribution | Low-rated, Medium-rated, High-rated products | Reviews don't skew bias |

2. **Perform slice analysis**
   - Count records per slice
   - Compare feature distributions across slices
   - Identify underrepresented groups

3. **Evaluate for bias**
   - Would low-income users get unfair "Red Light" recommendations?
   - Are certain categories systematically scored lower?
   - Do review-based features create bias?

4. **Implement mitigation if needed**
   - Re-sample underrepresented groups
   - Adjust feature calculations
   - Document trade-offs

5. **Document analysis**
   - `docs/bias_analysis_report.md`

### Tools & Alternatives

| Tool | Purpose | Alternative | When to Use Alternative |
|------|---------|-------------|------------------------|
| Fairlearn | Bias metrics | AIF360 | More comprehensive |
| Pandas groupby | Manual slicing | Custom Python | Full control |
| SliceFinder | Auto slice discovery | — | Exploratory |

---

## Phase 16: Pipeline Orchestration (Airflow DAGs)

### Objective
Structure the entire pipeline using Airflow DAGs with logical task connections.

### Steps

1. **Design DAG structure**

   ```
   [ingest_financial] ─┬─→ [version_raw] → [gen_schema] → [validate_raw]
   [ingest_products]  ─┤                         ↓
   [ingest_reviews]   ─┘                  [detect_anomalies]
                                                 ↓
                                          [preprocess]
                                                 ↓
                                        [validate_processed]
                                                 ↓
                                        [version_processed]
                                                 ↓
                                        [engineer_features]
                                                 ↓
                                        [validate_features]
                                                 ↓
                                         [version_features]
                                                 ↓
                                          [load_postgres]
                                                 ↓
                                         [generate_embeddings]
                                                 ↓
                                           [detect_bias]
                                                 ↓
                                             [complete]
   ```

2. **Create DAG definition**
   - Location: `dags/savvio_data_pipeline.py`
   - Schedule: `@daily` or `None` for manual

3. **Define tasks**
   - `PythonOperator` for processing
   - `BashOperator` for DVC
   - `EmailOperator` for alerts

4. **Set dependencies and error handling**

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Apache Airflow | Orchestration | Prefect, Dagster |
| PythonOperator | Python tasks | @task decorator |
| BashOperator | Shell commands | — |

---

## Phase 17: Testing

### Objective
Write comprehensive unit tests for each pipeline component.

### Steps

1. **Create test structure**
   ```
   tests/
   ├── test_ingestion.py
   ├── test_validation.py
   ├── test_preprocessing.py
   ├── test_features.py
   ├── test_anomaly.py
   ├── test_bias.py
   ├── test_database.py
   └── conftest.py
   ```

2. **Test categories**

   | Component | Test Cases |
   |-----------|------------|
   | Ingestion | File not found, empty file, success |
   | Validation | Invalid types, nulls, range violations |
   | Preprocessing | Missing values, transformations, review text |
   | Features | Edge cases, calculations, review aggregations |
   | Anomaly | Outlier detection, thresholds |
   | Bias | Slice distributions, fairness metrics |
   | Database | Connection, load, embeddings |

3. **Run in CI/CD**
   - pytest in GitHub Actions
   - Coverage reporting

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| pytest | Testing | unittest |
| pytest-cov | Coverage | coverage.py |
| unittest.mock | Mocking | pytest-mock |

---

## Phase 18: Tracking, Logging & Monitoring

### Objective
Implement logging and monitoring for pipeline observability.

### Steps

1. **Python logging**
   - Configure in each module
   - Levels: DEBUG, INFO, WARNING, ERROR
   - Format: `[timestamp] [level] [module] message`

2. **Track metrics**
   - Records ingested (financial, products, reviews)
   - Validation pass rate
   - Processing time
   - Anomalies detected

3. **Monitoring**
   - Airflow UI for tasks
   - GCP Cloud Logging for production

4. **Alerts**
   - Task failures → Email
   - Critical issues → Slack

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Python logging | App logs | loguru |
| Airflow UI | Task monitoring | — |
| GCP Cloud Logging | Centralized logs | ELK Stack |
| Grafana | Dashboards | GCP Monitoring |

---

## Phase 19: Pipeline Optimization

### Objective
Identify and resolve bottlenecks to optimize performance.

### Steps

1. **Analyze with Airflow Gantt chart**
   - Identify slow tasks
   - Find parallelization opportunities

2. **Common bottlenecks**

   | Bottleneck | Solution |
   |------------|----------|
   | Data download | Cache, retry logic |
   | JSONL parsing & flattening | Use Polars, batch processing |
   | Large file processing | Chunking, Polars |
   | Review aggregation | Batch processing |
   | Embedding generation | Batch processing |
   | DVC push | Async, compression |

3. **Implement optimizations**
   - Parallelize independent tasks (financial, products, reviews)
   - Switch to Polars if Pandas slow
   - Batch embedding generation
   - Cache intermediate results

4. **Measure improvements**
   - Before/after comparisons
   - Document gains

### Tools & Alternatives

| Tool | Purpose | Alternative |
|------|---------|-------------|
| Airflow Gantt | Visual analysis | — |
| Polars | Fast processing | Dask (distributed) |
| cProfile | Code profiling | line_profiler |