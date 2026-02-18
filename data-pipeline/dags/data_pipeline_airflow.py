from __future__ import annotations

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task
from src.features.run_features import feature_financial_task, feature_review_task
from src.validation.run_validation import validate_raw, validate_processed, validate_features, validate_raw_anomalies, validate_anomalies


from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule



# ---------- Default args ----------
default_args = {
    'owner': 'Murtaza Nipplewala',
    'start_date': datetime(2026, 2, 17),
    'retries': 2,
    'retry_delay': timedelta(minutes=0.3),
    'email': 'murtaza.sn786@gmail.com',
    'email_on_failure': True,
    'email_on_retry': False
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Data_pipeline_airflow",
    default_args=default_args,
    description="Data Pipeline Airflow DAG",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Murtaza Nipplewala": "https://github.com/MurtazaNipplewala/SavVio"},
    max_active_runs=1,
)

# -------------------- TASKS ------------------------

send_email = EmailOperator(
    task_id="send_email",
    to="murtaza.sn786@gmail.com",
    subject="Notification from Data Pipeline Airflow",
    html_content="<p>This is a notification email sent from Data Pipeline Airflow.</p>",
    dag=dag,
)

#----------------------------------------------------
# Data Ingestion  (runs in PARALLEL)
#----------------------------------------------------

ingest_financial = PythonOperator(
    task_id='ingest_financial_data',
    python_callable=ingest_financial_task,
    dag=dag,
)

ingest_products = PythonOperator(
    task_id='ingest_product_data',
    python_callable=ingest_product_task,
    dag=dag,
)

ingest_reviews = PythonOperator(
    task_id='ingest_review_data',
    python_callable=ingest_review_task,
    dag=dag,
)

#----------------------------------------------------
# Data Preprocessing  (runs in PARALLEL)
#----------------------------------------------------

preprocess_financial = PythonOperator(
    task_id='preprocess_financial_data',
    python_callable=preprocess_financial_task,
    dag=dag,
)

preprocess_products = PythonOperator(
    task_id='preprocess_product_data',
    python_callable=preprocess_product_task,
    dag=dag,
)

preprocess_reviews = PythonOperator(
    task_id='preprocess_review_data',
    python_callable=preprocess_review_task,
    dag=dag,
)

#----------------------------------------------------
# Raw Data Validation - after ingestion
#----------------------------------------------------

validate_raw_data = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw,
    dag=dag,
)

#----------------------------------------------------
# Raw Anomaly Detection (Tier 1 — light, INFO-only, never halts)
#----------------------------------------------------

detect_raw_anomalies = PythonOperator(
    task_id='detect_raw_anomalies',
    python_callable=validate_raw_anomalies,
    dag=dag,
)

#----------------------------------------------------
# Processed Data Validation - after preprocessing   
#----------------------------------------------------

validate_processed_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed,
    dag=dag,
)

#----------------------------------------------------
# Feature Engineering (runs in PARALLEL, after processed validation)
#----------------------------------------------------

feature_financial = PythonOperator(
    task_id='feature_financial_data',
    python_callable=feature_financial_task,
    dag=dag,
)

feature_reviews = PythonOperator(
    task_id='feature_review_data',
    python_callable=feature_review_task,
    dag=dag,
)

#----------------------------------------------------
# Anomaly Detection (Tier 2 — full, WARNING/CRITICAL, gates DB load)
#----------------------------------------------------

detect_anomalies = PythonOperator(
    task_id='detect_anomalies',
    python_callable=validate_anomalies,
    dag=dag,
)

#----------------------------------------------------
# Feature Validation - after anomaly detection (Tier 2)
#----------------------------------------------------

validate_features_data = PythonOperator(
    task_id='validate_features_data',
    python_callable=validate_features,
    dag=dag,
)

# Ingestion → Raw Val → Raw Anomaly (Tier 1, light) → Preprocess → Processed Val → Feature Eng → Anomaly (Tier 2, full) → Feature Val → Loading
[ingest_financial, ingest_products, ingest_reviews] >> validate_raw_data >> detect_raw_anomalies >> [preprocess_financial, preprocess_products, preprocess_reviews] >> validate_processed_data >> [feature_financial, feature_reviews] >> detect_anomalies >> validate_features_data

#----------------------------------------------------
# Load Featured Data into PostgreSQL
#----------------------------------------------------

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

# Feature Validation → Loading → Embeddings
validate_features_data >> [load_financial, load_products, load_reviews] >> generate_embeddings

#----------------------------------------------------
# Bias Analysis (runs in PARALLEL)
#----------------------------------------------------

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

[bias_financial, bias_products] >> complete