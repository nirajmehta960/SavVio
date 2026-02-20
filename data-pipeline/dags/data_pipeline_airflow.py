from __future__ import annotations

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task
from src.features.run_features import feature_financial_task, feature_review_task
from src.database.run_database import load_financial_task, load_products_task, load_reviews_task, generate_and_load_embedding_task
from src.validation.run_validation import validate_raw, validate_processed, validate_features, validate_raw_anomalies

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
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

# ==================== ERROR ALERT OPERATORS ====================

# --- Ingestion Errors ---
email_error_at_ingestion = EmailOperator(
    task_id="send_email_at_ingestion_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Ingestion",
    html_content="<p>Something went wrong at Ingestion stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_ingestion = SlackWebhookOperator(
    task_id="send_slack_at_ingestion_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Ingestion* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Raw Validation Errors ---
email_error_at_raw_validation = EmailOperator(
    task_id="send_email_at_raw_validation_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Raw Validation",
    html_content="<p>Something went wrong at Raw Data Validation stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_raw_validation = SlackWebhookOperator(
    task_id="send_slack_at_raw_validation_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Raw Validation* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Preprocessing Errors ---
email_error_at_preprocessing = EmailOperator(
    task_id="send_email_at_preprocessing_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Preprocessing",
    html_content="<p>Something went wrong at Preprocessing stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_preprocessing = SlackWebhookOperator(
    task_id="send_slack_at_preprocessing_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Preprocessing* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Processed Validation Errors ---
email_error_at_processed_validation = EmailOperator(
    task_id="send_email_at_processed_validation_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Processed Validation",
    html_content="<p>Something went wrong at Processed Data Validation stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_processed_validation = SlackWebhookOperator(
    task_id="send_slack_at_processed_validation_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Processed Validation* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Feature Engineering Errors ---
email_error_at_feature_engineering = EmailOperator(
    task_id="send_email_at_feature_engineering_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Feature Engineering",
    html_content="<p>Something went wrong at Feature Engineering stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_feature_engineering = SlackWebhookOperator(
    task_id="send_slack_at_feature_engineering_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Feature Engineering* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Featured Validation Errors ---
email_error_at_featured_validation = EmailOperator(
    task_id="send_email_at_featured_validation_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Featured Validation",
    html_content="<p>Something went wrong at Featured Data Validation stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_featured_validation = SlackWebhookOperator(
    task_id="send_slack_at_featured_validation_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Featured Validation* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- DB Loading Errors ---
email_error_at_DB_loading = EmailOperator(
    task_id="send_email_at_DB_loading_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at DB Loading",
    html_content="<p>Something went wrong at DB Loading stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_DB_loading = SlackWebhookOperator(
    task_id="send_slack_at_DB_loading_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *DB Loading* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# --- Bias Analysis Errors (placeholder) ---
email_error_at_bias_analysis = EmailOperator(
    task_id="send_email_at_bias_analysis_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Bias Analysis",
    html_content="<p>Something went wrong at Bias Analysis stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

slack_error_at_bias_analysis = SlackWebhookOperator(
    task_id="send_slack_at_bias_analysis_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Bias Analysis* stage.",
    channel="#group-34",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# ==================== SUCCESS ALERT OPERATORS ====================

email_pipeline_success = EmailOperator(
    task_id="send_email_pipeline_success",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Pipeline Completed Successfully",
    html_content="<p>All data has been successfully ingested, validated, processed, featured, and loaded into the database.</p>",
    dag=dag,
)

slack_pipeline_success = SlackWebhookOperator(
    task_id="send_slack_pipeline_success",
    slack_webhook_conn_id="slack_webhook",
    message=":large_green_circle: *SavVio Data Pipeline* — Pipeline completed *successfully*. All data loaded into DB.",
    channel="#group-34",
    dag=dag,
)

#----------------------------------------------------
# 1. Data Ingestion  (runs in PARALLEL)
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

# Ingestion error alerts (fire if ANY ingestion task fails)
[ingest_financial, ingest_products, ingest_reviews] >> email_error_at_ingestion
[ingest_financial, ingest_products, ingest_reviews] >> slack_error_at_ingestion

#----------------------------------------------------
# 2. Raw Data Validation  (after ALL ingestion succeeds)
#----------------------------------------------------

validate_raw_data = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw,
    dag=dag,
)

validate_raw_anomaly = PythonOperator(
    task_id='validate_raw_anomalies',
    python_callable=validate_raw_anomalies,
    dag=dag,
)

# Ingestion → validation (only if ALL ingestion tasks pass)
[ingest_financial, ingest_products, ingest_reviews] >> validate_raw_data
[ingest_financial, ingest_products, ingest_reviews] >> validate_raw_anomaly  # INFO-only, non-gating

# Raw validation error alerts
validate_raw_data >> email_error_at_raw_validation
validate_raw_data >> slack_error_at_raw_validation

#----------------------------------------------------
# 3. Data Preprocessing  (after raw validation passes)
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

# Raw validation gates preprocessing
validate_raw_data >> [preprocess_financial, preprocess_products, preprocess_reviews]

# Preprocessing error alerts
[preprocess_financial, preprocess_products, preprocess_reviews] >> email_error_at_preprocessing
[preprocess_financial, preprocess_products, preprocess_reviews] >> slack_error_at_preprocessing

#----------------------------------------------------
# 4. Processed Data Validation  (after ALL preprocessing succeeds)
#----------------------------------------------------

validate_processed_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed,
    dag=dag,
)

[preprocess_financial, preprocess_products, preprocess_reviews] >> validate_processed_data

# Processed validation error alerts
validate_processed_data >> email_error_at_processed_validation
validate_processed_data >> slack_error_at_processed_validation

#----------------------------------------------------
# 5. Feature Engineering  (after processed validation passes)
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

# Processed validation gates feature engineering
validate_processed_data >> [feature_financial, feature_reviews]

# Feature engineering error alerts
[feature_financial, feature_reviews] >> email_error_at_feature_engineering
[feature_financial, feature_reviews] >> slack_error_at_feature_engineering

#----------------------------------------------------
# 6. Featured Data Validation  (after ALL feature engineering succeeds)
#----------------------------------------------------

validate_featured_data = PythonOperator(
    task_id='validate_featured_data',
    python_callable=validate_features,
    dag=dag,
)

[feature_financial, feature_reviews] >> validate_featured_data

# Featured validation error alerts
validate_featured_data >> email_error_at_featured_validation
validate_featured_data >> slack_error_at_featured_validation

#----------------------------------------------------
# 7. Loading Data into PostgreSQL  (after featured validation passes)
#----------------------------------------------------

load_financial = PythonOperator(
    task_id='load_financial_profiles',
    python_callable=load_financial_task,
    dag=dag,
)

load_product = PythonOperator(
    task_id='load_products',
    python_callable=load_products_task,
    dag=dag,
)

load_review = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews_task,
    dag=dag,
)

generate_load_embeddings = PythonOperator(
    task_id='generate_load_embeddings',
    python_callable=generate_and_load_embedding_task,
    dag=dag,
)

# Featured validation gates DB loading
validate_featured_data >> [load_financial, load_product, load_review]

[load_financial, load_product, load_review] >> generate_load_embeddings

# DB loading error alerts
[load_financial, load_product, load_review, generate_load_embeddings] >> email_error_at_DB_loading
[load_financial, load_product, load_review, generate_load_embeddings] >> slack_error_at_DB_loading

#----------------------------------------------------
# 8. Pipeline Success  (fires after ALL DB tasks complete)
#----------------------------------------------------

generate_load_embeddings >> email_pipeline_success
generate_load_embeddings >> slack_pipeline_success

#----------------------------------------------------
# Airflow DAG for Bias Analysis
#----------------------------------------------------

# bias_financial = PythonOperator(
#     task_id='bias_analysis_financial',
#     python_callable=bias.analyze_financial_bias,
#     dag=dag
# )

# bias_products = PythonOperator(
#     task_id='bias_analysis_products',
#     python_callable=bias.analyze_product_bias,
#     dag=dag
# )

# # Run in parallel since they analyze independent tracks.
# [bias_financial, bias_products] >> complete