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

# -------------------- TASKS ------------------------

# Error Email Operators

email_error_at_ingestion = EmailOperator(
    task_id="send_email_at_ingestion_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Ingestion",
    html_content="<p>Something went wrong at Ingestion stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

email_error_at_preprocessing = EmailOperator(
    task_id="send_email_at_preprocessing_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Preprocessing",
    html_content="<p>Something went wrong at Preprocessing stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

email_error_at_feature_engineering = EmailOperator(
    task_id="send_email_at_feature_engineering_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Feature Engineering",
    html_content="<p>Something went wrong at Feature Engineering stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

email_error_at_DB_loading = EmailOperator(
    task_id="send_email_at_DB_loading_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at DB Loading",
    html_content="<p>Something went wrong at DB Loading stage.</p>",
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

email_error_at_bias_analysis = EmailOperator(
    task_id="send_email_at_bias_analysis_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Bias Analysis",
    html_content="<p>Something went wrong at Bias Analysis stage.</p>",
    dag=dag,
)

# Error Slack Operators (channel: #group-34)

slack_error_at_ingestion = SlackWebhookOperator(
    task_id="send_slack_at_ingestion_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Ingestion* stage.",
    channel="#group-34",
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

slack_error_at_feature_engineering = SlackWebhookOperator(
    task_id="send_slack_at_feature_engineering_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Feature Engineering* stage.",
    channel="#group-34",
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

slack_error_at_bias_analysis = SlackWebhookOperator(
    task_id="send_slack_at_bias_analysis_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Bias Analysis* stage.",
    channel="#group-34",
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
# Raw Data Validation  (after ingestion)
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

# Ingestion → Raw validation (both run after ingestion completes)
[ingest_financial, ingest_products, ingest_reviews] >> validate_raw_data
[ingest_financial, ingest_products, ingest_reviews] >> validate_raw_anomaly  # INFO-only, non-gating

# Ingestion error alerts (fire only if any ingestion task fails)
[ingest_financial, ingest_products, ingest_reviews] >> email_error_at_ingestion
[ingest_financial, ingest_products, ingest_reviews] >> slack_error_at_ingestion

#----------------------------------------------------
# Data Preprocessing  (runs in PARALLEL, after raw validation passes)
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

# Raw validation gates preprocessing (anomaly check does NOT gate)
validate_raw_data >> [preprocess_financial, preprocess_products, preprocess_reviews]

#----------------------------------------------------
# Processed Data Validation  (after preprocessing)
#----------------------------------------------------

validate_processed_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed,
    dag=dag,
)

# Preprocessing → Processed validation
[preprocess_financial, preprocess_products, preprocess_reviews] >> validate_processed_data

# Preprocessing error alerts
[preprocess_financial, preprocess_products, preprocess_reviews] >> email_error_at_preprocessing
[preprocess_financial, preprocess_products, preprocess_reviews] >> slack_error_at_preprocessing

#----------------------------------------------------
# Feature Engineering  (runs in PARALLEL, after processed validation passes)
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



#----------------------------------------------------
# Featured Data Validation  (after feature engineering)
#----------------------------------------------------

validate_featured_data = PythonOperator(
    task_id='validate_featured_data',
    python_callable=validate_features,
    dag=dag,
)

# Feature engineering → Featured validation
[feature_financial, feature_reviews] >> validate_featured_data

#----------------------------------------------------
# Loading Featured Data into PostgreSQL  (after featured validation passes)
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

# Feature Engineering error alerts
[feature_financial, feature_reviews] >> email_error_at_feature_engineering
[feature_financial, feature_reviews] >> slack_error_at_feature_engineering

# DB Loading error alerts
[load_financial, load_product, load_review, generate_load_embeddings] >> email_error_at_DB_loading
[load_financial, load_product, load_review, generate_load_embeddings] >> slack_error_at_DB_loading

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