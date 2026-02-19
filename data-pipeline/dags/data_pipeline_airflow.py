from __future__ import annotations

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task
from src.features.run_features import feature_financial_task, feature_review_task
from src.database.upload_to_postgres import load_financial_profiles, load_products, load_reviews
from src.database.upload_to_vector_db import generate_and_load
from src.database.db_connection import create_db_engine

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

error_at_ingestion = EmailOperator(
    task_id="send_email_at_ingestion_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Ingestion",
    html_content="<p>Something went wrong at Ingestion stage.</p>",
    dag=dag,
)

error_at_preprocessing = EmailOperator(
    task_id="send_email_at_preprocessing_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Preprocessing",
    html_content="<p>Something went wrong at Preprocessing stage.</p>",
    dag=dag,
)

error_at_feature_engineering = EmailOperator(
    task_id="send_email_at_feature_engineering_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Feature Engineering",
    html_content="<p>Something went wrong at Feature Engineering stage.</p>",
    dag=dag,
)

error_at_DB_loading = EmailOperator(
    task_id="send_email_at_DB_loading_error",
    to="murtaza.sn786@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at DB Loading",
    html_content="<p>Something went wrong at DB Loading stage.</p>",
    dag=dag,
)

error_at_bias_analysis = EmailOperator(
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
    dag=dag,
)

slack_error_at_preprocessing = SlackWebhookOperator(
    task_id="send_slack_at_preprocessing_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Preprocessing* stage.",
    channel="#group-34",
    dag=dag,
)

slack_error_at_feature_engineering = SlackWebhookOperator(
    task_id="send_slack_at_feature_engineering_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *Feature Engineering* stage.",
    channel="#group-34",
    dag=dag,
)

slack_error_at_DB_loading = SlackWebhookOperator(
    task_id="send_slack_at_DB_loading_error",
    slack_webhook_conn_id="slack_webhook",
    message=":red_circle: *SavVio Data Pipeline* — Error at *DB Loading* stage.",
    channel="#group-34",
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

# Ingestion fan-in → Preprocessing (all three run in parallel)
[ingest_financial, ingest_products, ingest_reviews] >> [preprocess_financial, preprocess_products, preprocess_reviews]

#----------------------------------------------------
# Feature Engineering  (runs in PARALLEL, after preprocessing)
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

# Preprocessing fan-in → Feature Engineering (both run in parallel)
[preprocess_financial, preprocess_products, preprocess_reviews] >> [feature_financial, feature_reviews]



#----------------------------------------------------
# Airflow DAG for Loading Featured Data into PostgreSQL
#----------------------------------------------------

load_financial = PythonOperator(
    task_id='load_financial_profiles',
    python_callable=load_financial_profiles,
    dag=dag
)

load_products = PythonOperator(
    task_id='load_products',
    python_callable=load_products,
    op_kwargs={'rating_variance_path': 'data/features/product_rating_variance.csv'},
    dag=dag
)

load_reviews = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews,
    dag=dag
)

generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_and_load,
    dag=dag
)

# Feature Engineering fan-in → Loading
[feature_financial, feature_reviews] >> [load_financial, load_products, load_reviews]

[load_financial, load_products, load_reviews] >> generate_embeddings

#----------------------------------------------------
# Airflow DAG for Bias Analysis
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

# Run in parallel since they analyze independent tracks.
[bias_financial, bias_products] >> complete