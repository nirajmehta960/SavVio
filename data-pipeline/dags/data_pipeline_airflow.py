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


# ==================== FAILURE CALLBACK ====================
# Fires ONLY when a task actually fails — NOT on upstream_failed (skipped).
# This prevents the cascading email/Slack flood.

ALERT_EMAIL = "murtaza.sn786@gmail.com"
SLACK_CHANNEL = "#group-34"

# Map task_id prefixes to human-readable stage names
STAGE_LABELS = {
    "ingest":      "Ingestion",
    "validate_raw": "Raw Validation",
    "preprocess":  "Preprocessing",
    "validate_processed": "Processed Validation",
    "feature":     "Feature Engineering",
    "validate_featured": "Featured Validation",
    "load":        "DB Loading",
    "generate":    "DB Loading (Embeddings)",
}

def _get_stage(task_id: str) -> str:
    """Derive a human-readable stage name from the task_id."""
    for prefix, label in STAGE_LABELS.items():
        if task_id.startswith(prefix):
            return label
    return task_id


def on_failure_alert(context):
    """
    on_failure_callback — sends Email + Slack alert when a task ACTUALLY fails.
    Does NOT fire when a task is merely skipped (upstream_failed).
    """
    ti = context["task_instance"]
    task_id = ti.task_id
    dag_id = ti.dag_id
    exec_date = context["logical_date"]
    stage = _get_stage(task_id)
    exception = context.get("exception", "Unknown error")

    # ----- Email -----
    email_task = EmailOperator(
        task_id=f"_alert_email_{task_id}",
        to=ALERT_EMAIL,
        subject=f"SavVio Data Pipeline — Error at {stage}",
        html_content=(
            f"<h3>Pipeline Error: {stage}</h3>"
            f"<p><b>Task:</b> {task_id}</p>"
            f"<p><b>DAG:</b> {dag_id}</p>"
            f"<p><b>Execution Date:</b> {exec_date}</p>"
            f"<p><b>Error:</b> {exception}</p>"
        ),
    )
    email_task.execute(context)

    # ----- Slack -----
    slack_task = SlackWebhookOperator(
        task_id=f"_alert_slack_{task_id}",
        slack_webhook_conn_id="slack_webhook",
        message=(
            f":red_circle: *SavVio Data Pipeline* — Error at *{stage}*\n"
            f">Task: `{task_id}`\n"
            f">Error: {exception}"
        ),
        channel=SLACK_CHANNEL,
    )
    slack_task.execute(context)


# ---------- Default args ----------
default_args = {
    'owner': 'Murtaza Nipplewala',
    'start_date': datetime(2026, 2, 17),
    'retries': 2,
    'retry_delay': timedelta(minutes=0.3),
    'email': 'murtaza.sn786@gmail.com',
    'email_on_failure': False,       # Handled by on_failure_callback instead
    'email_on_retry': False,
    'on_failure_callback': on_failure_alert,   # <-- ALL tasks get this
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

# ==================== SUCCESS ALERT OPERATORS ====================
# These use default ALL_SUCCESS, so they ONLY fire when the entire pipeline completes.

email_pipeline_success = EmailOperator(
    task_id="send_email_pipeline_success",
    to=ALERT_EMAIL,
    subject="SavVio Data Pipeline Airflow - Pipeline Completed Successfully",
    html_content="<p>All data has been successfully ingested, validated, processed, featured, and loaded into the database.</p>",
    dag=dag,
)

slack_pipeline_success = SlackWebhookOperator(
    task_id="send_slack_pipeline_success",
    slack_webhook_conn_id="slack_webhook",
    message=":large_green_circle: *SavVio Data Pipeline* — Pipeline completed *successfully*. All data loaded into DB.",
    channel=SLACK_CHANNEL,
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

#----------------------------------------------------
# 4. Processed Data Validation  (after ALL preprocessing succeeds)
#----------------------------------------------------

validate_processed_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed,
    dag=dag,
)

[preprocess_financial, preprocess_products, preprocess_reviews] >> validate_processed_data

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

#----------------------------------------------------
# 6. Featured Data Validation  (after ALL feature engineering succeeds)
#----------------------------------------------------

validate_featured_data = PythonOperator(
    task_id='validate_featured_data',
    python_callable=validate_features,
    dag=dag,
)

[feature_financial, feature_reviews] >> validate_featured_data

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