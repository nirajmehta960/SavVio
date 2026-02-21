from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task
from src.features.run_features import feature_financial_task, feature_review_task


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

# -------- Pipeline Start Logger --------

def log_pipeline_start():
    start_time = time.time()
    logger.info("🚀 Data Pipeline started")
    logger.info(f"Start timestamp: {start_time}")
    return start_time


log_start = PythonOperator(
    task_id="log_pipeline_start",
    python_callable=log_pipeline_start,
    dag=dag,
)


send_email = EmailOperator(
    task_id="send_email",
    to="murtaza.sn786@gmail.com",
    subject="Notification from Data Pipeline Airflow",
    html_content="<p>This is a notification email sent from Data Pipeline Airflow.</p>",
    dag=dag,
)

# -------- Ingestion Stage Logger --------

def log_ingestion_start():
    logger.info("📥 Ingestion stage started")
    return time.time()

def log_ingestion_end(ti):
    start_time = ti.xcom_pull(task_ids="log_ingestion_start")
    end_time = time.time()

    if start_time:
        duration = round(end_time - float(start_time), 2)
        logger.info(f"📥 Ingestion stage completed in {duration} seconds")

log_ingestion_start_task = PythonOperator(
    task_id="log_ingestion_start",
    python_callable=log_ingestion_start,
    dag=dag,
)

log_ingestion_end_task = PythonOperator(
    task_id="log_ingestion_end",
    python_callable=log_ingestion_end,
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

# -------- Preprocessing Stage Logger --------

def log_preprocessing_start():
    logger.info("🧹 Preprocessing stage started")
    return time.time()

def log_preprocessing_end(ti):
    start_time = ti.xcom_pull(task_ids="log_preprocessing_start")
    end_time = time.time()

    if start_time:
        duration = round(end_time - float(start_time), 2)
        logger.info(f"🧹 Preprocessing stage completed in {duration} seconds")


log_preprocessing_start_task = PythonOperator(
    task_id="log_preprocessing_start",
    python_callable=log_preprocessing_start,
    dag=dag,
)

log_preprocessing_end_task = PythonOperator(
    task_id="log_preprocessing_end",
    python_callable=log_preprocessing_end,
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

# Start → Ingestion Logger → Ingestion

log_start >> log_ingestion_start_task
log_ingestion_start_task >> [ingest_financial, ingest_products, ingest_reviews]

# Ingestion fan-in → Ingestion End Logger

[ingest_financial, ingest_products, ingest_reviews] >> log_ingestion_end_task

# Ingestion End → Preprocessing Start Logger

log_ingestion_end_task >> log_preprocessing_start_task

# Preprocessing Start → Preprocessing Tasks

log_preprocessing_start_task >> [preprocess_financial, preprocess_products, preprocess_reviews]

# Preprocessing fan-in → Preprocessing End Logger

[preprocess_financial, preprocess_products, preprocess_reviews] >> log_preprocessing_end_task

#----------------------------------------------------
# Feature Engineering  (runs in PARALLEL, after preprocessing)
#----------------------------------------------------

# -------- Feature Engineering Stage Logger --------

def log_feature_start():
    logger.info("⚙️ Feature Engineering stage started")
    return time.time()

def log_feature_end(ti):
    start_time = ti.xcom_pull(task_ids="log_feature_start")
    end_time = time.time()

    if start_time:
        duration = round(end_time - float(start_time), 2)
        logger.info(f"⚙️ Feature Engineering stage completed in {duration} seconds")


log_feature_start_task = PythonOperator(
    task_id="log_feature_start",
    python_callable=log_feature_start,
    dag=dag,
)

log_feature_end_task = PythonOperator(
    task_id="log_feature_end",
    python_callable=log_feature_end,
    dag=dag,
)

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

# Preprocessing End → Feature Start Logger

log_preprocessing_end_task >> log_feature_start_task

# Feature Start → Feature Tasks

log_feature_start_task >> [feature_financial, feature_reviews]

# Feature fan-in → Feature End Logger

[feature_financial, feature_reviews] >> log_feature_end_task


#----------------------------------------------------
# Airflow DAG for Loading Featured Data into PostgreSQL
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

# Feature End → Loading

log_feature_end_task >> [load_financial, load_products, load_reviews]

[load_financial, load_products, load_reviews] >> generate_embeddings

generate_embeddings >> [bias_financial, bias_products]
[bias_financial, bias_products] >> log_end

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

# -------- Pipeline End Logger --------

def log_pipeline_end(ti):
    start_time = ti.xcom_pull(task_ids="log_pipeline_start")
    end_time = time.time()

    if start_time:
        duration = round(end_time - float(start_time), 2)
        logger.info("✅ Data Pipeline completed successfully")
        logger.info(f"Total runtime: {duration} seconds")
    else:
        logger.warning("Start time not found in XCom")


log_end = PythonOperator(
    task_id="log_pipeline_end",
    python_callable=log_pipeline_end,
    trigger_rule=TriggerRule.ALL_DONE,  # runs even if upstream fails
    dag=dag,
)
