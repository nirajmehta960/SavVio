from __future__ import annotations

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task


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
# Data Preprocessing  (runs in PARALLEL, after ingestion)
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