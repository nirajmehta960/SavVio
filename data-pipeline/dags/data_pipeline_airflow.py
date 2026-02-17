from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Murtaza Nipplewala": "https://www.linkedin.com/in/murtazan/"},
    max_active_runs=1,
)

# -------------------- TASKS ------------------------

send_email = EmailOperator(
    task_id="send_email",
    to="murtaza.sn786@gmail.com",
    subject="Notification from Airflow",
    html_content="<p>This is a notification email sent from Airflow.</p>",
    dag=dag,
)

#----------------------------------------------------
# Airflow DAG for Data Ingestion
#----------------------------------------------------

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