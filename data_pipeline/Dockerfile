# latest Airflow image
FROM apache/airflow:3.1.7

# Copy both the data pipeline requirements and the savviocore package
COPY data_pipeline/data-requirements.txt .
COPY --chown=airflow:0 savviocore /opt/airflow/savviocore

# Install savviocore package and data pipeline requirements
RUN pip install -e /opt/airflow/savviocore
RUN pip install -r data-requirements.txt