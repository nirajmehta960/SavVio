# latest Airflow image
FROM apache/airflow:3.1.7

COPY data_requirements.txt .

# Install dependencies for data pipeline
RUN pip install -r data_requirements.txt