import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import yaml

# Import your modules
from data_pipeline_test import (
    data_loader_local_download,
    data_loader,
    preprocess,
    extract_features,
    tiff_transform,
    features_engineering,
    tabular_transform,
    spark_session,
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    "uhi_index_analytics_pipeline",
    default_args=default_args,
    description="UHI Index Analytics Data Pipeline",
    schedule_interval=None,  # Set to None for manual triggering or provide a cron expression
    catchup=False,
    tags=["uhi", "analytics", "data_pipeline"],
)


def prepare_data(**kwargs):
    # Load data using the local download method
    data = data_loader_local_download.data_loader()
    kwargs["ti"].xcom_push(key="raw_data", value=data)
    print("Data loaded successfully!")

def preprocess_data(**kwargs):
    # Get the raw data from XCom
    raw_data = kwargs["ti"].xcom_pull(key="raw_data")
    # Preprocess the data
    processed_data = preprocess.process_data(raw_data)
    kwargs["ti"].xcom_push(key="processed_data", value=processed_data)