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
    # spark_session
)

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'uhi_index_analytics_pipeline_decompose',
    default_args=default_args,
    description='UHI Index Analytics Data Pipeline Decomposition',
    schedule_interval=None,  # Set to None for manual triggering or provide a cron expression
    catchup=False,
    tags=['uhi', 'analytics', 'data_pipeline'],
)

# Create a SparkSession for the DAG execution
# def create_spark_session(**kwargs):
#     """Create a Spark session for the pipeline"""
#     from data_pipeline_test import spark_session
#     spark = spark_session.create_spark_session()
#     kwargs['ti'].xcom_push(key='spark_session_created', value=True)
#     print("Spark session created successfully!")
#     return True

# def terminate_spark_session(**kwargs):
#     """Terminate the Spark session after the pipeline completes"""
#     from data_pipeline_test import spark_session
#     import pyspark

#     # Try to get a SparkSession that's already started
#     try:
#         spark = pyspark.sql.SparkSession.builder.getOrCreate()
#         spark.stop()
#         print("Spark session terminated successfully!")
#     except Exception as e:
#         print(f"Error terminating Spark session: {e}")
#     return True

# Define task functions (breaking down each part of your pipeline)

def load_data(**kwargs):
    """Load raw data from Google Cloud Storage"""
    data = data_loader_local_download.data_loader()
    # Store the data in XCom for other tasks to use
    kwargs['ti'].xcom_push(key='raw_data', value=data)
    print("Data loaded successfully!")
    return True

def process_building_data(**kwargs):
    """Process building data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["building"], type="building")
    print("Building data processed!")
    return True

def process_street_data(**kwargs):
    """Process street data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["street"], type="street")
    print("Street data processed!")
    return True

def process_nyco_data(**kwargs):
    """Process nyco data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["nyco"], type="nyco")
    print("NYCO data processed!")
    return True

def process_nysp_data(**kwargs):
    """Process nysp data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["nysp"], type="nysp")
    print("NYSP data processed!")
    return True

def process_nyzd_data(**kwargs):
    """Process nyzd data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["nyzd"], type="nyzd")
    print("NYZD data processed!")
    return True

def process_population_data(**kwargs):
    """Process population data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["population"], type="population")
    print("Population data processed!")
    return True

def process_canopy_data(**kwargs):
    """Process canopy data"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    preprocess.input_data_handle(data["canopy"], type="canopy")
    print("Canopy data processed!")
    return True

def extract_building_features(**kwargs):
    """Extract features from building data"""
    extract_features.input_data_handle("building.geojson", type="building")
    print("Building features extracted!")
    return True

def extract_street_features(**kwargs):
    """Extract features from street data"""
    extract_features.input_data_handle("street.geojson", type="street")
    print("Street features extracted!")
    return True

def extract_nyco_features(**kwargs):
    """Extract features from nyco data"""
    extract_features.input_data_handle("nyco.geojson", type="nyco")
    print("NYCO features extracted!")
    return True

def extract_nysp_features(**kwargs):
    """Extract features from nysp data"""
    extract_features.input_data_handle("nysp.geojson", type="nysp")
    print("NYSP features extracted!")
    return True

def extract_nyzd_features(**kwargs):
    """Extract features from nyzd data"""
    extract_features.input_data_handle("nyzd.geojson", type="nyzd")
    print("NYZD features extracted!")
    return True

def convert_building_to_tiff(**kwargs):
    """Convert building data to TIFF"""
    tiff_transform.input_data_handle("building.geojson", type="building", resolution=30)
    tiff_transform.input_data_handle("building.geojson", type="building", resolution=100)
    print("Building data converted to TIFF!")
    return True

def convert_street_to_tiff(**kwargs):
    """Convert street data to TIFF"""
    tiff_transform.input_data_handle("street.geojson", type="street", resolution=30)
    tiff_transform.input_data_handle("street.geojson", type="street", resolution=100)
    tiff_transform.input_data_handle("street.geojson", type="street", resolution=500)
    tiff_transform.input_data_handle("street.geojson", type="street", resolution=1000)
    print("Street data converted to TIFF!")
    return True

def convert_zoning_to_tiff(**kwargs):
    """Convert zoning data to TIFF"""
    tiff_transform.input_data_handle("nyco.geojson", type="nyco", resolution=30)
    tiff_transform.input_data_handle("nysp.geojson", type="nysp", resolution=30)
    tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=30)
    tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=100)
    tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=200)
    tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=500)
    tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=1000)
    print("Zoning data converted to TIFF!")
    return True

def convert_satellite_to_tiff(**kwargs):
    """Convert satellite data to TIFF"""
    ti = kwargs['ti']
    data = ti.xcom_pull(key='raw_data', task_ids='load_data')
    
    tiff_transform.input_data_handle(data['aod'], type="aod", resolution=None)
    tiff_transform.input_data_handle(data['co'], type="co", resolution=None)
    tiff_transform.input_data_handle(data['hcho'], type="hcho", resolution=None)
    tiff_transform.input_data_handle(data['no2'], type="no2", resolution=None)
    tiff_transform.input_data_handle(data['so2'], type="so2", resolution=None)
    tiff_transform.input_data_handle(data['o3'], type="o3", resolution=None)
    
    tiff_transform.input_data_handle(data['landsat'], type="landsat", resolution=None)
    tiff_transform.input_data_handle(data['sentinel'], type="sentinel", resolution=None)
    print("Satellite data converted to TIFF!")
    return True

def convert_canopy_to_tiff(**kwargs):
    """Convert canopy data to TIFF"""
    tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=5)
    tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=10)
    tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=30)
    print("Canopy data converted to TIFF!")
    return True

def perform_base_feature_engineering(**kwargs):
    """Perform base feature engineering (no sliding window)"""
    features_engineering.calculate_indices(tifffile='landsat_8.tiff', source="landsat_8", savefile='1x1/landsat_indices.tiff', resolution=30)
    features_engineering.calculate_indices(tifffile='sentinel_2.tiff', source="sentinel_2", savefile='1x1/sentinel_indices.tiff', resolution=30)
    features_engineering.building_street_features(building_tiff='building_res30.tif', street_tiff='street_res30.tif', savefile='1x1/building_street_res30.tif', resolution=30)
    features_engineering.building_street_features(building_tiff='building_res100.tif', street_tiff='street_res100.tif', savefile='1x1/building_street_res100.tif', resolution=100)
    features_engineering.zoning_distance(tiff_file='nyzd_res30.tif', savefile='1x1/zoning_res30_distance.tiff')
    print("Base feature engineering completed!")
    return True

def perform_glcm_operations(window_size, **kwargs):
    """Perform GLCM operations with the given window size"""
    features_engineering.perform_glcm_operations(size=window_size)
    print(f"GLCM operations with window size {window_size} completed!")
    return True

def convert_to_tabular(size, **kwargs):
    """Convert data to tabular format for a specific size"""
    tabular_transform.mapping(size=size)
    print(f"Data converted to tabular format for size {size}!")
    return True

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

# Preprocessing tasks
process_building_task = PythonOperator(task_id='process_building', python_callable=process_building_data, provide_context=True, dag=dag)
process_street_task = PythonOperator(task_id='process_street', python_callable=process_street_data, provide_context=True, dag=dag)
process_nyco_task = PythonOperator(task_id='process_nyco', python_callable=process_nyco_data, provide_context=True, dag=dag)
process_nysp_task = PythonOperator(task_id='process_nysp', python_callable=process_nysp_data, provide_context=True, dag=dag)
process_nyzd_task = PythonOperator(task_id='process_nyzd', python_callable=process_nyzd_data, provide_context=True, dag=dag)
process_population_task = PythonOperator(task_id='process_population', python_callable=process_population_data, provide_context=True, dag=dag)
process_canopy_task = PythonOperator(task_id='process_canopy', python_callable=process_canopy_data, provide_context=True, dag=dag)

# Feature extraction tasks
extract_building_features_task = PythonOperator(task_id='extract_building_features', python_callable=extract_building_features, provide_context=True, dag=dag)
extract_street_features_task = PythonOperator(task_id='extract_street_features', python_callable=extract_street_features, provide_context=True, dag=dag)
extract_nyco_features_task = PythonOperator(task_id='extract_nyco_features', python_callable=extract_nyco_features, provide_context=True, dag=dag)
extract_nysp_features_task = PythonOperator(task_id='extract_nysp_features', python_callable=extract_nysp_features, provide_context=True, dag=dag)
extract_nyzd_features_task = PythonOperator(task_id='extract_nyzd_features', python_callable=extract_nyzd_features, provide_context=True, dag=dag)

# TIFF conversion tasks
convert_building_to_tiff_task = PythonOperator(task_id='convert_building_to_tiff', python_callable=convert_building_to_tiff, provide_context=True, dag=dag)
convert_street_to_tiff_task = PythonOperator(task_id='convert_street_to_tiff', python_callable=convert_street_to_tiff, provide_context=True, dag=dag)
convert_zoning_to_tiff_task = PythonOperator(task_id='convert_zoning_to_tiff', python_callable=convert_zoning_to_tiff, provide_context=True, dag=dag)
convert_satellite_to_tiff_task = PythonOperator(task_id='convert_satellite_to_tiff', python_callable=convert_satellite_to_tiff, provide_context=True, dag=dag)
convert_canopy_to_tiff_task = PythonOperator(task_id='convert_canopy_to_tiff', python_callable=convert_canopy_to_tiff, provide_context=True, dag=dag)

# Feature engineering tasks
perform_base_feature_engineering_task = PythonOperator(task_id='perform_base_feature_engineering', python_callable=perform_base_feature_engineering, provide_context=True, dag=dag)

glcm_tasks = {}
window_sizes = [1, 2, 4, 7]
for size in window_sizes:
    task_id = f'perform_glcm_operations_{size}'
    glcm_tasks[size] = PythonOperator(
        task_id=task_id,
        python_callable=perform_glcm_operations,
        op_kwargs={'window_size': size},
        provide_context=True,
        dag=dag,
    )

# Tabular conversion tasks
tabular_tasks = {}
tabular_sizes = ['1x1', '3x3', '5x5', '9x9', '15x15']
for size in tabular_sizes:
    task_id = f'convert_to_tabular_{size}'
    tabular_tasks[size] = PythonOperator(
        task_id=task_id,
        python_callable=convert_to_tabular,
        op_kwargs={'size': size},
        provide_context=True,
        dag=dag,
    )

# Add Spark session tasks
# create_spark_session_task = PythonOperator(
#     task_id='create_spark_session',
#     python_callable=create_spark_session,
#     provide_context=True,
#     dag=dag,
# )

# terminate_spark_session_task = PythonOperator(
#     task_id='terminate_spark_session',
#     python_callable=terminate_spark_session,
#     provide_context=True,
#     dag=dag,
#     trigger_rule='all_done',  # Run this task whether upstream tasks succeeded or failed
# )

# Define task dependencies

# Data loading -> preprocessing
load_data_task >> [
    process_building_task, process_street_task, process_nyco_task,
    process_nysp_task, process_nyzd_task, process_population_task, process_canopy_task
]

# Preprocessing -> feature extraction
process_building_task >> extract_building_features_task
process_street_task >> extract_street_features_task
process_nyco_task >> extract_nyco_features_task
process_nysp_task >> extract_nysp_features_task
process_nyzd_task >> extract_nyzd_features_task

process_canopy_task >> convert_canopy_to_tiff_task

# Feature extraction -> TIFF conversion
extract_building_features_task >> convert_building_to_tiff_task
extract_street_features_task >> convert_street_to_tiff_task
[extract_nyco_features_task, extract_nysp_features_task, extract_nyzd_features_task] >> convert_zoning_to_tiff_task

# Data loading -> satellite TIFF conversion (no need for feature extraction for satellite data)
load_data_task >> convert_satellite_to_tiff_task

# TIFF conversion -> Feature engineering
[convert_building_to_tiff_task, convert_street_to_tiff_task, convert_zoning_to_tiff_task, convert_satellite_to_tiff_task] >> perform_base_feature_engineering_task

# Base feature engineering -> GLCM operations
for size in window_sizes:
    perform_base_feature_engineering_task >> glcm_tasks[size]
    convert_canopy_to_tiff_task >> glcm_tasks[size]
    process_population_task >> glcm_tasks[size]

# Feature engineering -> Tabular conversion
feature_engineering_tasks = [perform_base_feature_engineering_task] + list(glcm_tasks.values())
for size in tabular_sizes:
    for task in feature_engineering_tasks:
        task >> tabular_tasks[size]

# Update DAG structure to include Spark session tasks
# Spark session should be created at the beginning
# create_spark_session_task >> load_data_task

# # All tabular tasks should be completed before terminating Spark
# for size in tabular_sizes:
#     tabular_tasks[size] >> terminate_spark_session_task
