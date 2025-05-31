from sedona.spark import *
import yaml
import os # Import os module

# Define the project root within the Docker container for clarity if needed elsewhere,
# though for this script, paths are relative to __file__ or from config.
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

# Construct the absolute path to the gcloud_config.yaml file
GCLOUD_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "gcloud_config.yaml")

with open(GCLOUD_CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)

def create_spark_session(
    core: int = 6,
    driver_memory: str = "8g", 
    gcp_project_id: str = config["project_id"], # Default to the project ID from the config
    # gcp_keyfile_path parameter will now expect an absolute path or a path relative to project root
    # The default value from config needs to be made absolute here.
    gcp_keyfile_path_from_config: str = config["keyfile_path"] 
):
    # Assuming keyfile_path in gcloud_config.yaml is relative to the directory containing gcloud_config.yaml
    # (i.e., data_pipeline_test directory)
    # Or, if it's meant to be relative to the project root:
    # KEYFILE_ABSOLUTE_PATH = os.path.join(PROJECT_ROOT_IN_CONTAINER, gcp_keyfile_path_from_config)
    # For consistency with data_loader.py and preprocess.py, let's assume it's relative to the script's dir
    KEYFILE_ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), gcp_keyfile_path_from_config)

    # If gcp_keyfile_path argument is provided to the function, it should already be absolute
    # or handled appropriately by the caller. Here we are setting the default.

    # Assuming you are using SedonaContext.builder() based on features_extraction.py
    # If not, use SparkSession.builder()
    builder = SedonaContext.builder() # Or SparkSession.builder()

    # Your existing configurations (application name, master, serializer, etc.)
    builder = builder.config("spark.app.name", "GeoSpatialPreprocessing")
    builder = builder.master(f"local[{core}]")
    builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    builder = builder.config(
        "spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator"
    )  # Add this line
    builder = builder.config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions")
    builder = builder.config("spark.driver.memory", driver_memory)

    # Prioritize user classpath to resolve Guava conflicts
    builder = builder.config("spark.driver.userClassPathFirst", "true")
    builder = builder.config("spark.executor.userClassPathFirst", "true")

    # Add existing packages and the GCS connector
    # Get existing_packages from your current spark_session.py or features_extraction.py
    # existing_packages = (
    #     "org.apache.sedona:sedona-spark-shaded-3.0_2.12:1.4.1,org.datasyslab:geotools-wrapper:1.7.1-28.5"
    # )
    gcs_connector_package = "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.12" # Changed from 2.2.18
    # all_packages = f"{existing_packages},{gcs_connector_package}"
    builder = builder.config("spark.jars.packages", gcs_connector_package)

    # Configure Hadoop for GCS
    builder = builder.config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    builder = builder.config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")

    # --- GCS Authentication and Project ID ---
    builder = builder.config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    builder = builder.config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEYFILE_ABSOLUTE_PATH)
    builder = builder.config("spark.hadoop.fs.gs.project.id", gcp_project_id)

    spark_session = builder.getOrCreate()

    # from sedona.register import SedonaRegistrator
    # SedonaRegistrator.registerAll(spark_session)

    return spark_session
