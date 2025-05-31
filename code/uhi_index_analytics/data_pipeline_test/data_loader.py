from google.cloud import storage
import google.auth
import os
import yaml

# Define the project root within the Docker container for clarity if needed elsewhere,
# though for this script, paths are relative to __file__ or from config.
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

# Construct the absolute path to the gcloud_config.yaml file
GCLOUD_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "gcloud_config.yaml")

with open(GCLOUD_CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)

# Assuming keyfile_path in gcloud_config.yaml is relative to the directory containing gcloud_config.yaml
# (i.e., data_pipeline_test directory)
KEYFILE_ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), config["keyfile_path"])
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEYFILE_ABSOLUTE_PATH
credentials, project = google.auth.default()

def data_loader():
    storage_client = storage.Client()
    BUCKET = storage_client.get_bucket(config["bucket_name"])
    
# 
    # process -> extract -> transform -> tiff -> engineering -> tabular
    building_data = config["raw_path"]["building"]
    street_data = config["raw_path"]["street"]
    nyco_data = config["raw_path"]["nyco"]
    nysp_data =  config["raw_path"]["nysp"]
    nyzd_data = config["raw_path"]["nyzd"]

    # process -> tiff -> engineering -> tabular
    population_data = config["raw_path"]["population"]
    canopy_data = config["raw_path"]["canopy"]

    # engineering -> tabular
    aod_data = config["raw_path"]["aod"]
    co_data = config["raw_path"]["co"]
    hcho_data = config["raw_path"]["hcho"]
    no2_data = config["raw_path"]["no2"]
    o3_data = config["raw_path"]["o3"]
    so2_data = config["raw_path"]["so2"]

    # engineering -> tabular
    landsat_data = config["raw_path"]["landsat"]
    sentinel_data = config["raw_path"]["sentinel"]
    
    data = {
        "building": BUCKET.blob(building_data).download_as_string(),
        "street": BUCKET.blob(street_data).download_as_string(),
        "nyco": BUCKET.blob(nyco_data).download_as_string(),
        "nysp": BUCKET.blob(nysp_data).download_as_string(),
        "nyzd": BUCKET.blob(nyzd_data).download_as_string(),
        "population": BUCKET.blob(population_data).download_as_bytes(),
        "canopy": BUCKET.blob(canopy_data).download_as_bytes(), 
        "aod": BUCKET.blob(aod_data).download_as_bytes(), 
        "co": BUCKET.blob(co_data).download_as_bytes(),
        "hcho": BUCKET.blob(hcho_data).download_as_bytes(),
        "no2": BUCKET.blob(no2_data).download_as_bytes(),
        "o3": BUCKET.blob(o3_data).download_as_bytes(),
        "so2": BUCKET.blob(so2_data).download_as_bytes(),
        "landsat": BUCKET.blob(landsat_data).download_as_bytes(),    
        "sentinel": BUCKET.blob(sentinel_data).download_as_bytes()
    }
    
    return data
