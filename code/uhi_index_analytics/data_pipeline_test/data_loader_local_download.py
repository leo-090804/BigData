import os
import yaml
import google.auth
import google.cloud.storage as storage

# Define the project root within the Docker container for clarity if needed elsewhere,
# though for this script, paths are relative to __file__ or from config.
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code/data_pipeline_test/cache/raw_phase"

# Construct the absolute path to the gcloud_config.yaml file
GCLOUD_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "gcloud_config.yaml")
# DATA_FILE_PATH = "data_pipeline_test/cache/raw_phase/"

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
    nysp_data = config["raw_path"]["nysp"]
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

    BUCKET.blob(building_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "building.geojson")),
    BUCKET.blob(street_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "street.geojson")),
    BUCKET.blob(nyco_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "nyco.geojson")),
    BUCKET.blob(nysp_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "nysp.geojson")),
    BUCKET.blob(nyzd_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "nyzd.geojson")),
    BUCKET.blob(population_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "population.tif")),
    BUCKET.blob(canopy_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "canopy.tif")),
    BUCKET.blob(aod_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "aod.tif")),
    BUCKET.blob(co_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "co.tif")),
    BUCKET.blob(hcho_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "hcho.tif")),
    BUCKET.blob(no2_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "no2.tif")),
    BUCKET.blob(o3_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "o3.tif")),
    BUCKET.blob(so2_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "so2.tif")),
    BUCKET.blob(landsat_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "landsat.tiff")),
    BUCKET.blob(sentinel_data).download_to_filename(os.path.join(PROJECT_ROOT_IN_CONTAINER, "sentinel.tiff")),

    data = {
        "building": os.path.join(PROJECT_ROOT_IN_CONTAINER, "building.geojson"),
        "street": os.path.join(PROJECT_ROOT_IN_CONTAINER, "street.geojson"),
        "nyco": os.path.join(PROJECT_ROOT_IN_CONTAINER, "nyco.geojson"),
        "nysp": os.path.join(PROJECT_ROOT_IN_CONTAINER, "nysp.geojson"),
        "nyzd": os.path.join(PROJECT_ROOT_IN_CONTAINER, "nyzd.geojson"),
        "population": os.path.join(PROJECT_ROOT_IN_CONTAINER, "population.tif"),
        "canopy": os.path.join(PROJECT_ROOT_IN_CONTAINER, "canopy.tif"),
        "aod": os.path.join(PROJECT_ROOT_IN_CONTAINER, "aod.tif"),
        "co": os.path.join(PROJECT_ROOT_IN_CONTAINER, "co.tif"),
        "hcho": os.path.join(PROJECT_ROOT_IN_CONTAINER, "hcho.tif"),
        "no2": os.path.join(PROJECT_ROOT_IN_CONTAINER, "no2.tif"),
        "o3": os.path.join(PROJECT_ROOT_IN_CONTAINER, "o3.tif"),
        "so2": os.path.join(PROJECT_ROOT_IN_CONTAINER, "so2.tif"),
        "landsat": os.path.join(PROJECT_ROOT_IN_CONTAINER, "landsat.tiff"),
        "sentinel": os.path.join(PROJECT_ROOT_IN_CONTAINER, "sentinel.tiff"),
    }
    return data


# if __name__ == "__main__":
#     data = data_loader()
#     print("Data downloaded successfully.")
    # You can add more logic here to process the downloaded data if needed.
