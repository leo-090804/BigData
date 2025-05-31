from google.cloud import storage
from google.oauth2 import service_account

import yaml

with open("data_pipeline/gcloud_config.yaml", "r") as file:
    config = yaml.safe_load(file)


def data_loader():
    storage_client = storage.Client()
    bucket_name = config["bucket_name"]
    prefix = "gs://{}/".format(bucket_name)

    # process -> extract -> transform -> tiff -> engineering -> tabular
    building_data = prefix + config["raw_path"]["building"]
    street_data = prefix + config["raw_path"]["street"]
    nyco_data = prefix + config["raw_path"]["nyco"]
    nysp_data = prefix + config["raw_path"]["nysp"]
    nyzd_data = prefix + config["raw_path"]["nyzd"]

    # process -> tiff -> engineering -> tabular
    population_data = prefix + config["raw_path"]["population"]
    canopy_data = prefix + config["raw_path"]["canopy"]

    # engineering -> tabular
    aod_data = prefix + config["raw_path"]["aod"]
    co_data = prefix + config["raw_path"]["co"]
    hcho_data = prefix + config["raw_path"]["hcho"]
    no2_data = prefix + config["raw_path"]["no2"]
    o3_data = prefix + config["raw_path"]["o3"]
    so2_data = prefix + config["raw_path"]["so2"]

    # engineering -> tabular
    landsat_data = prefix + config["raw_path"]["landsat"]
    sentinel_data = prefix + config["raw_path"]["sentinel"]

    data = {
        "building": building_data,
        "street": street_data,
        "nyco": nyco_data,
        "nysp": nysp_data,
        "nyzd": nyzd_data,
        "population": population_data,
        "canopy": canopy_data,
        "aod": aod_data,
        "co": co_data,
        "hcho": hcho_data,
        "no2": no2_data,
        "o3": o3_data,
        "so2": so2_data,
        "landsat": landsat_data,
        "sentinel": sentinel_data
    }
    
    return data