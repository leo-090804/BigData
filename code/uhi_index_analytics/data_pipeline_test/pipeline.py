from data_pipeline_test import (
    data_loader, 
    preprocess, 
    extract_features, 
    tiff_transform, 
    features_engineering, 
    tabular_transform,
    spark_session
)
import yaml
import os # Import os

# Define the project root within the Docker container
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

GCLOUD_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "gcloud_config.yaml") # Relative to this script
with open(GCLOUD_CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)

class DataPipeline:
    def __init__(self):
        self.spark = spark_session.create_spark_session()
        
        self.data = data_loader.data_loader()
        print("Data loaded successfully!")

        self.preprocess()
        print("Preprocessing completed!")

        self.features_extraction()
        print("Feature extraction completed!")

        self.convert_to_tiff()
        print("Conversion to TIFF completed!")
        
        self.perform_feature_engineering()
        print("Feature engineering completed!")
        
        self.convert_to_tabular()
        print("Conversion to tabular format completed!")
        
        self.spark.stop()
        print("Spark session stopped!")

    def preprocess(self):
        preprocess.input_data_handle(self.data["building"], type="building")
        preprocess.input_data_handle(self.data["street"], type="street")
        preprocess.input_data_handle(self.data["nyco"], type="nyco")
        preprocess.input_data_handle(self.data["nysp"], type="nysp")
        preprocess.input_data_handle(self.data["nyzd"], type="nyzd")

        preprocess.input_data_handle(self.data["population"], type="population")
        preprocess.input_data_handle(self.data["canopy"], type="canopy")

    def features_extraction(self):
        # Note: extract_features.py needs to be updated to construct absolute paths
        # for these input filenames, e.g., by prepending a base cache directory like
        # os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/preprocess_phase/", filename)
        extract_features.input_data_handle("building.geojson", type="building")
        extract_features.input_data_handle("street.geojson", type="street")
        extract_features.input_data_handle("nyco.geojson", type="nyco")
        extract_features.input_data_handle("nysp.geojson", type="nysp")
        extract_features.input_data_handle("nyzd.geojson", type="nyzd")

    def convert_to_tiff(self):
        # Note: tiff_transform.py needs to be updated to construct absolute paths
        # for these input filenames (e.g., from preprocess_phase cache) and output files.
        tiff_transform.input_data_handle("building.geojson", type="building", resolution=30)
        tiff_transform.input_data_handle("building.geojson", type="building", resolution=100)

        tiff_transform.input_data_handle("street.geojson", type="street", resolution=30)
        tiff_transform.input_data_handle("street.geojson", type="street", resolution=100)
        tiff_transform.input_data_handle("street.geojson", type="street", resolution=500)
        tiff_transform.input_data_handle("street.geojson", type="street", resolution=1000)

        tiff_transform.input_data_handle("nyco.geojson", type="nyco", resolution=30)
        tiff_transform.input_data_handle("nysp.geojson", type="nysp", resolution=30)
        tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=30)
        tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=100)
        tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=200)
        tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=500)
        tiff_transform.input_data_handle("nyzd.geojson", type="nyzd", resolution=1000)
        
        tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=5)
        tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=10)
        tiff_transform.input_data_handle("canopy_height_res1.tif", type="canopy", resolution=30)

        tiff_transform.input_data_handle(self.data['aod'], type="aod", resolution=None)
        tiff_transform.input_data_handle(self.data['co'], type="co", resolution=None)
        tiff_transform.input_data_handle(self.data['hcho'], type="hcho", resolution=None)
        tiff_transform.input_data_handle(self.data['no2'], type="no2", resolution=None)
        tiff_transform.input_data_handle(self.data['so2'], type="so2", resolution=None)
        tiff_transform.input_data_handle(self.data['o3'], type="o3", resolution=None)

        tiff_transform.input_data_handle(self.data['landsat'], type="landsat", resolution=None)
        tiff_transform.input_data_handle(self.data['sentinel'], type="sentinel", resolution=None)

    def perform_feature_engineering(self):
        sliding_window_size = ["base", 1, 2, 4, 7]
        for size in sliding_window_size:
            if size == "base":
                # Note: features_engineering.py needs to be updated to construct absolute paths
                # for tifffile (e.g., from tiff_transform_phase cache) and savefile.
                features_engineering.calculate_indices(tifffile='landsat_8.tiff', source="landsat_8", savefile='1x1/landsat_indices.tiff', resolution=30)
                features_engineering.calculate_indices(tifffile='sentinel_2.tiff', source="sentinel_2", savefile='1x1/sentinel_indices.tiff', resolution=30)

                features_engineering.building_street_features(building_tiff='building_res30.tif', street_tiff='street_res30.tif', savefile='1x1/building_street_res30.tif', resolution=30)
                features_engineering.building_street_features(building_tiff='building_res100.tif', street_tiff='street_res100.tif', savefile='1x1/building_street_res100.tif', resolution=100)

                features_engineering.zoning_distance(tiff_file='nyzd_res30.tif', savefile='1x1/zoning_res30_distance.tiff')
                
            else:
                # Note: features_engineering.py needs to be updated for perform_glcm_operations
                # to handle input/output paths correctly.
                features_engineering.perform_glcm_operations(size=size)
    
    def convert_to_tabular(self):
        sizes = ['1x1' ,'3x3', '5x5', '9x9', '15x15']
        for size in sizes:
            tabular_transform.mapping(size=size)

if __name__ == "__main__":
    pipeline = DataPipeline()