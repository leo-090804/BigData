import os
import yaml
import pandas as pd
from tqdm import tqdm
import rasterio
import rioxarray as rxr

# Define the project root within the Docker container
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

CSV_PATH = os.path.join(os.path.dirname(__file__), "mapping.csv") # Relative to this script
READ_DIR_BASE = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/")
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tabular_transform_phase/")
os.makedirs(SAVE_DIR_BASE, exist_ok=True) # Ensure base save directory exists

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml") # Relative to this script
with open(CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)
COORDS = config["coords"]


def map_satellite_data(tiff_path, csv_path, save_path):

    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    layer_num = 0
    with rasterio.open(tiff_path) as dts:
        layer_num = dts.count
        layer_names = dts.descriptions

    # Read the Excel file using pandas
    df = pd.read_csv(csv_path)
    latitudes = df["Latitude"].values
    longitudes = df["Longitude"].values

    df = pd.DataFrame()
    df["lat"] = latitudes
    df["long"] = longitudes

    for i in tqdm(range(layer_num), desc="Go through layer"):
        values = []
        # Iterate over the latitudes and longitudes, and extract the corresponding band values
        for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
            # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names from data.coords)
            cell_value = data.sel(x=lon, y=lat, band=i + 1, method="nearest").values
            values.append(cell_value)
        # Add column of feature
        df[layer_names[i]] = values

    df.to_csv(save_path, index=False)
    print("File saved!")

    return df

def mapping(size):
    read_folder = os.path.join(READ_DIR_BASE, f"{size}") # Use absolute base path
    save_folder = os.path.join(SAVE_DIR_BASE, f"{size}") # Use absolute base path
    os.makedirs(save_folder, exist_ok=True)
    
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc=f"{size}"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)

# The gg_cloud_saving function is removed as its functionality is integrated into mapping.
