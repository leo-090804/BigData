import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
from rtree import index
from tqdm import tqdm
import rasterio
from pyproj import Transformer
from shapely.geometry import box, Polygon
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter
from scipy.stats import entropy
import io # Added import

# Define the project root within the Docker container
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

READ_DIR = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/features_extracted_phase/")
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/") # Base for different resolutions
SAVE_DIR_1x1 = os.path.join(SAVE_DIR_BASE, "1x1/") # Specific for 1x1, adjust if logic changes
os.makedirs(READ_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_1x1, exist_ok=True)
# Ensure other resolution-specific save directories are created as needed, e.g., in mapping() or here if fixed.

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml") # Relative to this script
with open(CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)
COORDS = config["coords"]
TARGET_CRS = config["satellite_config"]["params"]["crs"] # Added target CRS


def input_data_handle(file_or_bytes, resolution, type):
    # Determine the correct save directory based on type or resolution if necessary
    # For now, most seem to go to SAVE_DIR_1x1, but canopy_height might differ if resolution varies
    # For simplicity, using SAVE_DIR_1x1 for most, adjust if specific subfolders per resolution are needed for all.
    current_save_dir = SAVE_DIR_1x1 
    # If resolution implies a different subfolder structure for outputs:
    # current_save_dir = os.path.join(SAVE_DIR_BASE, f"res{resolution}") # Example
    # os.makedirs(current_save_dir, exist_ok=True)


    if type == "building":
        # file_or_bytes is a filename here
        read_path = os.path.join(READ_DIR, file_or_bytes)
        save_path = os.path.join(current_save_dir, f"building_res{resolution}.tif")
        building_tiff(read_path, save_path, resolution)
    elif type == "street":
        read_path = os.path.join(READ_DIR, file_or_bytes)
        save_path = os.path.join(current_save_dir, f"street_res{resolution}.tif")
        street_tiff(read_path, save_path, resolution)
    elif type == "nyco":
        read_path = os.path.join(READ_DIR, file_or_bytes)
        save_path = os.path.join(current_save_dir, f"nyco_res{resolution}.tif")
        zoning_nyco_tiff(read_path, save_path, resolution)
    elif type == "nysp":
        read_path = os.path.join(READ_DIR, file_or_bytes)
        save_path = os.path.join(current_save_dir, f"nysp_res{resolution}.tif")
        zoning_nysp_tiff(read_path, save_path, resolution)
    elif type == "nyzd":
        read_path = os.path.join(READ_DIR, file_or_bytes)
        save_path = os.path.join(current_save_dir, f"nyzd_res{resolution}.tif")
        zoning_nyzd_tiff(read_path, save_path, resolution)
    elif type == "canopy":
        # For canopy, 'file_or_bytes' is a filename from preprocess (e.g., canopy_height_res1.tif)
        # This read path might need to point to where preprocess saves it.
        # Assuming preprocess saves to a path like: PROJECT_ROOT_IN_CONTAINER/data_pipeline_test/cache/tiff_transform_phase/1x1/canopy_height_res1.tif
        # The 'file_or_bytes' argument in the DAG for this type should be the *filename* 'canopy_height_res1.tif'
        # and the read path constructed here.
        # However, the DAG currently passes raw bytes for canopy from data_loader.
        # Let's assume 'file_or_bytes' is the *output path* from the previous step if it's a file,
        # or raw bytes if it's satellite data.
        # The original `resample_canopy_height` call in `if __name__ == "__main__":` uses:
        # `readfile = SAVE_DIR + f"canopy_height_res1.tif"`
        # This implies the input is a file path.
        # If the DAG passes raw bytes for canopy, this needs adjustment or a separate function.
        # For now, assuming file_or_bytes is a filename for canopy type that needs resampling.
        # This path needs to be where the *input* for resampling is located.
        # Let's assume it's in SAVE_DIR_1x1 as per the __main__ block.
        read_path = os.path.join(SAVE_DIR_1x1, file_or_bytes) # file_or_bytes should be 'canopy_height_res1.tif'
        save_path = os.path.join(current_save_dir, f"canopy_height_res{resolution}.tif")
        resample_canopy_height(read_path, save_path, resolution)

    # For satellite data, file_or_bytes is raw bytes
    elif type == "aod":
        save_path = os.path.join(current_save_dir, "aod_median.tif")
        save_tiff(file_or_bytes, save_path, type)
    elif type == "co":
        save_path = os.path.join(current_save_dir, "co_20210724_165223.tif")
        save_tiff(file_or_bytes, save_path, type)
    elif type == "hcho":
        save_path = os.path.join(current_save_dir, "hcho_20210724_165223.tif")
        save_tiff(file_or_bytes, save_path, type)
    elif type == "no2": # Corrected duplicate no2
        save_path = os.path.join(current_save_dir, "no2_20210724_165223.tif")
        save_tiff(file_or_bytes, save_path, type)
    elif type == "o3":
        save_path = os.path.join(current_save_dir, "o3_20210724_165223.tif")
        save_tiff(file_or_bytes, save_path, type)
    # Removed duplicate no2 entry, ensure SO2 is handled if needed
    elif type == "so2": # Added SO2 handling if it was intended
        save_path = os.path.join(current_save_dir, "so2_20210724_165223.tif") # Example filename
        save_tiff(file_or_bytes, save_path, type)


    elif type == "landsat":
        save_path = os.path.join(current_save_dir, "landsat_8.tif")
        save_tiff(file_or_bytes, save_path, type)
    elif type == "sentinel":
        save_path = os.path.join(current_save_dir, "sentinel_2.tif")
        save_tiff(file_or_bytes, save_path, type)


def sum_area_values(clipped_df, geometry_col, grid_cell):
    # transform to feet
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    clipped_df_feet = clipped_df.to_crs("EPSG:2263")
    grid_cell_feet = Polygon([transformer.transform(x, y) for x, y in grid_cell.exterior.coords])

    total_area = 0
    for _, row in clipped_df_feet.iterrows():
        intersect_area = row[geometry_col].intersection(grid_cell_feet)  # Clip the building to the grid cell
        total_area += intersect_area.area

    return total_area


def mean_values(clipped_df, value_col):
    total = 0
    count = 0
    for _, row in clipped_df.iterrows():
        # if row[value_col] is not None:
        total += row[value_col]
        count += 1

    if count == 0:
        raster_value = 0
    else:
        raster_value = total / count

    return raster_value


def std_dev_values(clipped_df, value_col):
    all_values = []
    count = 0
    for _, row in clipped_df.iterrows():
        all_values.append(row[value_col])
        count += 1

    if count == 0:
        raster_value = -1
    elif count == 1:
        raster_value = 0
    else:
        raster_value = np.std(all_values)

    return raster_value


def mode_cat_values(clipped_df, value_col):
    all_values = []
    count = 0
    for _, row in clipped_df.iterrows():
        all_values.append(row[value_col])
        count += 1

    if count == 0:
        raster_value = -1
    elif count == 1:
        raster_value = all_values[0]
    else:
        raster_value = mode(all_values)[0]

    return raster_value


def count_values(clipped_df):
    count = 0
    for _, row in clipped_df.iterrows():
        count += 1

    if count == 0:
        raster_value = 0
    else:
        raster_value = count

    return raster_value


def entropy_values(clipped_df, value_col, bin_width=3):
    bins = np.arange(0, 180 + bin_width, bin_width)
    labels = [f"{int(b)}-{int(b+bin_width)}" for b in bins[:-1]]
    value_bin = pd.cut(clipped_df[value_col], bins=bins, labels=labels, include_lowest=True, right=False)

    # 2. Count frequencies per bin
    freq = value_bin.value_counts().sort_index()

    # 3. Convert to probabilities
    probs = freq / freq.sum()

    return entropy(probs, base=np.e)


## Wrapper functions


def calculate_raster_value(operation, clipped_df, geometry_col, value_col, grid_cell):
    if operation == "mean":
        raster_value = mean_values(clipped_df, value_col)
    elif operation == "sum_area":
        raster_value = sum_area_values(clipped_df, geometry_col, grid_cell)
    elif operation == "std_dev":
        raster_value = std_dev_values(clipped_df, value_col)
    elif operation == "mode":
        raster_value = mode_cat_values(clipped_df, value_col)
    elif operation == "count":
        raster_value = count_values(clipped_df)
    elif operation == "entropy":
        raster_value = entropy_values(clipped_df, value_col)
    else:
        print("Operation must be mean, sum, std_dev, or mode")
        return

    return raster_value


def rasterize(df, geometry_col, value_col, operations, gt, height, width):
    """
    resolution: meters per pixel
    operations: array of calculation for raster value. Elements can be the following: mean, sum, mode, std_dev
    """

    # Build the Rtree index for faster spatial queries
    idx = index.Index()
    for i, geometry in enumerate(df[geometry_col]):
        idx.insert(i, geometry.bounds)  # Insert bounding boxes into the index

    rasters = []
    for k in range(len(operations)):
        rasters.append(np.zeros((height, width), dtype=np.float32))

    for i in tqdm(range(width)):
        for j in range(height):
            # Get the bounds of the current raster cell (grid square)
            x_min, y_max = gt * (i, j)
            x_max, y_min = gt * (i + 1, j + 1)
            grid_cell = box(x_min, y_min, x_max, y_max)

            # Clip building footprints by the current grid cell
            possible_matches = list(idx.intersection(grid_cell.bounds))  # Find potential matches
            intersect_idx = [idx for idx in possible_matches if df.iloc[idx][geometry_col].intersects(grid_cell)]
            clipped_df = df.iloc[intersect_idx]
            for k in range(len(operations)):
                operation = operations[k]
                rasters[k][j, i] = calculate_raster_value(operation, clipped_df, geometry_col, value_col, grid_cell)

    rasters.append("_")

    return tuple(rasters)


###############################################################################################################
## Extract feature for raster and create tiff files

def save_tiff(aod_bytes, savefile, type):
    """
    Saves AOD TIFF data (from bytes) to a file.
    Assumes the data is already processed.
    """
    
    # with io.BytesIO(aod_bytes) as mem_tiff:
    #     with rasterio.open(mem_tiff) as src:
    #         profile = src.profile
    #         data = src.read()  # Reads all bands
    
    with rasterio.open(aod_bytes) as src:
        profile = src.profile
        data = src.read()

    # Update profile for saving, ensure block sizes are multiples of 16 for tiled TIFFs.
    profile.update(tiled=True, compress='LZW', blockxsize=256, blockysize=256) 

    with rasterio.open(savefile, 'w', **profile) as dst:
        dst.write(data)
        # Set band description for the first band (assuming single effective band or primary band)
        # If multiple bands, this will only describe the first.
        # Adjust if specific descriptions per band are needed.
        if dst.count >= 1:
            dst.set_band_description(1, f"{type}")
    


def building_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = "geometry"

    # Create raster bounds
    scale = resolution / 111320.0  # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_height, std_dev_height, _ = rasterize(df, geometry_col, "heightroof", ["mean", "std_dev"], gt, height, width)

    mean_year, _ = rasterize(df, geometry_col, "cnstrct_yr", ["mean"], gt, height, width)

    # building_area, _ = rasterize(df, geometry_col, 'geometry', ['sum_area'],
    # gt, height, width)

    # Save raster
    bands = [
        mean_height,
        std_dev_height,
        mean_year,
        #  building_area
    ]
    band_names = [
        "building_height",
        "building_height_std",
        "building_year",
        #   'building_area'
    ]
    
    with rasterio.open(savefile, 'w', driver='GTiff', count=4, crs=df.crs,
                       dtype=mean_height.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def street_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = "geometry"

    # Convert type
    df["RW_TYPE"] = df["RW_TYPE"].astype("int")

    # Encode categories
    encoder = LabelEncoder()
    df["TrafDir"] = encoder.fit_transform(df["TrafDir"])
    print(list(encoder.classes_))
    df["direction"] = encoder.fit_transform(df["direction"])
    print(list(encoder.classes_))

    # Fill missing values
    df["street_width_avg"] = df.groupby("RW_TYPE")["street_width_avg"].transform(lambda x: x.fillna(x.median()))
    df["Number_Total_Lanes"] = df.groupby("RW_TYPE")["Number_Total_Lanes"].transform(lambda x: x.fillna(x.median()))

    # Create raster bounds
    scale = resolution / 111320.0  # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_width, _ = rasterize(df, geometry_col, "street_width_avg", ["mean"], gt, height, width)

    traffic_dir, _ = rasterize(df, geometry_col, "TrafDir", ["mode"], gt, height, width)

    mean_lanes, _ = rasterize(df, geometry_col, "Number_Total_Lanes", ["mean"], gt, height, width)

    orientation, _ = rasterize(df, geometry_col, "direction", ["mode"], gt, height, width)

    street_type, _ = rasterize(df, geometry_col, "RW_TYPE", ["mode"], gt, height, width)

    df_highway = df[df["RW_TYPE"] == 2]

    high_way, _ = rasterize(df_highway, geometry_col, "RW_TYPE", ["count"], gt, height, width)
    high_way[high_way >= 1] = 1

    # Save raster
    bands = [mean_width, traffic_dir, mean_lanes, orientation, street_type, high_way]
    band_names = ["street_width", "street_traffic", "street_lanes", "street_orientation", "street_type", "high_way"]
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=mean_width.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def zoning_nyco_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = "geometry"

    # Create raster bounds
    scale = resolution / 111320.0  # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Encoding
    # ['C1-2', 'C1-3', 'C1-4', 'C1-5', 'C2-1', 'C2-2', 'C2-3', 'C2-4', 'C2-5']
    # ['C1', 'C2']

    encoder = LabelEncoder()
    df["overlay_level2"] = encoder.fit_transform(df["overlay_level2"])
    print(list(encoder.classes_))
    df["overlay_level1"] = encoder.fit_transform(df["overlay_level1"])
    print(list(encoder.classes_))

    # Create raster
    overlay_level2, _ = rasterize(df, geometry_col, "overlay_level2", ["mode"], gt, height, width)

    overlay_level1, _ = rasterize(df, geometry_col, "overlay_level1", ["mode"], gt, height, width)

    # Save raster
    bands = [overlay_level2, overlay_level1]
    band_names = ["overlay_level2", "overlay_level1"]
    with rasterio.open(
        savefile,
        "w",
        driver="GTiff",
        count=2,
        crs=df.crs,
        dtype=overlay_level1.dtype,
        height=height,
        width=width,
        transform=gt,
    ) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_res{resolution}")


def zoning_nysp_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = "geometry"

    # Create raster bounds
    scale = resolution / 111320.0  # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Encoding
    # ['125th', 'C', 'CL', 'EC-2', 'EC-3', 'EHC', 'ETC', 'GC', 'HP', 'HRP', 'HRW', 'HY', 'IN', 'J', 'L', 'LIC', 'MMU', 'MP', 'MX-1', 'MX-13', 'MX-14', 'MX-15', 'MX-17', 'MX-18', 'MX-23', 'MX-24', 'MX-7', 'MX-9', 'MiD', 'NA-2', 'PC', 'PI', 'SRI', 'TA', 'U', 'WCh']
    # ['125th', 'C', 'CL', 'EC', 'EHC', 'ETC', 'GC', 'HP', 'HRP', 'HRW', 'HY', 'IN', 'J', 'L', 'LIC', 'MMU', 'MP', 'MX', 'MiD', 'NA', 'PC', 'PI', 'SRI', 'TA', 'U', 'WCh']
    encoder = LabelEncoder()
    df["sd_level2"] = encoder.fit_transform(df["sd_level2"])
    print(list(encoder.classes_))
    df["sd_level1"] = encoder.fit_transform(df["sd_level1"])
    print(list(encoder.classes_))

    # Create raster
    sd_level2, _ = rasterize(df, geometry_col, "sd_level2", ["mode"], gt, height, width)

    sd_level1, _ = rasterize(df, geometry_col, "sd_level1", ["mode"], gt, height, width)

    # Save raster
    bands = [sd_level1, sd_level2]
    band_names = ["sd_level1", "sd_level2"]
    with rasterio.open(
        savefile,
        "w",
        driver="GTiff",
        count=2,
        crs=df.crs,
        dtype=sd_level1.dtype,
        height=height,
        width=width,
        transform=gt,
    ) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_res{resolution}")


def zoning_nyzd_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = "geometry"

    # Create raster bounds
    scale = resolution / 111320.0  # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Encoding
    # ['C1-7', 'C1-7A', 'C1-8', 'C1-8A', 'C1-8X', 'C1-9', 'C2-7', 'C2-7A', 'C2-8', 'C2-8A', 'C3', 'C4-1', 'C4-2', 'C4-2A', 'C4-2F', 'C4-3', 'C4-4', 'C4-4A', 'C4-4D', 'C4-5', 'C4-5D', 'C4-5X', 'C4-6', 'C4-6A', 'C4-7', 'C5-1', 'C5-1A', 'C5-2', 'C5-2.5', 'C5-3', 'C5-P', 'C6-1', 'C6-2', 'C6-2A', 'C6-3', 'C6-3D', 'C6-3X', 'C6-4', 'C6-4.5', 'C6-4M', 'C6-4X', 'C6-5', 'C6-5.5', 'C6-6', 'C6-6.5', 'C6-7', 'C6-7T', 'C8-1', 'C8-2', 'C8-3', 'C8-4', 'M1-1', 'M1-1/R7-2', 'M1-1A/R7-3', 'M1-2', 'M1-2/R5B', 'M1-2/R5D', 'M1-2/R6A', 'M1-2/R7-2', 'M1-3', 'M1-3/R7X', 'M1-3/R8', 'M1-4', 'M1-4/R6A', 'M1-4/R7-3', 'M1-4/R7A', 'M1-4/R7D', 'M1-4/R7X', 'M1-4/R8A', 'M1-4/R9', 'M1-4/R9A', 'M1-5', 'M1-5/R10', 'M1-5/R6A', 'M1-5/R7-2', 'M1-5/R7-3', 'M1-5/R8A', 'M1-5/R9', 'M1-5/R9-1', 'M1-6', 'M1-6/R10', 'M1-6/R9', 'M2-1', 'M2-2', 'M2-3', 'M2-4', 'M3-1', 'M3-2', 'PARK', 'R1-2', 'R10', 'R10A', 'R10H', 'R2', 'R2A', 'R3-1', 'R3-2', 'R3A', 'R3X', 'R4', 'R4-1', 'R4A', 'R4B', 'R5', 'R5A', 'R5B', 'R5D', 'R6', 'R6-1', 'R6A', 'R6B', 'R7-1', 'R7-2', 'R7-3', 'R7A', 'R7B', 'R7D', 'R7X', 'R8', 'R8A', 'R8B', 'R8X', 'R9', 'R9A', 'R9X']
    # ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'M1', 'M2', 'M3', 'PARK', 'R1', 'R10', 'R10A', 'R10H', 'R2', 'R2A', 'R3', 'R3A', 'R3X', 'R4', 'R4A', 'R4B', 'R5', 'R5A', 'R5B', 'R5D', 'R6', 'R6A', 'R6B', 'R7', 'R7A', 'R7B', 'R7D', 'R7X', 'R8', 'R8A', 'R8B', 'R8X', 'R9', 'R9A', 'R9X']
    # ['C', 'M', 'P', 'R']
    encoder = LabelEncoder()
    df["zonedist_level3"] = encoder.fit_transform(df["zonedist_level3"])
    print(list(encoder.classes_))
    df["zonedist_level2"] = encoder.fit_transform(df["zonedist_level2"])
    print(list(encoder.classes_))
    df["zonedist_level1"] = encoder.fit_transform(df["zonedist_level1"])
    print(list(encoder.classes_))

    # Create raster
    zonedist_level3, _ = rasterize(df, geometry_col, "zonedist_level3", ["mode"], gt, height, width)

    zonedist_level2, _ = rasterize(df, geometry_col, "zonedist_level2", ["mode"], gt, height, width)

    zonedist_level1, _ = rasterize(df, geometry_col, "zonedist_level1", ["mode"], gt, height, width)

    # Create raster for existence of R, M, C, P
    residence_df = df[df["zonedist_level1"] == 3]
    has_residence, _ = rasterize(residence_df, geometry_col, "zonedist_level1", ["count"], gt, height, width)

    manufacture_df = df[df["zonedist_level1"] == 1]
    has_manufacture, _ = rasterize(manufacture_df, geometry_col, "zonedist_level1", ["count"], gt, height, width)

    commercial_df = df[df["zonedist_level1"] == 0]
    has_commercial, _ = rasterize(commercial_df, geometry_col, "zonedist_level1", ["count"], gt, height, width)

    park_df = df[df["zonedist_level1"] == 2]
    has_park, _ = rasterize(park_df, geometry_col, "zonedist_level1", ["count"], gt, height, width)

    # Save raster
    bands = [
        zonedist_level1,
        zonedist_level2,
        zonedist_level3,
        has_residence,
        has_manufacture,
        has_commercial,
        has_park,
    ]
    band_names = [
        "zonedist_level1",
        "zonedist_level2",
        "zonedist_level3",
        "has_residence",
        "has_manufacture",
        "has_commercial",
        "has_park",
    ]
    with rasterio.open(
        savefile,
        "w",
        driver="GTiff",
        count=7,
        crs=df.crs,
        dtype=zonedist_level1.dtype,
        height=height,
        width=width,
        transform=gt,
    ) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_res{resolution}")


def resample_canopy_height(readfile, savefile, resolution):
    resolution_degree = resolution / 111035

    with rasterio.open(readfile) as src:
        # Calculate new width and height
        data = src.read(1)
        band_name = src.descriptions

        # Convert to binary (1 for values > 0, 0 otherwise)
        # Cast to float32 for accurate mean calculation by uniform_filter
        binary_data_float = (data > 0).astype(np.float32)

        block_size = int(resolution_degree / src.res[0])  # Compute scale factor
        if block_size == 0:
            print(
                f"Warning: block_size is 0 for resolution {resolution} and source resolution {src.res[0]}. Adjusting to 1."
            )
            block_size = 1  # Avoid division by zero or empty blocks if resolution is too high vs source
        print(f"Using block_size: {block_size}")

        # Use a uniform filter to calculate the mean proportion of canopy pixels in each block
        mean_canopy_proportion = uniform_filter(binary_data_float, size=block_size, mode="constant", cval=0.0)

        # Calculate the count of canopy pixels by multiplying the mean proportion by the total number of pixels in a block
        # The result will be float, e.g., 784.0
        count_of_canopy_pixels_float = mean_canopy_proportion * (block_size**2)

        # Convert counts to an appropriate integer type (e.g., uint16)
        # This array now holds the sum of canopy pixels for each large cell of the original raster
        intermediate_counts = count_of_canopy_pixels_float.astype(np.uint16)

        # Downsample the counts to the target resolution
        count_resampled = intermediate_counts[::block_size, ::block_size]

        # Define output metadata
        transform = src.transform * src.transform.scale(
            (src.width / count_resampled.shape[1]), (src.height / count_resampled.shape[0])
        )

        new_meta = src.meta.copy()

        # Determine the data type for the output file
        # Band 1 (average height) will have the dtype of the source
        # Band 2 (canopy count) will have the dtype of count_resampled (np.uint16)
        # The output file's dtype should be able to accommodate both.
        output_file_dtype = np.promote_types(src.meta["dtype"], count_resampled.dtype)

        new_meta.update(
            {
                "transform": transform,
                "width": count_resampled.shape[1],
                "height": count_resampled.shape[0],
                "count": 2,
                "dtype": output_file_dtype,  # Set the promoted dtype for the output dataset
            }
        )

        # Perform resampling
        with rasterio.open(savefile, "w", **new_meta) as dst:
            # Read source data and resample for average canopy height (Band 1)
            resampled_avg_height = src.read(
                1, out_shape=(1, count_resampled.shape[0], count_resampled.shape[1]), resampling=Resampling.average
            )
            # Ensure the array being written matches the band's expected dtype if necessary
            dst.write(resampled_avg_height.astype(output_file_dtype, copy=False), 1)

            # Write the canopy count (Band 2)
            # Ensure the array being written matches the band's expected dtype
            dst.write(count_resampled.astype(output_file_dtype, copy=False), 2)

            dst.set_band_description(1, f"{band_name[0]}_res{resolution}")
            dst.set_band_description(2, f"{band_name[0]}_count_res{resolution}")


if __name__ == "__main__":
    # Define PROJECT_ROOT_IN_CONTAINER for standalone execution context
    PROJECT_ROOT_IN_CONTAINER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Adjust if needed

    # Update READ_DIR and SAVE_DIR for standalone execution
    READ_DIR_MAIN = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/features_extracted_phase/")
    SAVE_DIR_MAIN_1x1 = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/1x1/")
    os.makedirs(READ_DIR_MAIN, exist_ok=True)
    os.makedirs(SAVE_DIR_MAIN_1x1, exist_ok=True)

    # resolution = 1000
    # readfile_path = os.path.join(READ_DIR_MAIN, "building.geojson")
    # savefile_path = os.path.join(SAVE_DIR_MAIN_1x1, f"building_res{resolution}.tif") # Corrected to .tif
    # building_tiff(readfile_path, savefile_path, resolution)

    # readfile_path = os.path.join(READ_DIR_MAIN, "street.geojson")
    # savefile_path = os.path.join(SAVE_DIR_MAIN_1x1, f"street_res{resolution}.tif") # Corrected to .tif
    # street_tiff(readfile_path, savefile_path, resolution)
    # ... and so on for other calls in __main__

    resolution = 30
    # Assuming canopy_height_res1.tif is already in SAVE_DIR_MAIN_1x1 from a previous step (e.g., preprocess)
    readfile_canopy_path = os.path.join(SAVE_DIR_MAIN_1x1, f"canopy_height_res1.tif")
    savefile_canopy_path = os.path.join(SAVE_DIR_MAIN_1x1, f"canopy_height_res{resolution}.tif")
    # Ensure the input file for resample_canopy_height exists if running this block
    if os.path.exists(readfile_canopy_path):
        resample_canopy_height(readfile_canopy_path, savefile_canopy_path, resolution)
    else:
        print(f"Input file for canopy resampling not found: {readfile_canopy_path}")

    # ... rest of the __main__ block with paths adjusted similarly
