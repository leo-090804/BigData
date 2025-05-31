import numpy as np
from collections import deque
import yaml
import os
import rasterio
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

# Define the project root within the Docker container
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

READ_DIR = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/1x1/")
# SAVE_DIR is used for various outputs, including subfolders like '3x3' for GLCM.
# It might be better to define SAVE_DIR_BASE and then specific subdirectories as needed.
SAVE_DIR_BASE = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/") # Base save directory
os.makedirs(READ_DIR, exist_ok=True) # Ensure READ_DIR exists, though it's for reading.
os.makedirs(SAVE_DIR_BASE, exist_ok=True) # Ensure base SAVE_DIR exists

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml") # Relative to this script
with open(CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)
COORDS = config["coords"]
CRS = config["gge_engine_config"]["crs"] # Assuming gge_engine_config is correct, was satellite_config before


def check_range(array, min_val, max_val): # Renamed min, max to avoid shadowing builtins
    array[array < min_val] = min_val
    array[array > max_val] = max_val

    return array


def calculate_indices(tifffile_name, savefile_name, source, resolution):
    # tifffile_name is just the filename, e.g., "landsat_8.tiff"
    # savefile_name is relative path from SAVE_DIR_BASE, e.g., "1x1/landsat_indices.tiff"
    tiff_abs_path = os.path.join(READ_DIR, tifffile_name)
    save_abs_path = os.path.join(SAVE_DIR_BASE, savefile_name)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    
    scale = resolution / 111320.0
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    bands = {"red": 1, "blue": 2, "green": 3, "nir": 4, "swir16": 5, "swir22": 6}
    if source == "sentinel_2":
        bands["red"] = 4
        bands["blue"] = 2
        bands["green"] = 3
        bands["nir"] = 8
        bands["swir16"] = 11
        bands["swir22"] = 12

    with rasterio.open(tiff_abs_path) as dst:
        red = dst.read(bands["red"])
        blue = dst.read(bands["blue"])
        green = dst.read(bands["green"])
        nir08 = dst.read(bands["nir"])
        swir16 = dst.read(bands["swir16"])
        swir22 = dst.read(bands["swir22"])

    ndvi = check_range((nir08 - red) / (nir08 + red), -1, 1)
    evi = check_range(2.5 * (nir08 - red) / (nir08 + 6 * red - 7.5 * blue + 1 + 1e-10), -1, 1)
    savi = check_range((nir08 - red) * 1.5 / (nir08 + red + 0.5 + 1e-10), -1.5, 1.5)
    gndvi = check_range((nir08 - green) / (nir08 + green + 1e-10), -1, 1)
    arvi = check_range((nir08 - (red - (blue - red))) / (nir08 + (red - (blue - red)) + 1e-10), -1, 1)

    term = (2 * nir08 + 1) ** 2 - 8 * (nir08 - red)
    term = np.maximum(term, 0)  # Ensure no negative values inside sqrt
    msavi = check_range((2 * nir08 + 1 - np.sqrt(term)) / 2, 0, 1)

    ndwi = check_range((green - nir08) / (green + nir08), -1, 1)
    mndwi = check_range((green - swir16) / (green + swir16 + 1e-10), -1, 1)
    awei_nsh = 4 * (green - swir16) - (0.25 * nir08 + 2.75 * swir22)
    awei_sh = green + 2.5 * nir08 - 1.5 * (swir16 + swir22) - 0.25 * blue
    ndsi = check_range((green - swir16) / (green + swir16 + 1e-10), -1, 1)
    nbr = check_range((nir08 - swir22) / (nir08 + swir22 + 1e-10), -1, 1)
    si = check_range((swir16 - blue) / (swir16 + blue + 1e-10), -1, 1)
    ndbi = check_range((swir16 - nir08) / (swir16 + nir08 + 1e-10), -1, 1)
    ui = check_range((swir16 - red) / (swir16 + red + 1e-10), -1, 1)
    ibi = check_range((ndbi - (savi + mndwi)) / (ndbi + (savi + mndwi) + 1e-10), -1, 1)
    albedo = check_range(
        (0.356 * blue + 0.130 * red + 0.373 * nir08 + 0.085 * swir16 + 0.072 * swir22 - 0.018) / 1.016, 0, 1
    )

    bands = [ndvi, evi, savi, gndvi, arvi, msavi, ndwi, awei_nsh, awei_sh, ndsi, nbr, si, ndbi, ui, ibi, albedo]
    band_names = [
        "ndvi",
        "evi",
        "savi",
        "gndvi",
        "arvi",
        "msavi",
        "ndwi",
        "awei_nsh",
        "awei_sh",
        "ndsi",
        "nbr",
        "si",
        "ndbi",
        "ui",
        "ibi",
        "albedo",
    ]
    with rasterio.open(
        save_abs_path,
        "w",
        driver="GTiff",
        count=len(bands),
        crs=CRS,
        dtype=ndvi.dtype,
        height=height,
        width=width,
        transform=gt,
    ) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{source}_{band_names[i]}")


def building_street_features(building_tiff_name, street_tiff_name, savefile_name, resolution=30):
    
    building_tiff_abs_path = os.path.join(READ_DIR, building_tiff_name)
    street_tiff_abs_path = os.path.join(READ_DIR, street_tiff_name)
    save_abs_path = os.path.join(SAVE_DIR_BASE, savefile_name) # Assuming savefile_name is like "1x1/building_street_res30.tif"
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    
    with rasterio.open(building_tiff_abs_path) as dst:
        building_height = dst.read(1)
        building_area = dst.read(4)
        meta = dst.meta

    with rasterio.open(street_tiff_abs_path) as dst:
        street_width = dst.read(1)

    street_width = np.where(street_width == 0, np.nan, street_width)
    var = building_height / street_width
    var = np.where(np.isin(var, [np.inf, -np.inf, np.nan]), -1, var)

    # Calculate the building's area per pixel area (30 x 30 m^2)
    building_area_per_pixel = building_area / (resolution**2 * 10.764)

    bands = [var, building_area_per_pixel]
    band_names = ["var", "building_area_per_pixel"]
    meta["count"] = 2
    with rasterio.open(save_abs_path, "w", **meta) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_res{resolution}")


def extract_glcm(tiff_abs_path, save_abs_path, size): # tiff_abs_path and save_abs_path should be absolute
    ops = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_arr = []
    glcm_name = []

    with rasterio.open(tiff_abs_path) as dst:
        meta = dst.meta
        desc = dst.descriptions

        for k in range(dst.count):
            raster_arr = dst.read(k + 1)

            raster_scaled = ((raster_arr - raster_arr.min()) / (raster_arr.max() - raster_arr.min() + 1) * 16).astype(
                np.uint8
            )
            m, n = raster_scaled.shape

            for op in ops:
                smoothed_raster = np.zeros((m, n))
                for i in tqdm(range(m)):
                    for j in range(n):
                        neighbors = raster_scaled[
                            max(i - size, 0) : min(i + 1 + size, m), max(j - size, 0) : min(j + 1 + size, n)
                        ]
                        glcm = graycomatrix(
                            neighbors, distances=[1], angles=[np.pi / 4], levels=16, symmetric=True, normed=True
                        )
                        smoothed_raster[i, j] = graycoprops(glcm, op)[0, 0]

                glcm_arr.append(smoothed_raster)
                glcm_name.append(f"{desc[k]}_{op}_{size*2 + 1}x{size*2 + 1}")

    meta["count"] = len(glcm_arr)
    with rasterio.open(save_abs_path, "w", **meta) as dst:
        for i, band in enumerate(glcm_arr):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, glcm_name[i])


def calculate_distance(raster_arr):
    rows, cols = raster_arr.shape
    dist = np.full((rows, cols), np.inf)
    queue = deque()

    # Initialize queue with all '1' positions and set their distance to 0
    ones = np.argwhere(raster_arr > 0)
    for r, c in ones:
        dist[r, c] = 0
        queue.append((r, c))

    directions = [
        (-1, 0, 1),
        (1, 0, 1),
        (0, -1, 1),
        (0, 1, 1),  # Up, Down, Left, Right (distance = 1)
        (-1, -1, 2**0.5),
        (-1, 1, 2**0.5),
        (1, -1, 2**0.5),
        (1, 1, 2**0.5),  # Diagonal moves (distance = sqrt(2))
    ]

    # BFS to compute shortest distances
    while queue:
        r, c = queue.popleft()
        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = dist[r, c] + cost
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    queue.append((nr, nc))

    return dist


def zoning_distance(tiff_file_name, savefile_name):
    # calculate the distance of a pixel to the closest distance
    
    tiff_abs_path = os.path.join(READ_DIR, tiff_file_name)
    save_abs_path = os.path.join(SAVE_DIR_BASE, savefile_name) # Assuming savefile_name is like "1x1/zoning_res30_distance.tiff"
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    
    raster_arr = []
    with rasterio.open(tiff_abs_path) as dst:
        band_names = dst.descriptions[3:]
        meta = dst.meta
        for i in range(3, 7, 1):
            band = dst.read(i + 1)
            raster_arr.append(band)

    distance_raster_arr = []
    for i, raster in tqdm(enumerate(raster_arr), desc="Iterating bands"):
        distance_raster = calculate_distance(raster)
        distance_raster_arr.append(distance_raster)

    # save_tiff
    meta["count"] = 4
    with rasterio.open(save_abs_path, "w", **meta) as dst:
        for i, band in enumerate(distance_raster_arr):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_distance")


def smoothing_filter(raster_arr, operation, size=1):  # 3x3: size = 1, 5x5: size = 2
    m, n = raster_arr.shape
    smoothed_raster = np.zeros((m, n))

    for i in tqdm(range(m)):
        for j in range(n):
            neighbors = raster_arr[max(i - size, 0) : min(i + 1 + size, m), max(j - size, 0) : min(j + 1 + size, n)]

            if np.isnan(neighbors).all():
                smoothed_raster[i, j] = np.nan
            elif operation == "mean":
                smoothed_raster[i, j] = np.nanmean(neighbors)
            elif operation == "std_dev":
                smoothed_raster[i, j] = np.nanstd(neighbors)
            elif operation == "median":
                smoothed_raster[i, j] = np.nanmedian(neighbors)
            else:
                print("Only except mean, median, and std_dev operation.")

    return smoothed_raster


def smooth_tiff_file(tiff_abs_path, save_abs_path, operation, size=1): # tiff_abs_path and save_abs_path should be absolute
    # read file, get number of layers, name of layer
    raster_arr = []
    with rasterio.open(tiff_abs_path) as dst:
        num_bands = dst.count
        band_names = dst.descriptions
        meta = dst.meta
        for i in range(num_bands):
            band = dst.read(i + 1)
            raster_arr.append(band)

    smoothed_raster_arr = []
    for i, raster in tqdm(enumerate(raster_arr), desc="Iterating bands"):
        smoothed_raster = smoothing_filter(raster, operation, size=size)
        smoothed_raster_arr.append(smoothed_raster)

    # save_tiff
    with rasterio.open(save_abs_path, "w", **meta) as dst:
        for i, band in enumerate(smoothed_raster_arr):
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, f"{band_names[i]}_{operation}_{size*2 + 1}x{size*2 + 1}")


def perform_glcm_operations(size):
    glcm_lst = ["landsat", "building", "street", "canopy", "aod", "population", "co", "hcho", "no2", "o3", 'so2']
    
    # Define specific save directory for GLCM outputs, e.g., a 'glcm' subfolder or '3x3' as used in the original code
    glcm_save_dir = os.path.join(SAVE_DIR_BASE, f"{size*2+1}x{size*2+1}_glcm_outputs") # Example: creates a folder like '3x3_glcm_outputs'
    os.makedirs(glcm_save_dir, exist_ok=True)

    smoothing_save_dir = os.path.join(SAVE_DIR_BASE, f"{size*2+1}x{size*2+1}_smoothed_outputs") # Example for smoothed
    os.makedirs(smoothing_save_dir, exist_ok=True)


    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc=f"{size*2+1}x{size*2+1}"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path_abs = os.path.join(READ_DIR, filename)

            operation = "mean"
            # savefile_smooth_mean_abs = os.path.join(SAVE_DIR_BASE, f"3x3/{size*2+1}x{size*2+1}_{operation}_{filename}") # Original structure
            savefile_smooth_mean_abs = os.path.join(smoothing_save_dir, f"{operation}_{filename}")
            if os.path.exists(savefile_smooth_mean_abs):
                print(f"File exists: {savefile_smooth_mean_abs}")
            else:
                smooth_tiff_file(file_path_abs, savefile_smooth_mean_abs, operation, size=size)

            operation = "std_dev"
            # savefile_smooth_std_abs = os.path.join(SAVE_DIR_BASE, f"3x3/{size*2+1}x{size*2+1}_{operation}_{filename}") # Original structure
            savefile_smooth_std_abs = os.path.join(smoothing_save_dir, f"{operation}_{filename}")
            if os.path.exists(savefile_smooth_std_abs):
                print(f"File exists: {savefile_smooth_std_abs}")
            else:
                smooth_tiff_file(file_path_abs, savefile_smooth_std_abs, operation, size=size)

            if any(filename.startswith(prefix) for prefix in glcm_lst):
                # savefile_glcm_abs = os.path.join(SAVE_DIR_BASE, f"3x3/{size*2+1}x{size*2+1}_glcm_{filename}") # Original structure
                savefile_glcm_abs = os.path.join(glcm_save_dir, f"glcm_{filename}")
                if os.path.exists(savefile_glcm_abs):
                    print(f"File exists: {savefile_glcm_abs}")
                else:
                    extract_glcm(file_path_abs, savefile_glcm_abs, size=size)

