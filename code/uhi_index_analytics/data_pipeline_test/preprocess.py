from flask import g
import ijson
import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
import os
import rasterio
from rasterio.mask import mask
import rioxarray as rxr
from pyproj import Transformer
import io

# Define the project root within the Docker container
PROJECT_ROOT_IN_CONTAINER = "/opt/airflow/project_code"

# READ_DIR = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/raw_phase/")
SAVE_DIR = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/preprocess_phase/")
os.makedirs(SAVE_DIR, exist_ok=True)

# Construct the absolute path to the config.yaml file
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)
COORDS = config["coords"]
CRS = config["satellite_config"]["params"]["crs"]

# Construct the absolute path to the gcloud_config.yaml file
GCLOUD_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "gcloud_config.yaml")
with open(GCLOUD_CONFIG_FILE_PATH, "r") as file:
    gcloud_config = yaml.safe_load(file)

# Assuming keyfile_path in gcloud_config.yaml is relative to the directory containing gcloud_config.yaml
KEYFILE_ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), gcloud_config["keyfile_path"])
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEYFILE_ABSOLUTE_PATH

def input_data_handle(readfile, type):
    if type == "building":
        filter_building(readfile, "building.geojson")
    elif type == "street":
        filter_street(readfile, "street.geojson")
    elif type == "nyco":
        filter_zoning(readfile, "nyco.geojson")
    elif type == "nysp":
        filter_zoning(readfile, "nysp.geojson")
    elif type == "nyzd":
        filter_zoning(readfile, "nyzd.geojson")
    elif type == "population":
        # Construct absolute path for savefile
        population_savefile = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/1x1/population_res1000.tiff")
        os.makedirs(os.path.dirname(population_savefile), exist_ok=True)
        filter_population_tiff(readfile, population_savefile)
    elif type == "canopy":
        # Construct absolute path for savefile
        canopy_savefile = os.path.join(PROJECT_ROOT_IN_CONTAINER, "data_pipeline_test/cache/tiff_transform_phase/1x1/canopy_height_res1.tif")
        os.makedirs(os.path.dirname(canopy_savefile), exist_ok=True)
        crop_and_reshape(readfile, canopy_savefile)


def filter_building(readfile, savefile_name):
    # Read data from json content
    features = []
    # Assuming readfile is a string or bytes containing the JSON data
    if isinstance(readfile, bytes):
        readfile = readfile.decode('utf-8') # Decode if bytes

    # with io.StringIO(readfile) as f: # Now readfile is guaranteed to be a string
    #     for feature in ijson.items(f, "item"):
    #         features.append(feature)

    with open(readfile, "r", encoding="utf-8") as f:
        for feature in ijson.items(f, "item"):
            features.append(feature)
    
    print("Load data successfully!")

    df = gpd.GeoDataFrame(features)

    # Drop missing value
    df = df.dropna(subset=["the_geom", "bin", "cnstrct_yr", "heightroof"], how="any")

    # Convert data type
    df["lstmoddate"] = pd.to_datetime(df["lstmoddate"])
    df["bin"] = df["bin"].astype("int")
    df["cnstrct_yr"] = df["cnstrct_yr"].astype("int")
    df["heightroof"] = df["heightroof"].astype("float")
    df["feat_code"] = df["feat_code"].astype("int")
    df["base_bbl"] = df["base_bbl"].astype("object")
    df["mpluto_bbl"] = df["mpluto_bbl"].astype("object")

    # Filter condition
    built_before_2021 = df["cnstrct_yr"] <= 2021
    in_mahanttan_bronx = (df["bin"] // 10**6).isin([1, 2])
    higher_12_feet = df["heightroof"] >= 12
    is_building = df["feat_code"].isin([1006, 2100])
    constructed_before_date = (df["lstmoddate"] < "2021-07-24") & (df["lststatype"].isin(["Constructed"]))
    df = df[built_before_2021 & in_mahanttan_bronx & higher_12_feet & is_building & constructed_before_date]

    # Filter the areas
    df["the_geom"] = df["the_geom"].apply(lambda x: shape(x) if x is not None else x)
    df = df.set_geometry("the_geom", crs="EPSG:4326")
    df = df.cx[COORDS[0] : COORDS[2], COORDS[1] : COORDS[3]]

    # Calculate ground area
    df = df.to_crs(epsg=2263)
    df["shape_area"] = df["the_geom"].area
    larger_400_feet = df["shape_area"] >= 400
    df = df[larger_400_feet]

    # Save data
    df = df.to_crs(epsg=4326)
    df.to_file(os.path.join(SAVE_DIR, savefile_name), driver="GeoJSON")

    return df


def filter_street(readfile, savefile_name):
    # Read data from geojson content
    # Assuming readfile is a string or bytes containing the GeoJSON data
    if isinstance(readfile, bytes):
        readfile = readfile.decode('utf-8') # Decode if bytes
        
    # df = gpd.read_file(io.StringIO(readfile)) # Now readfile is guaranteed to be a string
    # print("Load data successfully!")
    
    df = gpd.read_file(readfile)
    print("Load data successfully!")

    # Filter condition
    df["RW_TYPE"] = df["RW_TYPE"].str.strip()

    is_street = ~df["FeatureTyp"].isin(["2", "5", "7", "9", "F"])
    not_imaginary = ~df["SegmentTyp"].isin(["G", "F"])
    canyon_type = ~df["RW_TYPE"].isin(["4", "12", "14"])
    constructed = df["Status"] == "2"

    df = df[is_street & not_imaginary & canyon_type & constructed]

    # Filter feature
    feature_to_keep = [
        "OBJECTID",
        "SegmentID",
        "Join_ID",
        "StreetCode",
        "Street",
        "TrafDir",
        "StreetWidth_Min",
        "StreetWidth_Max",
        "RW_TYPE",
        "POSTED_SPEED",
        "Number_Travel_Lanes",
        "Number_Park_Lanes",
        "Number_Total_Lanes",
        "FeatureTyp",
        "SegmentTyp",
        "BikeLane",
        "BIKE_TRAFDIR",
        "XFrom",
        "YFrom",
        "XTo",
        "YTo",
        "ArcCenterX",
        "ArcCenterY",
        "NodeIDFrom",
        "NodeIDTo",
        "NodeLevelF",
        "NodeLevelT",
        "TRUCK_ROUTE_TYPE",
        "Shape__Length",
        "geometry",
    ]

    df = df[feature_to_keep]

    # Filter area
    df = df.set_geometry("geometry", crs="EPSG:4326")
    df = df.cx[COORDS[0] : COORDS[2], COORDS[1] : COORDS[3]]

    # Save data
    df.to_file(os.path.join(SAVE_DIR, savefile_name), driver="GeoJSON")

    return df


def filter_zoning(readfile, savefile_name):

    # Assuming readfile is a string or bytes containing the GeoJSON data
    # if isinstance(readfile, bytes):
    #     readfile = readfile.decode('utf-8') # Decode if bytes
        
    # df = gpd.read_file(io.StringIO(readfile)) # Now readfile is guaranteed to be a string
    # print("Load data successfully!")
    
    df = gpd.read_file(readfile)
    print("Load data successfully!")

    # Filter condition
    df = df.set_geometry("geometry", crs="EPSG:4326")
    df = df.cx[COORDS[0] : COORDS[2], COORDS[1] : COORDS[3]]

    # Save data
    df.to_file(os.path.join(SAVE_DIR, savefile_name), driver="GeoJSON")


def filter_population_tiff(pop_file, savefile_abs_path):
    # Create a Polygon (bounding box) using Shapely
    bbox_geom = box(COORDS[0], COORDS[1], COORDS[2], COORDS[3])

    # Load the shapely geometry into GeoDataFrame for masking
    gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs="EPSG:4326")

    # Open the TIFF file from bytes
    # Assuming pop_file is a bytes object containing the TIFF data
    # with rasterio.open(io.BytesIO(pop_file)) as src:
    #     gdf = gdf.to_crs(src.crs)
    #     out_image, out_transform = mask(src, gdf.geometry, crop=True)
    #     out_image = out_image.squeeze()
    # print(out_image.shape)

    with rasterio.open(pop_file) as src:
        gdf = gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_image = out_image.squeeze()
        print(out_image.shape)

        # Save the clipped image to a new file
        with rasterio.open(
            savefile_abs_path,
            "w",
            driver="GTiff",
            count=1,
            crs=gdf.crs,
            dtype=out_image.dtype,
            height=out_image.shape[0],
            width=out_image.shape[1],
            transform=out_transform,
        ) as dst:
            dst.write(out_image, 1)
            dst.set_band_description(1, "population_res1000")


def crop_and_reshape(readfile, savefile_abs_path):
    # Assuming readfile is a bytes object containing the TIFF data
    # with rasterio.open(io.BytesIO(readfile)) as src:
    #     source_crs = src.crs  # Get the source CRS of the TIFF file

    with rasterio.open(readfile) as src:
        source_crs = src.crs  # Get the source CRS of the TIFF file

    # Create a transformer to convert coordinates from target CRS to source CRS
    transformer = Transformer.from_crs(CRS, source_crs, always_xy=True)

    # Transform the bounding box coordinates
    xmin_source, ymin_source = transformer.transform(COORDS[0], COORDS[1])
    xmax_source, ymax_source = transformer.transform(COORDS[2], COORDS[3])

    # Define the bounding box in the source CRS
    bbox_source = (xmin_source, ymin_source, xmax_source, ymax_source)

    # Step 1: Open the TIFF file using rioxarray from bytes
    # rds = rxr.open_rasterio(io.BytesIO(readfile))
    rds = rxr.open_rasterio(readfile)
    rds_cropped = rds.rio.clip_box(*bbox_source)
    rds_reprojected = rds_cropped.rio.reproject(CRS)

    # Step 4: Save the cropped and reprojected raster to a new file
    rds_reprojected.rio.to_raster(savefile_abs_path)

    with rasterio.open(savefile_abs_path, "r+") as dst:
        dst.set_band_description(1, "canopy_heigth")
