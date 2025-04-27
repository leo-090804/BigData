from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, expr, lit, degrees, atan2, split, substring
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, ArrayType, StructType, StructField
from sedona.spark import SedonaContext
from sedona.sql.types import GeometryType
import os
import yaml
import math
from pyspark.sql.functions import trim
from pyproj import Transformer


HDFS_PREFIX = "hdfs://26.3.217.119:9000"
READ_DIR = f"{HDFS_PREFIX}/climate_data/uhi_index_analytics/preprocessed/"
SAVE_DIR = f"{HDFS_PREFIX}/climate_data/uhi_index_analytics/features_extraction/"


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def create_spark_session(
    core: int = 6,
    driver_menory: str = "8g",
):
    # Create a Sedona Context using individual config calls
    builder = SedonaContext.builder()

    # Set application name
    builder = builder.config("spark.app.name", "GeoSpatialPreprocessing")

    # Add each configuration individually
    builder = builder.config(
        "spark.jars.packages",
        "org.apache.sedona:sedona-spark-shaded-3.0_2.12:1.4.1,org.datasyslab:geotools-wrapper:1.4.0-28.2",
    )
    builder = builder.master(f"local[{core}]")
    builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    builder = builder.config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions")
    builder = builder.config("spark.sql.catalog.sedona", "org.apache.sedona.sql.SpatialCatalog")
    builder = builder.config("spark.sql.catalog.sedona.options", "{}")
    builder = builder.config("spark.driver.memory", f"{driver_menory}")

    # Create and return the Sedona context
    sedona = builder.getOrCreate()

    return sedona


def extract_feature_building(spark, readfile: str, savefile: str):
    """Extract building features using PySpark and save as Parquet"""
    # Read parquet file from preprocessed directory
    preprocess_df = spark.read.parquet(f"{READ_DIR}{readfile}")

    # Select and cast columns
    extract_feature_df = preprocess_df.select(
        col("bin").cast(IntegerType()),
        col("cnstrct_yr").cast(IntegerType()),
        col("heightroof").cast(FloatType()),
        col("shape_area").cast(FloatType()),
        col("geometry"), # Assuming geometry is already in correct format from preprocessing
        col("base_bbl").cast(StringType()),
        col("mpluto_bbl").cast(StringType()),
    )

    # Configure HDFS replication factor
    spark.conf.set("spark.hadoop.dfs.replication", "1")

    # Save as Parquet in features_extraction directory
    output_path = f"{SAVE_DIR}{savefile}"
    extract_feature_df.write.format("parquet").mode("overwrite").save(output_path)
    print(f"Building features saved to {output_path}")


def convert_to_float(df, column):
    return df.withColumn(column, expr(f"CAST(NULLIF(TRIM({column}), '') AS FLOAT)"))


def feet_to_degree(df, lon_col, lat_col):
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    def transform_coords(lon, lat):
        if lon is not None and lat is not None:
            return transformer.transform(lon, lat)
        return (None, None)

    transform_udf = udf(
        transform_coords, StructType([StructField("lon", FloatType()), StructField("lat", FloatType())])
    )
    return df.withColumn("transformed", transform_udf(col(lon_col), col(lat_col)))


def street_direction(df):
    df = df.withColumn("dx", col("XTo") - col("XFrom"))
    df = df.withColumn("dy", col("YTo") - col("YFrom"))
    df = df.withColumn("angle", expr("degrees(atan2(dy, dx))"))
    df = df.withColumn("angle", when(col("angle") < 0, col("angle") + 360).otherwise(col("angle")))
    df = df.withColumn(
        "direction",
        when(
            ((22.5 <= col("angle")) & (col("angle") < 67.5)) | ((157.5 <= col("angle")) & (col("angle") < 202.5)),
            "NE-SW",
        )
        .when(
            ((67.5 <= col("angle")) & (col("angle") < 112.5)) | ((247.5 <= col("angle")) & (col("angle") < 292.5)),
            "N-S",
        )
        .when(
            ((112.5 <= col("angle")) & (col("angle") < 157.5)) | ((292.5 <= col("angle")) & (col("angle") < 337.5)),
            "NW-SE",
        )
        .otherwise("E-W"),
    )
    return df


def extract_feature_street(spark, readfile, savefile):
    # Read GeoJSON file
    df = spark.read.parquet(f"{READ_DIR}{readfile}")
    print("Input schema:")
    df.printSchema()

    # Convert string to float
    columns_to_convert = [
        "RW_TYPE",
        "Number_Travel_Lanes",
        "Number_Park_Lanes",
        "Number_Total_Lanes",
        "POSTED_SPEED",
        "BikeLane",
        "TRUCK_ROUTE_TYPE",
    ]
    for col_name in columns_to_convert:
        df = convert_to_float(df, col_name)
    df = df.withColumn("RW_TYPE", col("RW_TYPE").cast("int"))

    # Handle BIKE_TRAFDIR
    df = df.withColumn(
        "BIKE_TRAFDIR", when(expr("TRIM(BIKE_TRAFDIR) = ''"), None).otherwise(expr("TRIM(BIKE_TRAFDIR)"))
    )

    # Calculate average street width
    df = df.withColumn("street_width_avg", (col("StreetWidth_Min") + col("StreetWidth_Max")) / 2)

    # Transform coordinates
    df = feet_to_degree(df, "XFrom", "YFrom")
    df = df.withColumn("XFrom", col("transformed.lon")).withColumn("YFrom", col("transformed.lat")).drop("transformed")
    df = feet_to_degree(df, "XTo", "YTo")
    df = df.withColumn("XTo", col("transformed.lon")).withColumn("YTo", col("transformed.lat")).drop("transformed")

    # Calculate street direction
    df = street_direction(df)

    # Extract features
    df = df.select(
        [
            "OBJECTID",
            "Join_ID",
            "StreetCode",
            "Street",
            "TrafDir",
            "StreetWidth_Min",
            "StreetWidth_Max",
            "street_width_avg",
            "RW_TYPE",
            "POSTED_SPEED",
            "Number_Travel_Lanes",
            "Number_Park_Lanes",
            "Number_Total_Lanes",
            "FeatureTyp",
            "SegmentTyp",
            "BikeLane",
            "BIKE_TRAFDIR",
            "TRUCK_ROUTE_TYPE",
            "Shape__Length",
            "direction",
            "geometry",
        ]
    )

    df.show(truncate=False)

    # Save to Parquet
    output_path = f"{SAVE_DIR}{savefile}"
    df.write.format("parquet").mode("overwrite").save(output_path)
    print(f"Street features saved to {output_path}")


def extract_feature_zoning_nyzd(spark: SparkSession, readfile: str, savefile: str):
    """Extract zoning district features using PySpark and save as Parquet"""
    # Read parquet file
    df = spark.read.parquet(f"{READ_DIR}{readfile}")

    # Extract zonedist levels
    df = df.withColumnRenamed("ZONEDIST", "zonedist_level3")  # Assuming input column is ZONEDIST

    # Split to get level 2 (first part before '-')
    df = df.withColumn("zonedist_level2", split(col("zonedist_level3"), "-")[0])

    # First character for level 1
    df = df.withColumn("zonedist_level1", substring(col("zonedist_level3"), 1, 1))

    # Select relevant columns (adjust if needed)
    final_df = df.select(
        "OBJECTID", "zonedist_level1", "zonedist_level2", "zonedist_level3", "geometry", "Shape__Area", "Shape__Length"
    )  # Add other columns if necessary

    # Configure HDFS replication factor
    spark.conf.set("spark.hadoop.dfs.replication", "1")

    # Save as Parquet
    output_path = f"{SAVE_DIR}{savefile}"
    final_df.write.format("parquet").mode("overwrite").save(output_path)
    print(f"NYZD Zoning features saved to {output_path}")
    return final_df


def extract_feature_zoning_nyco(spark: SparkSession, readfile: str, savefile: str):
    """Extract commercial overlay features using PySpark and save as Parquet"""
    # Read parquet file
    df = spark.read.parquet(f"{READ_DIR}{readfile}")

    # Extract overlay levels
    df = df.withColumnRenamed("OVERLAY", "overlay_level2")  # Assuming input column is OVERLAY

    # Split to get level 1 (first part before '-')
    df = df.withColumn("overlay_level1", split(col("overlay_level2"), "-")[0])

    # Select relevant columns (adjust if needed)
    final_df = df.select(
        "OBJECTID", "overlay_level1", "overlay_level2", "geometry", "Shape__Area", "Shape__Length"
    )  # Add other columns if necessary

    # Configure HDFS replication factor
    spark.conf.set("spark.hadoop.dfs.replication", "1")

    # Save as Parquet
    output_path = f"{SAVE_DIR}{savefile}"
    final_df.write.format("parquet").mode("overwrite").save(output_path)
    print(f"NYCO Zoning features saved to {output_path}")
    return final_df


def extract_feature_zoning_nysp(spark: SparkSession, readfile: str, savefile: str):
    """Extract special purpose district features using PySpark and save as Parquet"""
    # Read parquet file
    df = spark.read.parquet(f"{READ_DIR}{readfile}")

    # Extract SD levels
    df = df.withColumnRenamed("SDLBL", "sd_level2")  # Assuming input column is SDLBL

    # Split to get level 1 (first part before '-') - handle cases where there might not be a '-'
    df = df.withColumn(
        "sd_level1", when(col("sd_level2").contains("-"), split(col("sd_level2"), "-")[0]).otherwise(col("sd_level2"))
    )  # Or set to null/other value if preferred

    # Select relevant columns (adjust if needed)
    final_df = df.select(
        "OBJECTID", "sd_level1", "sd_level2", "geometry", "Shape__Area", "Shape__Length"
    )  # Add other columns if necessary

    # Configure HDFS replication factor
    spark.conf.set("spark.hadoop.dfs.replication", "1")

    # Save as Parquet
    output_path = f"{SAVE_DIR}{savefile}"
    final_df.write.format("parquet").mode("overwrite").save(output_path)
    print(f"NYSP Zoning features saved to {output_path}")
    return final_df


if __name__ == "__main__":
    # Initialize Spark session
    os.environ["PYSPARK_PYTHON"] = os.path.join(os.environ["VIRTUAL_ENV"], "Scripts", "python.exe")
    spark = create_spark_session()

    building_input = "building.parquet"
    building_output = "building_features.parquet"
    extract_feature_building(spark, building_input, building_output)

    street_input = "street.parquet"
    street_output = "street_features.parquet"
    extract_feature_street(spark, street_input, street_output)

    zoning_nyzd_input = "nyzd.parquet"
    zoning_nyzd_output = "zoning_nyzd_features.parquet"
    extract_feature_zoning_nyzd(spark, zoning_nyzd_input, zoning_nyzd_output)

    zoning_nyco_input = "nyco.parquet"
    zoning_nyco_output = "zoning_nyco_features.parquet"
    extract_feature_zoning_nyco(spark, zoning_nyco_input, zoning_nyco_output)

    zoning_nysp_input = "nysp.parquet"
    zoning_nysp_output = "zoning_nysp_features.parquet"
    extract_feature_zoning_nysp(spark, zoning_nysp_input, zoning_nysp_output)

    spark.stop()
    print("Spark session stopped.")
