from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.sql.functions import (
    explode,
    trim,
)

def filter_building(spark, config, readfile, savefile=None):
    COORDS = config["coords"]

    # Use multiline option for JSON arrays
    raw = spark.read.format("json").option("multiline", "true").load(readfile)

    print("Input schema:")
    raw.printSchema()

    raw.createOrReplaceTempView("buildings_raw")

    raw = spark.sql(
        """
        SELECT 
            bin, cnstrct_yr, heightroof, the_geom as geometry, base_bbl, mpluto_bbl,
            TO_TIMESTAMP(lstmoddate) as lstmoddate,
            feat_code, lststatype
        FROM buildings_raw
        WHERE 
            the_geom IS NOT NULL AND 
            bin IS NOT NULL AND 
            cnstrct_yr IS NOT NULL AND 
            heightroof IS NOT NULL
        """
    )

    # Convert data types
    raw = (
        raw.withColumn("bin", col("bin").cast(IntegerType()))
        .withColumn("cnstrct_yr", col("cnstrct_yr").cast(IntegerType()))
        .withColumn("heightroof", col("heightroof").cast(FloatType()))
        .withColumn("feat_code", col("feat_code").cast(IntegerType()))
        .withColumn("base_bbl", col("base_bbl").cast(StringType()))
        .withColumn("mpluto_bbl", col("mpluto_bbl").cast(StringType()))
    )

    filtered = raw.filter(
        (col("cnstrct_yr") <= 2021)
        & (col("bin") / 1000000).cast(IntegerType()).isin(1, 2)
        & (col("heightroof") >= 12)
        & (col("feat_code").isin(1006, 2100))
        & (col("lstmoddate") < "2021-07-24")
        & (col("lststatype") == "Constructed")
    )

    filtered.createOrReplaceTempView("buildings_filtered")

    # Convert the_geom GeoJSON to WKT format for spatial operations
    bbox_query = f"""
    SELECT * FROM buildings_filtered
    WHERE ST_Contains(
        ST_GeomFromWKT('POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]}, 
                              {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]}, 
                              {COORDS[0]} {COORDS[1]}))'),
        ST_GeomFromGeoJSON(to_json(geometry))
    )
    """

    bbox = spark.sql(bbox_query)

    bbox.createOrReplaceTempView("buildings_bbox")

    area_query = """
        SELECT *, 
               ST_Area(ST_Transform(ST_GeomFromGeoJSON(to_json(geometry)), 'EPSG:4326', 'EPSG:2263')) as shape_area
        FROM buildings_bbox
        """

    shape_area = spark.sql(area_query)

    shape_area.createOrReplaceTempView("buildings_shape_area")

    filter_shape = shape_area.filter(col("shape_area") >= 400)

    filter_shape.createOrReplaceTempView("buildings_filter_shape")

    final = spark.sql(
        """
        SELECT bin, cnstrct_yr, heightroof, geometry, base_bbl, mpluto_bbl,
               lstmoddate, feat_code, lststatype, shape_area
        FROM buildings_filter_shape
        """
    )

    # spark.conf.set("spark.hadoop.dfs.replication", "1")
    # final.write.format("parquet").mode("overwrite").save(f"{savefile}")
    
    return final


def filter_street(spark, config, readfile, savefile=None):
    COORDS = config["coords"]

    # Read GeoJSON file
    df = spark.read.format("json").option("multiline", "true").load(readfile)

    print("Input schema:")
    df.printSchema()

    if "features" in df.columns:
        streets_df = df.select(explode("features").alias("feature"))
        streets_df = streets_df.select(
            col("feature.id").alias("id"),
            col("feature.geometry").alias("geometry"),
            col("feature.properties.*"),  # Flatten properties
        )
    else:
        # If already at feature level
        streets_df = df

    # Register temporary view
    streets_df = streets_df.withColumn("RW_TYPE", trim(col("RW_TYPE")))

    filtered_df = streets_df.filter(
        (~col("FeatureTyp").isin("2", "5", "7", "9", "F"))
        & (~col("SegmentTyp").isin("G", "F"))
        & (~col("RW_TYPE").isin("4", "12", "14"))
        & (col("Status") == "2")
        & (col("geometry").isNotNull())  # Make sure geometry exists
    )

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

    select_df = filtered_df.select([col(c) for c in feature_to_keep if c in filtered_df.columns])

    # Convert numeric fields to proper types
    select_df = select_df.withColumn("StreetWidth_Min", col("StreetWidth_Min").cast(IntegerType()))
    select_df = select_df.withColumn("StreetWidth_Max", col("StreetWidth_Max").cast(IntegerType()))
    select_df = select_df.withColumn("Shape__Length", col("Shape__Length").cast(FloatType()))

    # Apply spatial filter using Sedona
    select_df.createOrReplaceTempView("streets_filtered")

    # Create bounding box query
    bbox_query = f"""
    SELECT * FROM streets_filtered
    WHERE ST_Contains(
        ST_GeomFromWKT('POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]}, 
                              {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]}, 
                              {COORDS[0]} {COORDS[1]}))'),
        ST_GeomFromGeoJSON(to_json(geometry))
    )
    """

    spatial_df = spark.sql(bbox_query)

    # Calculate length in meters using NYC State Plane
    spatial_df.createOrReplaceTempView("streets_bbox")

    length_query = """
    SELECT *, 
           ST_Length(ST_Transform(ST_GeomFromGeoJSON(to_json(geometry)), 'EPSG:4326', 'EPSG:2263')) as length_meters
    FROM streets_bbox
    """

    length_df = spark.sql(length_query)

    spark.conf.set("spark.hadoop.dfs.replication", "1")

    # Save to parquet for efficiency
    # length_df.write.format("parquet").mode("overwrite").save(savefile)
    # print(f"Data saved to {savefile}.parquet")
    
    return length_df


def filter_zoning(spark, config, readfile, savefile=None):
    COORDS = config["coords"]

    # Read GeoJSON file
    df = spark.read.format("json").option("multiline", "true").load(readfile)

    print("Input schema:")
    df.printSchema()

    if "features" in df.columns:
        zoning_df = df.select(explode("features").alias("feature"))
        zoning_df = zoning_df.select(
            col("feature.id").alias("id"),
            col("feature.geometry").alias("geometry"),
            col("feature.properties.*"),  # Flatten properties
        )
    else:
        # If already at feature level
        zoning_df = df

    zoning_df = zoning_df.withColumn("Shape__Area", col("Shape__Area").cast(FloatType()))
    zoning_df = zoning_df.withColumn("Shape__Length", col("Shape__Length").cast(FloatType()))

    zoning_df.createOrReplaceTempView("zoning")

    # bbox_query = f"""
    # SELECT * FROM zoning
    # WHERE ST_Contains(
    #     ST_GeomFromWKT('POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]},
    #                           {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]},
    #                           {COORDS[0]} {COORDS[1]}))'),
    #     ST_GeomFromGeoJSON(to_json(geometry))
    # )
    # """

    bbox_query = f"""
    SELECT * FROM zoning
    WHERE geometry IS NOT NULL AND
    ST_Contains(
        ST_GeomFromWKT('POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]}, 
                              {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]}, 
                              {COORDS[0]} {COORDS[1]}))'),
        ST_Point(
            (geometry.coordinates[0][0][0] + geometry.coordinates[0][2][0])/2, 
            (geometry.coordinates[0][0][1] + geometry.coordinates[0][2][1])/2
        )
    )
    """

    spatial_df = spark.sql(bbox_query)

    # spark.conf.set("spark.hadoop.dfs.replication", "1")

#     # Save to parquet for efficiency
#     # spatial_df.write.format("parquet").mode("overwrite").save(savefile)
#     # print(f"Data saved to {savefile}.parquet")

    return spatial_df


def filter_population(spark, config, input_tiff_path, output_tiff_path=None): # Added output_tiff_path=None for consistency, but it won't be used
    """
    Clips a population GeoTIFF to the configured bounding box using Sedona SQL.
    Returns the clipped raster as a DataFrame.
    """
    COORDS = config["coords"]
    # Bbox is initially in EPSG:4326
    bbox_wkt_epsg4326 = f"POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]}, {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]}, {COORDS[0]} {COORDS[1]}))"

    # Get SRID of the input raster
    srid_row = spark.sql(
        f"SELECT RS_SRID(RS_FromPath('{input_tiff_path}')) AS srid"
    ).first()
    srid_val = srid_row["srid"] if srid_row else 0

    if srid_val == 0:
        print(f"Warning: Input raster {input_tiff_path} returned SRID=0; ensure correct CRS.")

    # Perform clipping via SQL
    clipped_df = spark.sql(f"""
        SELECT RS_Clip(
            RS_FromPath('{input_tiff_path}'),
            1,
            ST_Transform(
                ST_SetSRID(ST_GeomFromWKT('{bbox_wkt_epsg4326}'), 4326),
                {srid_val}
            ),
            TRUE
        ) AS clipped_raster
    """)

    print(f"Population TIFF clipped (SRID {srid_val}) from {input_tiff_path}")
    return clipped_df


def process_canopy(spark, config, input_tiff_path, output_tiff_path=None): # Added output_tiff_path=None for consistency, but it won't be used
    """
    Crops and reprojects a canopy GeoTIFF using Sedona SQL.
    Clips to COORDS in the raster's original CRS, then reprojects to the target CRS from config.
    Returns the processed raster as a DataFrame.
    """
    COORDS = config["coords"]
    TARGET_CRS_EPSG_CODE = config["gge_engine_config"]["crs"] # e.g., "EPSG:4326"
    
    try:
        TARGET_SRID = int(TARGET_CRS_EPSG_CODE.split(":")[1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid TARGET_CRS_EPSG_CODE: {TARGET_CRS_EPSG_CODE}. Expected format 'EPSG:XXXX'. Error: {e}")

    # Bounding box WKT in the target CRS (e.g., EPSG:4326)
    bbox_wkt_target_crs = f"POLYGON(({COORDS[0]} {COORDS[1]}, {COORDS[2]} {COORDS[1]}, {COORDS[2]} {COORDS[3]}, {COORDS[0]} {COORDS[3]}, {COORDS[0]} {COORDS[1]}))"

    # Get source SRID
    row = spark.sql(
        f"SELECT RS_SRID(RS_FromPath('{input_tiff_path}')) AS srid"
    ).first()
    source_srid = row['srid'] if row else 0
    if source_srid == 0:
        print(f"Warning: Input raster {input_tiff_path} SRID=0; ensure correct source CRS.")

    # Clip in source CRS
    clipped_view = "clipped_src"
    spark.sql(f"""
        SELECT RS_Clip(
            RS_FromPath('{input_tiff_path}'),
            1,
            ST_Transform(
                ST_SetSRID(ST_GeomFromWKT('{bbox_wkt_target_crs}'), {TARGET_SRID}),
                {source_srid}
            ),
            TRUE
        ) AS raster
    """).createOrReplaceTempView(clipped_view)

    # Reproject to target CRS
    reprojected_df = spark.sql(f"""
        SELECT RS_Resample(raster, '{TARGET_CRS_EPSG_CODE}') AS reprojected_raster
        FROM {clipped_view}
    """)

    print(f"Canopy TIFF clipped and reprojected from {input_tiff_path} to {TARGET_CRS_EPSG_CODE}")
    return reprojected_df

# if __name__ == "__main__":
#     spark = create_spark_session()

#     filter_building(spark, config, f"{READ_DIR}building/building.json", f"{SAVE_DIR}building.parquet")

#     filter_street(spark, config, f"{READ_DIR}LION.geojson", f"{SAVE_DIR}street.parquet")

#     zoning_list = ["nyco", "nysp", "nyzd"]
#     for zoning_file in tqdm(zoning_list, desc="Processing zoning files"):
#         filter_zoning(spark, config, f"{READ_DIR}{zoning_file}.geojson", f"{SAVE_DIR}{zoning_file}.parquet")

#     spark.stop()
#     print("Processing completed successfully.")
