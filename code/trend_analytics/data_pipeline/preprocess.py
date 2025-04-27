from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import avg


HDFS_PREFIX = "hdfs://26.3.217.119:9000"
SAVE_DIR = f"{HDFS_PREFIX}/climate_data/trend_analytics/preprocessed/"
READ_DIR = f"{HDFS_PREFIX}/climate_data/trend_analytics/"


def create_saprk_session() -> SparkSession:
    spark = SparkSession.builder \
        .appName("Climate Data Preprocessing") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark


def co2_preprocess_data(spark: SparkSession, input_path: str, output_path: str) -> None:
    co2_df = spark.read.csv(
        f"{input_path}co2_mm_mlo.csv",
        header=True,
        schema="year INT, month INT, decimal_date DOUBLE, average DOUBLE, deseasonalized DOUBLE, ndays INT, sdev DOUBLE, unc DOUBLE",
    )

    # CO2 Data Preprocessing
    clean_co2 = co2_df.filter((col("average") > 0) & (col("year") >= 1958) & (col("month").between(1, 12)))

    selected_features_co2 = clean_co2.select("year", "month", "decimal_date", "average", "deseasonalized")

    annual_co2 = selected_features_co2.groupBy("year").agg(avg("average").alias("avg_co2_ppm")).orderBy("year")
    deseasonalized_co2 = selected_features_co2.groupBy("year").agg(avg("deseasonalized").alias("avg_deseasonalized_ppm")).orderBy("year")

    # Save the preprocessed data
    clean_co2.write.parquet(f"{output_path}clean_co2.parquet", mode="overwrite")
    annual_co2.write.parquet(f"{output_path}annual_co2.parquet", mode="overwrite")
    deseasonalized_co2.write.parquet(f"{output_path}deseasonalized_co2.parquet", mode="overwrite")


if __name__ == "__main__":
    spark = create_saprk_session()

    # Preprocess CO2 data
    co2_preprocess_data(spark, READ_DIR, SAVE_DIR)

    # Stop the Spark session
    spark.stop()
    print("Preprocessing completed successfully.")
