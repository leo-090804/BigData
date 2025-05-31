from pyspark.sql.functions import col, udf, when, expr, lit, degrees, atan2, split, substring, explode, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, ArrayType, StructType, StructField
from sedona.spark import SedonaContext
from pyspark.sql import SparkSession

import yaml
from tqdm import tqdm

import os
import math

from pyproj import Transformer

from google.cloud import storage
from google.oauth2 import service_account

from data_pipeline import data_loader, preprocess, features_extraction, to_tiff, features_engineering, to_tabular
from data_pipeline_test import spark_session

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# class DataPipeline:
#     def __init__(self):
#         self.spark = spark_session.create_spark_session()
#         self.data = data_loader.data_loader()
        
#     def preprocess_data(self):
#         # Preprocess the data
#         self.data = preprocess.preprocess(self.data)
        
#     def extract_features(self):
#         # Extract features from the data
#         self.data = features_extraction.extract_features(self.data)
        
#     def convert_to_tiff(self):
#         # Convert the data to TIFF format
#         self.data = to_tiff.convert_to_tiff(self.data)
        
#     def perform_feature_engineering(self):
#         # Perform feature engineering on the data
#         self.data = features_engineering.feature_engineering(self.data)
        
#     def convert_to_tabular(self):
#         # Convert the data to tabular format
#         self.data = to_tabular.convert_to_tabular(self.data)
        
spark_session_obj = spark_session.create_spark_session()
data = data_loader.data_loader()

# Preprocess the data
pop = preprocess.filter_population(spark_session_obj, config, data["population"])
canopy = preprocess.process_canopy(spark_session_obj, config, data["canopy"])