# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark


# COMMAND ----------

# Install pymongo
%pip install pymongo


# COMMAND ----------

from pymongo import MongoClient
import json
import urllib.parse

# Load credentials from JSON file (ensure this file is correctly placed)
with open('/dbfs/FileStore/credentials_mongodb.json') as f:
    login = json.load(f)

# Assign credentials to variables
username = login['username']
password = urllib.parse.quote(login['password'])  # Ensure the password is URL encoded
host = login['host']

# Construct the MongoDB connection string
url = f"mongodb+srv://{username}:{password}@{host}/?tls=true&retryWrites=true&w=majority"

# Connect to MongoDB
try:
    client = MongoClient(url)
    client.server_info()  # Forces a call and checks connectivity
    print("Connected to MongoDB successfully!")
except Exception as e:
    print("Error connecting to MongoDB:", e)

