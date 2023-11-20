# Databricks notebook source
##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# It is configured and can be executed as the batch_inference_job in the batch_inference_job workflow defined under
# ``mlops_stacks/resources/batch-inference-workflow-resource.yml``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
#  * model_name (required) - The name of the model to be used in batch inference.
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# A Hive-registered Delta table containing the input features.
dbutils.widgets.text("input_table_name", "", label="Input Table Name")
# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "", label="Output Table Name")
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.main.mlops-stacks-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1,Define input and output variables

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
model_name = dbutils.widgets.get("model_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
alias = "Champion"
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

from mlflow import MlflowClient

# Get model version from alias
client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import mlflow
from pyspark.sql.functions import struct, lit, to_timestamp

from datetime import timedelta, timezone
import math
import mlflow.pyfunc
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_pickup_datetime"], F.lit(15)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_dropoff_datetime"], F.lit(30)
                )
            ),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df

def get_table_data(spark_session, input_table_path
    ):
    raw_data = spark_session.read.format("delta").load(input_table_path)
    # raw_data = spark.read.format("delta").load(input_table_path)
    return rounded_taxi_data(raw_data)


def predict_batch(
    spark_session, model_uri, input_table_name, output_table_name, model_version, ts
):
    """
    Apply the model at the specified URI for batch inference on the table with name input_table_name,
    writing results to the table with name output_table_name
    """
    mlflow.set_registry_uri("databricks-uc")
    # table = spark_session.table(input_table_name)
    table = get_table_data(spark_session, input_table_name)
    
    from databricks.feature_store import FeatureStoreClient
    
    fs_client = FeatureStoreClient()

    prediction_df = fs_client.score_batch(
        model_uri,
        table
    )
    output_df = (
        prediction_df.withColumn("prediction", prediction_df["prediction"])
        .withColumn("model_version", lit(model_version))
        .withColumn("inference_timestamp", to_timestamp(lit(ts)))
    )
    
    output_df.display()
    # Model predictions are written to the Delta table provided as input.
    # Delta is the default format in Databricks Runtime 8.0 and above.
    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)

# COMMAND ----------
# DBTITLE 1,Load model and run inference

# from predict import predict_batch

predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
dbutils.notebook.exit(output_table_name)
