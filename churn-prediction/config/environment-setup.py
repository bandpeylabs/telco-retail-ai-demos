# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC
# MAGIC This notebook sets up the environment for the Telco Churn Prediction project.
# MAGIC It creates the necessary catalog and schema, and sets up global variables.

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

with open('config/environment.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['main']
tables = config['tables']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Global Variables

# COMMAND ----------

# Extract configuration values
catalog = config['catalog']
schema = config['schema']
bronze_layer = config['bronze']
silver_layer = config.get('silver', 'silver')
gold_layer = config.get('gold', 'gold')

# Set global variables that will be used across all notebooks
globals()['catalog'] = catalog
globals()['schema'] = schema
globals()['db_name'] = schema  # For backward compatibility
globals()['bronze_layer'] = bronze_layer
globals()['silver_layer'] = silver_layer
globals()['gold_layer'] = gold_layer

# Load table configurations
globals()['tables'] = config.get('tables', {})
globals()['feature_store'] = config.get('feature_store', {})

print(f"Configuration loaded:")
print(f"  Catalog: {catalog}")
print(f"  Schema: {schema}")
print(f"  Bronze Layer: {bronze_layer}")
print(f"  Silver Layer: {silver_layer}")
print(f"  Gold Layer: {gold_layer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema

# COMMAND ----------

# Create catalog if it doesn't exist
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    print(f"✅ Catalog '{catalog}' created/verified successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not create catalog '{catalog}': {str(e)}")

# COMMAND ----------

# Create schema if it doesn't exist
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    print(f"✅ Schema '{catalog}.{schema}' created/verified successfully")
except Exception as e:
    print(
        f"⚠️  Warning: Could not create schema '{catalog}.{schema}': {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Current Catalog and Schema

# COMMAND ----------

# Set the current catalog and schema for the session
try:
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
    print(f"✅ Current catalog set to: {catalog}")
    print(f"✅ Current schema set to: {schema}")
except Exception as e:
    print(f"⚠️  Warning: Could not set current catalog/schema: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# Verify the current catalog and schema
try:
    current_catalog = spark.sql("SELECT current_catalog() as catalog").collect()[
        0]['catalog']
    current_schema = spark.sql("SELECT current_schema() as schema").collect()[
        0]['schema']

    print(f"✅ Verification successful:")
    print(f"  Current Catalog: {current_catalog}")
    print(f"  Current Schema: {current_schema}")

    if current_catalog == catalog and current_schema == schema:
        print("✅ Environment setup completed successfully!")
    else:
        print("⚠️  Warning: Current catalog/schema doesn't match expected values")

except Exception as e:
    print(f"⚠️  Warning: Could not verify current catalog/schema: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Variables Available
# MAGIC
# MAGIC The following global variables are now available for use in subsequent cells:
# MAGIC - `catalog`: The catalog name (e.g., "demos")
# MAGIC - `schema`: The schema name (e.g., "telco")
# MAGIC - `db_name`: Alias for schema (for backward compatibility)
# MAGIC - `bronze_layer`: The bronze layer name (e.g., "bronze")
# MAGIC - `silver_layer`: The silver layer name (e.g., "silver")
# MAGIC - `gold_layer`: The gold layer name (e.g., "gold")
# MAGIC - `tables`: Dictionary containing table names for each layer
# MAGIC - `feature_store`: Dictionary containing feature store table names
# MAGIC
# MAGIC You can use these variables in your notebooks like:
# MAGIC ```python
# MAGIC # Example usage
# MAGIC table_name = f"{catalog}.{schema}.your_table_name"
# MAGIC spark.table(table_name)
# MAGIC
# MAGIC # Access table names from config
# MAGIC customer_table = tables['bronze']['customers']
# MAGIC full_table_name = f"{catalog}.{schema}.{customer_table}"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## DONE
# MAGIC
# MAGIC Environment setup completed. You can now proceed with your data processing workflows.
