# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Feature Engineering
# MAGIC
# MAGIC This notebook performs feature engineering on the customer data to prepare features for churn prediction.
# MAGIC We'll use modern Spark 3.5.2 capabilities and Unity Catalog for feature store operations.

# COMMAND ----------

# MAGIC %run ./config/environment-setup

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from graphframes import *
from math import comb
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Customer Data from Silver Layer

# COMMAND ----------

# Read customer data from silver layer using config-based table names
customer_df = spark.table(
    f"{catalog}.{schema}.{tables['silver']['customers']}")
display(customer_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering with Spark SQL
# MAGIC
# MAGIC We'll use Spark SQL and DataFrame operations for feature engineering, which is the most efficient and scalable approach.
# MAGIC This method works reliably across all Spark versions and provides better performance for large datasets.

# COMMAND ----------

# DBTITLE 1,Define customer featurization function
def compute_customer_features(data):
    """
    Perform feature engineering on customer data using Spark SQL and UDFs.

    Args:
        data: Spark DataFrame with customer data

    Returns:
        Spark DataFrame with engineered features
    """
    from pyspark.sql.functions import col, when, lit

    # Start with the original data
    result_df = data

    # One-Hot Encoding for categorical variables using Spark SQL
    categorical_columns = [
        'gender', 'partner', 'dependents', 'senior_citizen',
        'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'contract', 'paperless_billing', 'payment_method'
    ]

    # Create one-hot encoded columns for each categorical variable
    for column in categorical_columns:
        if column in result_df.columns:
            # Get unique values for the column
            unique_values = result_df.select(column).distinct().collect()
            unique_values = [row[column]
                             for row in unique_values if row[column] is not None]

            # Create one-hot encoded columns
            for value in unique_values:
                # Clean the value for column name
                clean_value = re.sub(r'[^a-zA-Z0-9_]', '_', str(value)).lower()
                clean_value = re.sub(r'_+', '_', clean_value).strip('_')

                # Create the one-hot encoded column
                new_column_name = f"{column}_{clean_value}"
                result_df = result_df.withColumn(
                    new_column_name,
                    when(col(column) == value, 1).otherwise(0)
                )

            # Drop the original categorical column
            result_df = result_df.drop(column)

    # Convert churn label to binary
    result_df = result_df.withColumn(
        'churn',
        when(col('churn') == 'Yes', 1).otherwise(0)
    )

    # Clean up column names
    for column in result_df.columns:
        clean_name = re.sub(r'[\(\)]', ' ', column).lower()
        clean_name = re.sub(r'[ -]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')

        if clean_name != column:
            result_df = result_df.withColumnRenamed(column, clean_name)

    # Drop missing values
    result_df = result_df.dropna()

    return result_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Feature Engineering

# COMMAND ----------


# Remove mobile_number column as it's not needed for features
customer_df_clean = customer_df.drop('mobile_number')

# Apply feature engineering
customer_features_df = compute_customer_features(customer_df_clean)

# COMMAND ----------

display(customer_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Profiling Report

# COMMAND ----------

# Display feature statistics
display(customer_features_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Integration with Unity Catalog
# MAGIC
# MAGIC We'll use the latest Feature Store API that's compatible with Unity Catalog.
# MAGIC This provides better governance, lineage tracking, and feature discovery.

# COMMAND ----------

# Initialize Feature Store client
fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Table
# MAGIC
# MAGIC We'll create a feature table in Unity Catalog with proper schema and metadata.

# COMMAND ----------

# Get feature table name from config
feature_table_name = f"{catalog}.{schema}.{feature_store['customer_features']}"

try:
    # Drop existing table if it exists
    fs.drop_table(feature_table_name)
    print(f"✅ Dropped existing feature table: {feature_table_name}")
except Exception as e:
    print(f"ℹ️  No existing table to drop: {e}")

# COMMAND ----------

# Create feature table with Unity Catalog
customer_feature_table = fs.create_table(
    name=feature_table_name,
    primary_keys=['customer_id'],
    schema=customer_features_df.schema,
    description="""
    Customer features for churn prediction.
    
    Features include:
    - One-hot encoded categorical variables (gender, partner, dependents, etc.)
    - Service-related features (phone_service, internet_service, etc.)
    - Contract and billing features (contract_type, payment_method, etc.)
    - Target variable: churn (1 = churned, 0 = retained)
    
    Derived from telco_churn_customers_silver table.
    """
)

print(f"✅ Created feature table: {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Features to Feature Store

# COMMAND ----------

# Write features to feature store
fs.write_table(
    df=customer_features_df,
    name=feature_table_name,
    mode='overwrite'
)

print(
    f"✅ Successfully wrote {customer_features_df.count()} feature records to {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Feature Store Architecture](./churn-prediction/architecture/screenshots/featurestore.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Verification

# COMMAND ----------

# Verify the feature table was created correctly
feature_table_info = fs.get_table(feature_table_name)
print(f"Feature Table Info:")
print(f"  Name: {feature_table_info.name}")
print(f"  Primary Keys: {feature_table_info.primary_keys}")
print(f"  Description: {feature_table_info.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Feature Engineering Completed Successfully**
# MAGIC
# MAGIC - **Spark SQL Approach**: Used native Spark SQL operations for optimal performance and reliability
# MAGIC - **Unity Catalog Integration**: Feature table created with proper governance
# MAGIC - **Configuration-Driven**: Used table names from environment config
# MAGIC - **Latest Feature Store API**: Compatible with Unity Catalog and modern Databricks
# MAGIC - **Scalable**: Works efficiently with large datasets across all Spark versions
# MAGIC
# MAGIC The customer features are now available in the feature store and ready for model training.
