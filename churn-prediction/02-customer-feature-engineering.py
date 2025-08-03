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
# MAGIC ## Feature Engineering with Modern Spark 3.5.2
# MAGIC
# MAGIC Spark 3.5.2 has native support for pandas API, making it more efficient than the older `pyspark.pandas` approach.
# MAGIC We'll use the modern pandas API on Spark for feature engineering.

# COMMAND ----------

# DBTITLE 1,Define customer featurization function


def compute_customer_features(data):
    """
    Perform feature engineering on customer data using modern Spark pandas API.

    Args:
        data: Spark DataFrame with customer data

    Returns:
        Spark DataFrame with engineered features
    """
    # Convert to pandas API on Spark (modern approach for Spark 3.5.2)
    data_pandas = data.pandas_api()

    # One-Hot Encoding for categorical variables
    categorical_columns = [
        'gender', 'partner', 'dependents', 'senior_citizen',
        'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'contract', 'paperless_billing', 'payment_method'
    ]

    data_encoded = data_pandas.get_dummies(
        data_pandas,
        columns=categorical_columns,
        dtype='int64'
    )

    # Convert churn label to binary
    data_encoded['churn'] = data_encoded['churn'].map({'Yes': 1, 'No': 0})
    data_encoded = data_encoded.astype({'churn': 'int32'})

    # Clean up column names
    data_encoded.columns = [re.sub(r'[\(\)]', ' ', name).lower()
                            for name in data_encoded.columns]
    data_encoded.columns = [re.sub(r'[ -]', '_', name).lower()
                            for name in data_encoded.columns]

    # Drop missing values
    data_encoded = data_encoded.dropna()

    # Convert back to Spark DataFrame
    return data_encoded.to_spark()

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
# MAGIC - **Modern Spark 3.5.2**: Used native pandas API on Spark for better performance
# MAGIC - **Unity Catalog Integration**: Feature table created with proper governance
# MAGIC - **Configuration-Driven**: Used table names from environment config
# MAGIC - **Latest Feature Store API**: Compatible with Unity Catalog and modern Databricks
# MAGIC
# MAGIC The customer features are now available in the feature store and ready for model training.
