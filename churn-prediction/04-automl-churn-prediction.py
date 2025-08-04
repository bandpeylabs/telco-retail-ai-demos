# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction with Databricks AutoML
# MAGIC
# MAGIC This notebook demonstrates how to use Databricks AutoML to create a churn prediction model
# MAGIC using the feature tables we created in the previous steps.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC AutoML automatically generates state-of-the-art models for classification, regression, and forecasting.
# MAGIC It provides a glass-box solution that empowers data teams without taking away control.
# MAGIC
# MAGIC ### Key Benefits:
# MAGIC - **Automated Feature Engineering**: Handles missing values, encoding, scaling
# MAGIC - **Model Selection**: Tests multiple algorithms and hyperparameters
# MAGIC - **Best Practices**: Implements ML best practices automatically
# MAGIC - **Reproducible**: Generates notebooks with full ML pipeline
# MAGIC - **Deployable**: Models can be directly deployed to production

# COMMAND ----------

# MAGIC %run ./config/environment-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Feature Data
# MAGIC
# MAGIC We'll load the customer features and graph features from the feature store
# MAGIC and combine them for the AutoML experiment.

# COMMAND ----------

from databricks import automl
from databricks.feature_store import FeatureStoreClient
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Customer Features

# COMMAND ----------

# DBTITLE 1,Load customer features from feature store
fs = FeatureStoreClient()

# Load customer features
customer_features_table = f"{catalog}.{schema}.{feature_store['customer_features']}"
print(f"Loading customer features from: {customer_features_table}")

try:
    customer_features = fs.read_table(name=customer_features_table)
    print(f"✅ Loaded customer features: {customer_features.count()} rows")
    display(customer_features.limit(5))
except Exception as e:
    print(f"❌ Error loading customer features: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Graph Features

# COMMAND ----------

# DBTITLE 1,Load graph features from feature store
# Load graph features
graph_features_table = f"{catalog}.{schema}.{feature_store['graph_features']}"
print(f"Loading graph features from: {graph_features_table}")

try:
    graph_features = fs.read_table(name=graph_features_table)
    print(f"✅ Loaded graph features: {graph_features.count()} rows")
    display(graph_features.limit(5))
except Exception as e:
    print(f"❌ Error loading graph features: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine Features

# COMMAND ----------

# DBTITLE 1,Combine customer and graph features
# Combine features for AutoML
print("Combining customer and graph features...")

features = customer_features.join(
    graph_features, on='customer_id', how='inner')

print(f"✅ Combined features: {features.count()} rows")
print(f"📊 Feature columns: {len(features.columns)}")

# Display feature summary
display(features.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Check

# COMMAND ----------

# DBTITLE 1,Check data quality and target distribution
# Check target distribution
target_counts = features.groupBy('churn').count().orderBy('churn')
display(target_counts)

# Calculate class balance
total_rows = features.count()
churn_counts = target_counts.collect()
churn_0_count = churn_counts[0]['count'] if churn_counts[0]['churn'] == 0 else churn_counts[1]['count']
churn_1_count = churn_counts[1]['count'] if churn_counts[1]['churn'] == 1 else churn_counts[0]['count']

print(f"📊 Target Distribution:")
print(
    f"  - Non-churned customers (0): {churn_0_count} ({churn_0_count/total_rows*100:.1f}%)")
print(
    f"  - Churned customers (1): {churn_1_count} ({churn_1_count/total_rows*100:.1f}%)")

# Check for missing values
print(f"\n🔍 Missing Values Check:")
for col in features.columns:
    missing_count = features.filter(f"{col} IS NULL").count()
    if missing_count > 0:
        print(f"  - {col}: {missing_count} missing values")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Experiment Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure AutoML Parameters
# MAGIC
# MAGIC We'll configure AutoML with the following settings:
# MAGIC - **Target**: `churn` (binary classification)
# MAGIC - **Primary Metric**: ROC AUC (good for imbalanced datasets)
# MAGIC - **Excluded Columns**: `customer_id` (identifier, not a feature)
# MAGIC - **Timeout**: 30 minutes (reasonable for experimentation)
# MAGIC - **Framework Exclusions**: Exclude some frameworks for faster results

# COMMAND ----------

# DBTITLE 1,Configure and run AutoML experiment
print("🚀 Starting AutoML experiment...")
print("=" * 50)

# AutoML configuration
automl_config = {
    "target_col": "churn",
    "primary_metric": "log_loss",  # Changed from roc_auc to log_loss
    "exclude_cols": ["customer_id"],
    "timeout_minutes": 30,
    "max_trials": 20,  # Limit trials for faster results
    "exclude_frameworks": ["sklearn"],  # Exclude sklearn for faster results
    "data_dir": "/dbfs/automl/churn_prediction",
    # Add timestamp to avoid conflicts
    "experiment_name": f"telco_churn_prediction_{catalog}_{schema}_{int(pd.Timestamp.now().timestamp())}"
}

print("📋 AutoML Configuration:")
for key, value in automl_config.items():
    print(f"  - {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run AutoML Experiment

# COMMAND ----------

# DBTITLE 1,Execute AutoML experiment
try:
    # Run AutoML experiment
    print("🚀 Starting AutoML experiment...")
    print(f"⏱️  Timeout: {automl_config['timeout_minutes']} minutes")
    print(f"🎯 Target: {automl_config['target_col']}")
    print(f"📊 Primary Metric: {automl_config['primary_metric']}")

    summary = automl.classify(
        features,
        **automl_config
    )

    print("✅ AutoML experiment completed successfully!")

    # Safely display best model info
    try:
        best_trial = getattr(summary, 'best_trial', None)
        if best_trial:
            model_path = getattr(best_trial, 'model_path', 'N/A')
            print(f"🏆 Best model: {model_path}")

            metrics = getattr(best_trial, 'metrics', {})
            if isinstance(metrics, dict):
                best_score = metrics.get('log_loss', 'N/A')
                print(f"📊 Best score (log_loss): {best_score}")
        else:
            print("⚠️  Best trial information not available")
    except Exception as e:
        print(f"⚠️  Could not display best model info: {e}")

except Exception as e:
    print(f"❌ AutoML experiment failed: {e}")

    # Provide helpful error messages
    if "experiment already exists" in str(e).lower():
        print("\n💡 Solution: The experiment name already exists.")
        print("   - The notebook automatically adds a timestamp to avoid conflicts")
        print("   - Try running the cell again, or manually delete the existing experiment")
    elif "roc_auc" in str(e).lower():
        print("\n💡 Solution: ROC AUC metric is not available in this AutoML version.")
        print("   - The notebook now uses 'log_loss' as the primary metric")
        print("   - This is a suitable alternative for binary classification")
    else:
        print(f"\n💡 General troubleshooting:")
        print("   - Check that your data has the target column 'churn'")
        print("   - Ensure you have sufficient permissions to create experiments")
        print("   - Try reducing timeout_minutes or max_trials for faster execution")

    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Experiment Summary

# COMMAND ----------

# DBTITLE 1,Display experiment results
print("📊 AutoML Experiment Results:")
print("=" * 50)

# Display experiment summary with safe property access
try:
    print(f"🎯 Target Column: {getattr(summary, 'target_col', 'N/A')}")
    print(f"📈 Primary Metric: {getattr(summary, 'primary_metric', 'N/A')}")
    print(f"🔢 Total Trials: {len(getattr(summary, 'trials', []))}")
    print(
        f"⏱️  Total Runtime: {getattr(summary, 'experiment_duration', 'N/A')} minutes")
except Exception as e:
    print(f"⚠️  Could not display experiment summary: {e}")

# Display best trial details with safe property access
try:
    best_trial = getattr(summary, 'best_trial', None)
    if best_trial:
        print(f"\n🏆 Best Trial Details:")
        print(f"  - Model Path: {getattr(best_trial, 'model_path', 'N/A')}")
        print(f"  - Algorithm: {getattr(best_trial, 'model_type', 'N/A')}")

        # Safely access metrics
        metrics = getattr(best_trial, 'metrics', {})
        if isinstance(metrics, dict):
            print(f"  - Log Loss: {metrics.get('log_loss', 'N/A')}")
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"  - Precision: {metrics.get('precision', 'N/A')}")
            print(f"  - Recall: {metrics.get('recall', 'N/A')}")
            print(f"  - F1 Score: {metrics.get('f1_score', 'N/A')}")
        else:
            print(f"  - Metrics: {metrics}")
    else:
        print("⚠️  No best trial information available")
except Exception as e:
    print(f"⚠️  Could not display best trial details: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance Analysis

# COMMAND ----------

# DBTITLE 1,Analyze feature importance
try:
    # Get feature importance from best model with safe access
    best_trial = getattr(summary, 'best_trial', None)
    if best_trial:
        feature_importance = getattr(best_trial, 'feature_importance', None)

        if feature_importance is not None:
            print("🔍 Feature Importance (Top 20):")
            print("=" * 40)

            # Convert to pandas for easier display
            if isinstance(feature_importance, list):
                importance_df = pd.DataFrame(feature_importance)
            elif isinstance(feature_importance, dict):
                importance_df = pd.DataFrame([feature_importance])
            else:
                importance_df = pd.DataFrame(feature_importance)

            if not importance_df.empty and 'importance' in importance_df.columns:
                importance_df = importance_df.sort_values(
                    'importance', ascending=False).head(20)

                display(importance_df)

                # Print top features
                print("\n🏆 Top 10 Most Important Features:")
                for idx, row in importance_df.head(10).iterrows():
                    feature_name = row.get('feature', f'Feature_{idx}')
                    importance_val = row.get('importance', 0)
                    print(f"  {idx+1}. {feature_name}: {importance_val:.4f}")
            else:
                print("⚠️  Feature importance data format not recognized")
        else:
            print("⚠️  Feature importance not available for this model type")
    else:
        print("⚠️  No best trial available for feature importance analysis")

except Exception as e:
    print(f"⚠️  Could not retrieve feature importance: {e}")
    print("This is normal for some model types that don't provide feature importance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Deployment Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Model Information

# COMMAND ----------

# DBTITLE 1,Save model information for deployment
# Save experiment results with safe property access
try:
    best_trial = getattr(summary, 'best_trial', None)
    metrics = getattr(best_trial, 'metrics', {}) if best_trial else {}

    experiment_results = {
        "experiment_id": getattr(summary, 'experiment_id', 'N/A'),
        "best_model_path": getattr(best_trial, 'model_path', 'N/A') if best_trial else 'N/A',
        "best_score": metrics.get('log_loss', 'N/A'),
        "best_algorithm": getattr(best_trial, 'model_type', 'N/A') if best_trial else 'N/A',
        "total_trials": len(getattr(summary, 'trials', [])),
        "experiment_duration": getattr(summary, 'experiment_duration', 'N/A'),
        # Exclude target and customer_id
        "feature_count": len(features.columns) - 2,
        "data_shape": f"{features.count()} rows x {len(features.columns)} columns"
    }
except Exception as e:
    print(f"⚠️  Could not save experiment results: {e}")
    experiment_results = {
        "feature_count": len(features.columns) - 2,
        "data_shape": f"{features.count()} rows x {len(features.columns)} columns"
    }

print("💾 Experiment Results Summary:")
for key, value in experiment_results.items():
    print(f"  - {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Deployment Code

# COMMAND ----------

# DBTITLE 1,Generate deployment-ready code
print("🚀 Deployment Code Generated:")
print("=" * 40)

deployment_code = f"""
# Model Deployment Code
# Generated from AutoML experiment: {getattr(summary, 'experiment_id', 'N/A')}

from databricks import automl
from databricks.feature_store import FeatureStoreClient

# Load the best model
model = automl.load_model("{getattr(getattr(summary, 'best_trial', None), 'model_path', 'MODEL_PATH_NOT_AVAILABLE')}")

# Feature store client
fs = FeatureStoreClient()

# Load features for prediction
customer_features = fs.read_table(name="{customer_features_table}")
graph_features = fs.read_table(name="{graph_features_table}")

# Combine features
features = customer_features.join(graph_features, on='customer_id', how='inner')

# Make predictions
predictions = model.predict(features)

# Display results
display(predictions.select('customer_id', 'prediction', 'probability'))
"""

print(deployment_code)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **AutoML Experiment Completed Successfully**
# MAGIC
# MAGIC - **Feature Engineering**: Combined customer and graph features
# MAGIC - **Model Training**: AutoML tested multiple algorithms and hyperparameters
# MAGIC - **Best Model**: Selected based on ROC AUC performance
# MAGIC - **Feature Importance**: Identified key predictive features
# MAGIC - **Deployment Ready**: Model can be deployed to production
# MAGIC
# MAGIC The AutoML experiment has created a production-ready churn prediction model
# MAGIC that leverages both traditional customer features and advanced graph-based features.
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. **Model Deployment**: Deploy the best model to production
# MAGIC 2. **Monitoring**: Set up model monitoring and drift detection
# MAGIC 3. **A/B Testing**: Compare with existing models
# MAGIC 4. **Feature Engineering**: Iterate on feature engineering based on importance analysis
