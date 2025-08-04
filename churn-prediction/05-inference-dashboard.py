# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Inference & Dashboard Data Preparation
# MAGIC
# MAGIC This notebook loads the trained AutoML model and generates predictions for dashboard creation.
# MAGIC It prepares clean gold data that can be used for building interactive dashboards and reports.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC - **Model Loading**: Load the best AutoML model from the previous experiment
# MAGIC - **Feature Preparation**: Load and combine customer and graph features
# MAGIC - **Prediction Generation**: Generate churn predictions with probabilities
# MAGIC - **Dashboard Data**: Create clean, structured data for visualization
# MAGIC - **Data Export**: Save results to gold layer for dashboard consumption

# COMMAND ----------

# MAGIC %run ./config/environment-setup

# COMMAND ----------

from databricks import automl
from databricks.feature_store import FeatureStoreClient
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Trained Model
# MAGIC
# MAGIC We'll load the best model from the AutoML experiment. If no experiment ID is provided,
# MAGIC we'll use the most recent experiment or allow manual specification.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Configuration

# COMMAND ----------

# DBTITLE 1,Configure model loading parameters
# Model configuration - you can modify these values
model_config = {
    "experiment_name": f"telco_churn_prediction_{catalog}_{schema}",
    "model_path": None,  # Will be auto-detected if None
    "experiment_id": None,  # Will be auto-detected if None
    "use_latest": True,  # Use the latest experiment if no specific ID
    "fallback_model_path": None  # Manual model path if auto-detection fails
}

print("🔧 Model Configuration:")
for key, value in model_config.items():
    print(f"  - {key}: {value}")

# COMMAND ----------

# DBTITLE 1,Load the best model from AutoML experiment


def load_best_model():
    """Load the best model from AutoML experiment."""

    try:
        # Try to load model directly if path is provided
        if model_config["model_path"]:
            print(
                f"Loading model from specified path: {model_config['model_path']}")
            return automl.load_model(model_config["model_path"])

        # Try fallback model path if provided
        if model_config["fallback_model_path"]:
            print(
                f"Using fallback model path: {model_config['fallback_model_path']}")
            return automl.load_model(model_config["fallback_model_path"])

        # Otherwise, try to find the latest experiment
        print("Searching for AutoML experiment...")

        # Import MLflow to search experiments
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Search for experiments with our naming pattern
        experiments = client.search_experiments(
            filter_string=f"name LIKE '%{model_config['experiment_name']}%'"
        )

        if not experiments:
            print(
                f"⚠️  No experiments found with name pattern: {model_config['experiment_name']}")
            print(
                "💡 Please provide a model_path in model_config or run the AutoML experiment first.")
            raise ValueError(
                f"No experiments found with name pattern: {model_config['experiment_name']}")

        # Get the latest experiment
        latest_experiment = max(experiments, key=lambda x: x.creation_time)
        experiment_id = latest_experiment.experiment_id

        print(
            f"Found experiment: {latest_experiment.name} (ID: {experiment_id})")

        # Get the best run from this experiment - try different metrics
        try:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                # Changed from roc_auc to log_loss
                order_by=["metrics.log_loss ASC"],
                max_results=1
            )
        except:
            # Fallback to any available metric
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                max_results=1
            )

        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_id}")

        best_run = runs[0]
        model_uri = best_run.info.artifact_uri + "/model"

        print(f"Loading best model from run: {best_run.info.run_id}")
        print(f"Model URI: {model_uri}")

        return automl.load_model(model_uri)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Ensure AutoML experiment has been run successfully")
        print("   - Provide a model_path in model_config")
        print("   - Check that the experiment name pattern is correct")
        print("   - Verify you have permissions to access the experiment")
        raise


# Load the model
model = load_best_model()
print("✅ Model loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Feature Data
# MAGIC
# MAGIC Load the customer and graph features that were used for training.

# COMMAND ----------

# DBTITLE 1,Load feature data from feature store
fs = FeatureStoreClient()

# Load customer features
customer_features_table = f"{catalog}.{schema}.{feature_store['customer_features']}"
print(f"Loading customer features from: {customer_features_table}")

customer_features = fs.read_table(name=customer_features_table)
print(f"✅ Loaded customer features: {customer_features.count()} rows")

# Load graph features
graph_features_table = f"{catalog}.{schema}.{feature_store['graph_features']}"
print(f"Loading graph features from: {graph_features_table}")

graph_features = fs.read_table(name=graph_features_table)
print(f"✅ Loaded graph features: {graph_features.count()} rows")

# COMMAND ----------

# DBTITLE 1,Combine features for prediction
# Combine features (same as training)
features = customer_features.join(
    graph_features, on='customer_id', how='inner')

print(f"✅ Combined features: {features.count()} rows")
print(f"📊 Feature columns: {len(features.columns)}")

# Display sample
display(features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Predictions
# MAGIC
# MAGIC Use the loaded model to generate churn predictions and probabilities.

# COMMAND ----------

# DBTITLE 1,Generate predictions
print("🚀 Generating predictions...")

# Make predictions
predictions = model.predict(features)

print(f"✅ Generated predictions for {predictions.count()} customers")

# Display prediction schema and sample
print("📊 Prediction Schema:")
predictions.printSchema()

display(predictions.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Dashboard Data
# MAGIC
# MAGIC Create clean, structured data for dashboard consumption with additional
# MAGIC business-relevant features and risk categories.

# COMMAND ----------

# DBTITLE 1,Create dashboard-ready data
print("📊 Preparing dashboard data...")

# Create comprehensive dashboard data
dashboard_data = predictions.select(
    'customer_id',
    'prediction',
    'probability'
).join(
    features.select('customer_id', 'churn', 'tenure',
                    'monthly_charges', 'total_charges'),
    on='customer_id',
    how='inner'
)

# Add business logic and risk categories
dashboard_data = dashboard_data.withColumn(
    'churn_probability',
    F.col('probability').getItem(1)  # Probability of churning
).withColumn(
    'risk_category',
    F.when(F.col('churn_probability') >= 0.8, 'High Risk')
    .when(F.col('churn_probability') >= 0.6, 'Medium Risk')
    .when(F.col('churn_probability') >= 0.4, 'Low Risk')
    .otherwise('Very Low Risk')
).withColumn(
    'prediction_accuracy',
    F.when(F.col('prediction') == F.col('churn'), 'Correct')
    .otherwise('Incorrect')
).withColumn(
    'revenue_impact',
    F.col('monthly_charges') * 12  # Annual revenue
).withColumn(
    'tenure_category',
    F.when(F.col('tenure') >= 60, 'Long-term (>5 years)')
    .when(F.col('tenure') >= 24, 'Medium-term (2-5 years)')
    .when(F.col('tenure') >= 12, 'Short-term (1-2 years)')
    .otherwise('New (<1 year)')
)

print(f"✅ Dashboard data prepared: {dashboard_data.count()} rows")
display(dashboard_data.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality & Summary Statistics
# MAGIC
# MAGIC Generate summary statistics and data quality metrics for the dashboard.

# COMMAND ----------

# DBTITLE 1,Generate summary statistics
print("📈 Summary Statistics:")
print("=" * 50)

# Overall statistics
total_customers = dashboard_data.count()
churned_customers = dashboard_data.filter(F.col('churn') == 1).count()
predicted_churned = dashboard_data.filter(F.col('prediction') == 1).count()

print(f"👥 Total Customers: {total_customers}")
print(
    f"🔴 Actual Churned: {churned_customers} ({churned_customers/total_customers*100:.1f}%)")
print(
    f"🔮 Predicted Churned: {predicted_churned} ({predicted_churned/total_customers*100:.1f}%)")

# Risk distribution
risk_distribution = dashboard_data.groupBy('risk_category').agg(
    F.count('*').alias('customer_count'),
    F.avg('churn_probability').alias('avg_churn_probability'),
    F.avg('monthly_charges').alias('avg_monthly_charges')
).orderBy('customer_count', ascending=False)

print(f"\n🎯 Risk Category Distribution:")
display(risk_distribution)

# Accuracy metrics
correct_predictions = dashboard_data.filter(
    F.col('prediction_accuracy') == 'Correct').count()
accuracy = correct_predictions / total_customers

print(
    f"\n📊 Prediction Accuracy: {accuracy:.3f} ({correct_predictions}/{total_customers})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Gold Layer
# MAGIC
# MAGIC Save the dashboard data to the gold layer for consumption by visualization tools.

# COMMAND ----------

# DBTITLE 1,Save dashboard data to gold layer
# Create gold table name
dashboard_table_name = f"{catalog}.{schema}.{tables['gold']['dashboard_data']}"

print(f"💾 Saving dashboard data to: {dashboard_table_name}")

# Write to gold layer
dashboard_data.write.mode("overwrite").saveAsTable(dashboard_table_name)

print("✅ Dashboard data saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Dashboard Insights
# MAGIC
# MAGIC Create additional insights and aggregations for dashboard consumption.

# COMMAND ----------

# DBTITLE 1,Generate dashboard insights
print("🔍 Generating dashboard insights...")

# 1. Risk distribution by tenure
tenure_risk = dashboard_data.groupBy('tenure_category', 'risk_category').agg(
    F.count('*').alias('customer_count'),
    F.avg('churn_probability').alias('avg_churn_probability'),
    F.sum('revenue_impact').alias('total_revenue_impact')
).orderBy('tenure_category', 'customer_count', ascending=False)

print("📊 Risk Distribution by Tenure:")
display(tenure_risk)

# 2. Revenue at risk
revenue_at_risk = dashboard_data.filter(F.col('risk_category').isin(['High Risk', 'Medium Risk'])).agg(
    F.count('*').alias('customers_at_risk'),
    F.sum('revenue_impact').alias('revenue_at_risk'),
    F.avg('churn_probability').alias('avg_churn_probability')
)

print("💰 Revenue at Risk Analysis:")
display(revenue_at_risk)

# 3. Prediction confidence analysis
confidence_analysis = dashboard_data.withColumn(
    'confidence_level',
    F.when(F.col('churn_probability') >= 0.9, 'Very High Confidence')
    .when(F.col('churn_probability') >= 0.7, 'High Confidence')
    .when(F.col('churn_probability') >= 0.5, 'Medium Confidence')
    .otherwise('Low Confidence')
).groupBy('confidence_level').agg(
    F.count('*').alias('customer_count'),
    F.avg('churn_probability').alias('avg_churn_probability'),
    F.avg('monthly_charges').alias('avg_monthly_charges')
).orderBy('customer_count', ascending=False)

print("🎯 Prediction Confidence Analysis:")
display(confidence_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export for External Dashboards
# MAGIC
# MAGIC Create additional exports for external dashboard tools like Tableau, Power BI, etc.

# COMMAND ----------

# DBTITLE 1,Export data for external dashboards
print("📤 Exporting data for external dashboards...")

# Export to CSV for external tools
csv_path = f"/dbfs/FileStore/churn_dashboard_data_{catalog}_{schema}.csv"

# Convert to pandas for CSV export
dashboard_pdf = dashboard_data.toPandas()
dashboard_pdf.to_csv(csv_path, index=False)

print(f"✅ CSV exported to: {csv_path}")
print(
    f"📊 Exported {len(dashboard_pdf)} rows with {len(dashboard_pdf.columns)} columns")

# Create summary statistics for external consumption
summary_stats = {
    'total_customers': total_customers,
    'churned_customers': churned_customers,
    'churn_rate': churned_customers/total_customers,
    'predicted_churned': predicted_churned,
    'prediction_accuracy': accuracy,
    'high_risk_customers': dashboard_data.filter(F.col('risk_category') == 'High Risk').count(),
    'medium_risk_customers': dashboard_data.filter(F.col('risk_category') == 'Medium Risk').count(),
    'avg_churn_probability': dashboard_data.agg(F.avg('churn_probability')).collect()[0][0],
    'total_revenue': dashboard_data.agg(F.sum('revenue_impact')).collect()[0][0]
}

print("📈 Summary Statistics for Dashboard:")
for key, value in summary_stats.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.3f}")
    else:
        print(f"  - {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Data Schema
# MAGIC
# MAGIC Document the data schema for dashboard developers.

# COMMAND ----------

# DBTITLE 1,Document dashboard data schema
print("📋 Dashboard Data Schema:")
print("=" * 50)

schema_doc = """
## Dashboard Data Schema

### Core Fields:
- **customer_id**: Unique customer identifier
- **prediction**: Model prediction (0 = No Churn, 1 = Churn)
- **churn_probability**: Probability of churning (0.0 - 1.0)
- **churn**: Actual churn status (0 = No Churn, 1 = Churn)

### Risk Analysis:
- **risk_category**: High Risk, Medium Risk, Low Risk, Very Low Risk
- **confidence_level**: Very High, High, Medium, Low Confidence

### Business Metrics:
- **tenure**: Customer tenure in months
- **tenure_category**: New, Short-term, Medium-term, Long-term
- **monthly_charges**: Monthly charges
- **total_charges**: Total charges
- **revenue_impact**: Annual revenue (monthly_charges * 12)

### Model Performance:
- **prediction_accuracy**: Correct or Incorrect
- **probability**: Array of probabilities [P(no_churn), P(churn)]

### Dashboard Usage:
1. **Risk Monitoring**: Filter by risk_category
2. **Revenue Analysis**: Use revenue_impact for financial dashboards
3. **Tenure Analysis**: Group by tenure_category
4. **Model Performance**: Use prediction_accuracy for model monitoring
5. **Confidence Analysis**: Filter by confidence_level for high-confidence predictions
"""

print(schema_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Inference & Dashboard Data Preparation Completed**
# MAGIC
# MAGIC - **Model Loading**: Successfully loaded AutoML model
# MAGIC - **Prediction Generation**: Generated predictions for all customers
# MAGIC - **Dashboard Data**: Created comprehensive dataset with business metrics
# MAGIC - **Gold Layer**: Saved data to gold layer for dashboard consumption
# MAGIC - **External Export**: Created CSV export for external tools
# MAGIC - **Documentation**: Provided schema documentation for dashboard developers
# MAGIC
# MAGIC ### Available Data:
# MAGIC 1. **Gold Table**: `{catalog}.{schema}.{tables['gold']['dashboard_data']}`
# MAGIC 2. **CSV Export**: `/dbfs/FileStore/churn_dashboard_data_{catalog}_{schema}.csv`
# MAGIC 3. **Summary Statistics**: Available for dashboard KPIs
# MAGIC
# MAGIC ### Dashboard Recommendations:
# MAGIC - **Risk Monitoring Dashboard**: Track high-risk customers
# MAGIC - **Revenue Impact Dashboard**: Monitor revenue at risk
# MAGIC - **Model Performance Dashboard**: Track prediction accuracy
# MAGIC - **Customer Segmentation Dashboard**: Analyze by tenure and risk categories
