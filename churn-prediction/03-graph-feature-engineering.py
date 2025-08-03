# Databricks notebook source
# MAGIC %md
# MAGIC # Graph Feature Engineering
# MAGIC
# MAGIC This notebook generates graph-based features for churn prediction using Spark GraphFrames.
# MAGIC We'll analyze the customer call network to extract features that capture social influence and network centrality.

# COMMAND ----------

# MAGIC %run ./config/environment-setup

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from graphframes import *
from math import comb
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from warnings import filterwarnings
filterwarnings('ignore', 'DataFrame.sql_ctx is an internal property')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Graph Data from Silver Layer
# MAGIC
# MAGIC We'll load the vertex and edge DataFrames that were created during the exploratory data analysis phase.

# COMMAND ----------

# DBTITLE 1,Read vertex_df
vertex_df = spark.table(f"{catalog}.{schema}.{tables['silver']['vertex']}")
display(vertex_df)

# COMMAND ----------

# DBTITLE 1,Read edge_df
edge_df = spark.table(f"{catalog}.{schema}.{tables['silver']['edge']}")
display(edge_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Graph Structure
# MAGIC
# MAGIC We'll use GraphFrames to create a graph representation of the customer call network.

# COMMAND ----------

# DBTITLE 1,Creating a graph using GraphFrames
g = GraphFrame(vertex_df, edge_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Graph Metrics
# MAGIC
# MAGIC Let's start by calculating fundamental graph metrics that capture the connectivity patterns of each customer.

# COMMAND ----------

# DBTITLE 1,Degree Centrality
# MAGIC %md
# MAGIC **Degree centrality** measures the number of connections each customer has in the call network.
# MAGIC This is a fundamental measure of how connected a customer is to others.

# COMMAND ----------

degree_df = g.degrees
graph_features_df = vertex_df.alias('customer').join(degree_df, degree_df.id == vertex_df.id, 'left')\
                             .select('customer.id', 'degree')\
                             .withColumnRenamed('id', 'customer_id')\
                             .fillna(0, "degree")

display(graph_features_df.orderBy(F.col("degree").desc()))

# COMMAND ----------

# DBTITLE 1,In-degree Centrality
# MAGIC %md
# MAGIC **In-degree centrality** measures how many incoming calls a customer receives.
# MAGIC This indicates the customer's popularity or influence in the network.

# COMMAND ----------

indegree_df = g.inDegrees
graph_features_df = graph_features_df.alias('features').join(indegree_df, indegree_df.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'inDegree')\
    .fillna(0, "inDegree")\
    .withColumnRenamed("inDegree", "in_degree")
display(graph_features_df.orderBy(F.col("inDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Out-degree Centrality
# MAGIC %md
# MAGIC **Out-degree centrality** measures how many outgoing calls a customer makes.
# MAGIC This indicates the customer's level of engagement with the service.

# COMMAND ----------

outdegree_df = g.outDegrees
graph_features_df = graph_features_df.alias('features').join(outdegree_df, outdegree_df.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'outDegree')\
    .fillna(0, "outDegree")\
    .withColumnRenamed("outDegree", "out_degree")
display(graph_features_df.orderBy(F.col("outDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Degree Ratio Analysis
# MAGIC %md
# MAGIC **Degree ratios** help us understand the balance between incoming and outgoing connections.
# MAGIC This can reveal patterns in customer behavior and network influence.

# COMMAND ----------

def degreeRatio(x, d):
    if d == 0:
        return 0.0
    else:
        return x/d


degreeRatioUDF = F.udf(degreeRatio, FloatType())

graph_features_df = graph_features_df.withColumn(
    "in_degree_ratio", degreeRatioUDF(F.col("in_degree"), F.col("degree")))
graph_features_df = graph_features_df.withColumn(
    "out_degree_ratio", degreeRatioUDF(F.col("out_degree"), F.col("degree")))
display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Graph Analytics
# MAGIC
# MAGIC Now let's compute more sophisticated graph metrics that capture deeper network properties.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PageRank Centrality
# MAGIC
# MAGIC **PageRank** is a measure of node importance that considers both the number and quality of connections.
# MAGIC It was originally developed by Google founders Larry Page and Sergey Brin for ranking web pages.
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     A[Customer A] --> B[Customer B]
# MAGIC     B --> C[Customer C]
# MAGIC     C --> A
# MAGIC     D[Customer D] --> A
# MAGIC     E[Customer E] --> A
# MAGIC
# MAGIC     style A fill:#ff9999
# MAGIC     style B fill:#99ccff
# MAGIC     style C fill:#99ccff
# MAGIC     style D fill:#99ccff
# MAGIC     style E fill:#99ccff
# MAGIC ```
# MAGIC
# MAGIC In this example, Customer A has the highest PageRank because it receives links from multiple customers
# MAGIC and is part of a connected triangle, making it a central node in the network.

# COMMAND ----------

# DBTITLE 0,PageRank
# Calculating pagerank

pr_df = g.pageRank(resetProbability=0.15,
                   tol=0.01).vertices.select('id', 'pagerank')
graph_features_df = graph_features_df.alias('features').join(pr_df, pr_df.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'pagerank')

display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Triangle Count
# MAGIC
# MAGIC **Triangle count** measures how many triangles each customer participates in.
# MAGIC A triangle occurs when three customers are all connected to each other.
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     A[Customer A] --- B[Customer B]
# MAGIC     B --- C[Customer C]
# MAGIC     A --- C
# MAGIC
# MAGIC     style A fill:#ff9999
# MAGIC     style B fill:#99ccff
# MAGIC     style C fill:#99ccff
# MAGIC ```
# MAGIC
# MAGIC This triangle indicates strong social connections and potential influence patterns.

# COMMAND ----------

# DBTITLE 0,Triangle Count
# Calculating triangle count

trian_count = g.triangleCount()

graph_features_df = graph_features_df.alias('features').join(trian_count.select('id', 'count'), trian_count.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'count')\
    .withColumnRenamed("count", "trian_count")

display(graph_features_df.orderBy(F.col("trian_count").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clustering Coefficient
# MAGIC
# MAGIC **Clustering coefficient** measures how tightly connected a customer's neighbors are to each other.
# MAGIC It's calculated as: `cc(i) = (Number of triangles with corner i) / (Number of possible triangles with corner i)`
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     A[Customer A] --- B[Customer B]
# MAGIC     A --- C[Customer C]
# MAGIC     A --- D[Customer D]
# MAGIC     B --- C
# MAGIC     C --- D
# MAGIC
# MAGIC     style A fill:#ff9999
# MAGIC     style B fill:#99ccff
# MAGIC     style C fill:#99ccff
# MAGIC     style D fill:#99ccff
# MAGIC ```
# MAGIC
# MAGIC Customer A has a high clustering coefficient because its neighbors (B, C, D) are well-connected to each other.

# COMMAND ----------

# DBTITLE 0,Clustering coefficient
# Calculating clustering coefficient


def clusterCoefficient(t, e):
    if e == 0 or t == 0:
        return 0.0
    else:
        return t/comb(e, 2)


clusterCoefficientUDF = F.udf(clusterCoefficient, FloatType())

graph_features_df = graph_features_df.withColumn(
    "cc", clusterCoefficientUDF(F.col("trian_count"), F.col("degree")))
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df.orderBy(F.col("degree").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Community Detection
# MAGIC
# MAGIC **Label Propagation Algorithm (LPA)** identifies communities or clusters of customers who are more connected to each other than to the rest of the network.
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     subgraph "Community 1"
# MAGIC         A1[Customer A1] --- A2[Customer A2]
# MAGIC         A2 --- A3[Customer A3]
# MAGIC         A1 --- A3
# MAGIC     end
# MAGIC
# MAGIC     subgraph "Community 2"
# MAGIC         B1[Customer B1] --- B2[Customer B2]
# MAGIC         B2 --- B3[Customer B3]
# MAGIC         B1 --- B3
# MAGIC     end
# MAGIC
# MAGIC     A3 --- B1
# MAGIC
# MAGIC     style A1 fill:#ff9999
# MAGIC     style A2 fill:#ff9999
# MAGIC     style A3 fill:#ff9999
# MAGIC     style B1 fill:#99ccff
# MAGIC     style B2 fill:#99ccff
# MAGIC     style B3 fill:#99ccff
# MAGIC ```

# COMMAND ----------

# DBTITLE 0,Community detection
communities = g.labelPropagation(maxIter=25)
display(communities)

# COMMAND ----------

# DBTITLE 1,Calculating community statistics
comm_avg = communities.groupBy('label')\
    .agg(F.avg("monthly_charges").alias("comm_avg_monthly_charges"),
         F.avg("total_charges").alias("comm_avg_total_charges"),
         F.avg("tenure").alias("comm_avg_tenure"),
         F.count("id").alias("comm_size"))
display(comm_avg)

# COMMAND ----------

# DBTITLE 1,Community deviation analysis
communities = communities.join(comm_avg, on='label', how='left')
communities = communities.withColumn('comm_dev_avg_monthly_charges', F.col(
    'comm_avg_monthly_charges')-F.col('monthly_charges'))
communities = communities.withColumn('comm_dev_avg_total_charges', F.col(
    'comm_avg_total_charges')-F.col('total_charges'))
communities = communities.withColumn(
    'comm_dev_avg_tenure', F.col('comm_avg_tenure')-F.col('tenure'))
display(communities)

# COMMAND ----------

graph_features_df = graph_features_df.alias('features')\
    .join(communities.alias('comm'),
          communities.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'comm.comm_avg_monthly_charges', 'comm.comm_avg_total_charges', 'comm.comm_avg_tenure', 'comm.comm_size',
            'comm.comm_dev_avg_monthly_charges', 'comm.comm_dev_avg_total_charges', 'comm.comm_dev_avg_tenure')
display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neighbor Analysis
# MAGIC
# MAGIC We'll analyze the characteristics of each customer's direct neighbors to understand social influence patterns.

# COMMAND ----------

# DBTITLE 1,Calculating neighbor averages
edge_df_1 = edge_df.withColumnRenamed(
    'src', 'id').withColumnRenamed('dst', 'nbgh')
edge_df_2 = edge_df.withColumnRenamed(
    'dst', 'id').withColumnRenamed('src', 'nbgh')
und_edge_df = edge_df_1.union(edge_df_1)
und_edge_df = und_edge_df.alias('edge').join(vertex_df.select('id', 'monthly_charges', 'total_charges', 'tenure').alias('vertex'),
                                             und_edge_df.nbgh == vertex_df.id, how='left')\
    .select('edge.*', 'vertex.monthly_charges', 'vertex.total_charges', 'vertex.tenure')\
    .groupBy('id')\
    .agg(F.avg("monthly_charges").alias("nghb_avg_monthly_charges"),
         F.avg("total_charges").alias("nghb_avg_total_charges"),
         F.avg("tenure").alias("nghb_avg_tenure"))
graph_features_df = graph_features_df.alias('features')\
    .join(und_edge_df.alias('nbgh'),
          und_edge_df.id == graph_features_df.customer_id, 'left')\
    .select('features.*', 'nbgh.nghb_avg_monthly_charges', 'nbgh.nghb_avg_total_charges', 'nbgh.nghb_avg_tenure')
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Profiling Report

# COMMAND ----------

display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Integration with Unity Catalog
# MAGIC
# MAGIC We'll save the graph features to the feature store for use in model training.

# COMMAND ----------


fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Graph Feature Table

# COMMAND ----------

# Get feature table name from config
graph_feature_table_name = f"{catalog}.{schema}.{feature_store['graph_features']}"

try:
    # Drop existing table if it exists
    fs.drop_table(graph_feature_table_name)
    print(
        f"✅ Dropped existing graph feature table: {graph_feature_table_name}")
except Exception as e:
    print(f"ℹ️  No existing table to drop: {e}")

# COMMAND ----------

# Create graph feature table with Unity Catalog
graph_feature_table = fs.create_table(
    name=graph_feature_table_name,
    primary_keys=['customer_id'],
    schema=graph_features_df.schema,
    description="""
    Graph-based features for churn prediction.
    
    Features include:
    - Centrality measures (degree, in-degree, out-degree, PageRank)
    - Network structure (triangle count, clustering coefficient)
    - Community features (community averages and deviations)
    - Neighbor characteristics (average neighbor metrics)
    
    Derived from telco customer call network analysis.
    """
)

print(f"✅ Created graph feature table: {graph_feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Graph Features to Feature Store

# COMMAND ----------

# Write graph features to feature store
fs.write_table(
    df=graph_features_df,
    name=graph_feature_table_name,
    mode='overwrite'
)

print(
    f"✅ Successfully wrote {graph_features_df.count()} graph feature records to {graph_feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Graph Feature Engineering Completed Successfully**
# MAGIC
# MAGIC - **Centrality Metrics**: Degree, in-degree, out-degree, and PageRank centrality
# MAGIC - **Network Structure**: Triangle count and clustering coefficient
# MAGIC - **Community Analysis**: Label propagation for community detection
# MAGIC - **Social Influence**: Neighbor average characteristics
# MAGIC - **Unity Catalog Integration**: Graph features stored with proper governance
# MAGIC - **Configuration-Driven**: Used table names from environment config
# MAGIC
# MAGIC The graph features are now available in the feature store and ready for model training.
# MAGIC These features capture the social network dynamics that may influence customer churn behavior.
