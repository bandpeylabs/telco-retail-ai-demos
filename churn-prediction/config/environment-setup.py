# Databricks notebook source
# MAGIC %pip install -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('config/environment.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['main']

# COMMAND ----------

# MAGIC %md
# MAGIC ## DONE
