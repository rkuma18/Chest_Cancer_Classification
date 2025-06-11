# Chest_Cancer_Classification

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9 .Update the dvc.yaml



import dagshub
dagshub.init(repo_owner='rkuma18', repo_name='Chest_Cancer_Classification', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



export MLFLOW_TRACKING_USERNAME=rkuma18
export MLFLOW_TRACKING_PASSWORD=f83980e8f1c2a11703578b88e07f97aa18e532f3
export MLFLOW_TRACKING_URI=https://dagshub.com/rkuma18/Chest_Cancer_Classification.mlflow/