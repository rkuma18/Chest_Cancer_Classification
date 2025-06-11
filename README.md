# Chest Cancer Classification

A machine learning project for classifying chest cancer using medical imaging data. This project implements a complete ML pipeline with MLflow integration for experiment tracking and model management.

## Project Overview

This project aims to classify chest cancer from medical images using machine learning techniques. It includes a complete ML pipeline with proper configuration management, experiment tracking, and model versioning.

## Prerequisites

- Python 3.x
- DVC
- MLflow
- DagsHub account

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/rkuma18/Chest_Cancer_Classification.git
cd Chest_Cancer_Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MLflow tracking:
```bash
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password
export MLFLOW_TRACKING_URI=https://dagshub.com/rkuma18/Chest_Cancer_Classification.mlflow/
```

## Project Structure

```
├── config
│   ├── config.yaml
│   ├── params.yaml
│   └── secrets.yaml
├── src
│   ├── components
│   ├── config
│   └── pipeline
├── main.py
└── dvc.yaml
```

## Workflow

1. Update `config.yaml` with project configurations
2. Update `secrets.yaml` with sensitive information (optional)
3. Update `params.yaml` with model parameters
4. Update the entity definitions
5. Update the configuration manager in src/config
6. Update the components
7. Update the pipeline
8. Update `main.py`
9. Update `dvc.yaml`

## MLflow Integration

The project uses MLflow for experiment tracking. Example usage:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param('parameter_name', 'value')
    mlflow.log_metric('metric_name', 1)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
