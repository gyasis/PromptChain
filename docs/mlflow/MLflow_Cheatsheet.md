# MLflow Comprehensive Cheatsheet

> **Version**: MLflow 3.x (2025)
> **Last Updated**: December 2025

## What is MLflow?

**MLflow** is an open-source platform for managing the end-to-end machine learning lifecycle. It provides:

- **Experiment Tracking**: Log parameters, metrics, and artifacts from ML experiments
- **Model Registry**: Centralized model store with versioning, staging, and deployment
- **Model Serving**: Deploy models as REST APIs locally or in production
- **Projects**: Package ML code for reproducible runs
- **GenAI/LLM Observability**: Trace and evaluate LLM applications (MLflow 3.x)

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [CLI Commands](#cli-commands)
4. [Python API - Tracking](#python-api---tracking)
5. [Python API - MlflowClient](#python-api---mlflowclient-low-level)
6. [Model Logging & Loading](#model-logging--loading)
7. [Autologging](#autologging)
8. [Model Registry](#model-registry)
9. [MLflow Projects](#mlflow-projects)
10. [Model Serving & Deployment](#model-serving--deployment)
11. [MLflow UI/Webapp](#mlflow-uiwebapp)
12. [Environment Variables](#environment-variables)
13. [Complete Example](#complete-example-end-to-end-ml-workflow)

---

## Installation

```bash
# Basic installation
pip install mlflow

# With extras for MLServer deployment
pip install mlflow[extras]

# With Databricks extras
pip install "mlflow[databricks]>=3.1"

# For specific backends
pip install mlflow psycopg2 boto3  # PostgreSQL + S3

# With gateway (for LLM routing)
pip install mlflow[gateway]
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Experiment** | Container for organizing related runs |
| **Run** | Single execution of ML code with logged params, metrics, artifacts |
| **Artifact** | Output files (models, plots, data) stored with a run |
| **Model** | Packaged ML model in MLflow format with multiple flavors |
| **Model Registry** | Centralized model store with versioning and stage management |
| **Tracking URI** | Location where MLflow stores experiment/run data |
| **Trace** | LLM call observability data (inputs, outputs, latency) |

---

## CLI Commands

### Server & UI

```bash
# Start MLflow UI (default: http://localhost:5000)
mlflow ui

# Start on custom port
mlflow ui --port 5001

# Start tracking server with SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 --port 5000

# Production server with PostgreSQL + S3
mlflow server \
  --backend-store-uri postgresql://user:password@host:5432/mlflowdb \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 --port 5000

# Server with artifact proxying (recommended for security)
mlflow server \
  --backend-store-uri postgresql://user:password@host:5432/mlflowdb \
  --artifacts-destination s3://my-bucket/artifacts \
  --host 0.0.0.0 --port 5000
```

### Experiments Management

```bash
# Create experiment
mlflow experiments create -n "my-experiment"
mlflow experiments create -n "my-experiment" -l s3://bucket/artifacts

# List experiments
mlflow experiments list

# Get experiment details
mlflow experiments get -n "my-experiment"

# Search experiments
mlflow experiments search --filter "name LIKE 'my-%'"

# Delete/Restore experiment
mlflow experiments delete -n "my-experiment"
mlflow experiments restore -n "my-experiment"

# Rename experiment
mlflow experiments rename --experiment-id 1 --new-name "renamed-experiment"
```

### Runs Management

```bash
# List runs in experiment
mlflow runs list --experiment-id 0

# Describe a run
mlflow runs describe --run-id <run-id>

# Delete a run
mlflow runs delete --run-id <run-id>
```

### Artifacts

```bash
# Download artifacts
mlflow artifacts download --run-id <run-id> --artifact-path model -d ./downloaded

# Download from artifact URI
mlflow artifacts download --artifact-uri runs:/<run-id>/model -d ./downloaded

# List artifacts
mlflow artifacts list --run-id <run-id>
```

### Running Projects

```bash
# Run from local directory
mlflow run .

# Run from Git repository
mlflow run https://github.com/mlflow/mlflow-example.git

# Run with parameters
mlflow run . -P alpha=0.5 -P l1_ratio=0.1

# Run specific entry point
mlflow run . -e train

# Run with specific experiment
mlflow run . --experiment-name "my-experiment"

# Run with conda/virtualenv/local environment
mlflow run . --env-manager conda
mlflow run . --env-manager virtualenv
mlflow run . --env-manager local
```

### Model Serving

```bash
# Serve model locally (REST API)
mlflow models serve -m runs:/<run-id>/model -p 5001

# Serve registered model
mlflow models serve -m "models:/my-model/1" -p 5001

# Serve by stage
mlflow models serve -m "models:/my-model/Production" -p 5001

# Serve by alias (MLflow 2.3+)
mlflow models serve -m "models:/my-model@champion" -p 5001

# Serve with MLServer (high performance, K8s compatible)
mlflow models serve -m runs:/<run-id>/model -p 5001 --enable-mlserver

# Build Docker image
mlflow models build-docker -m runs:/<run-id>/model -n my-model-image

# Generate prediction from file
mlflow models predict -m runs:/<run-id>/model -i input.csv -o output.csv
```

### Autolog CLI (Claude Code Integration)

```bash
# Set up MLflow tracing for Claude Code
mlflow autolog claude

# Set up in specific directory
mlflow autolog claude ~/my-project

# Check tracing status
mlflow autolog claude --status

# Disable tracing
mlflow autolog claude --disable

# Custom tracking URI
mlflow autolog claude -u file://./custom-mlruns
mlflow autolog claude -u sqlite:///mlflow.db

# Databricks backend
mlflow autolog claude -u databricks -e 123456789
```

### Garbage Collection

```bash
# Clean up deleted runs and experiments
mlflow gc --backend-store-uri sqlite:///mlflow.db
```

---

## Python API - Tracking

### Basic Setup

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
# Or use environment variable: MLFLOW_TRACKING_URI

# Get current tracking URI
uri = mlflow.get_tracking_uri()

# Set/create experiment
mlflow.set_experiment("my-experiment")

# Create experiment with artifact location
experiment_id = mlflow.create_experiment(
    "my-experiment",
    artifact_location="s3://bucket/artifacts"
)
```

### Running Experiments

```python
import mlflow

# Context manager approach (recommended)
with mlflow.start_run(run_name="my-run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")

# Manual start/end
run = mlflow.start_run()
mlflow.log_param("param1", "value1")
mlflow.end_run()

# Nested runs
with mlflow.start_run(run_name="parent"):
    mlflow.log_param("parent_param", 1)

    with mlflow.start_run(run_name="child", nested=True):
        mlflow.log_param("child_param", 2)

# Resume existing run
with mlflow.start_run(run_id="existing-run-id"):
    mlflow.log_metric("new_metric", 0.5)

# Get active run
active_run = mlflow.active_run()
if active_run:
    print(f"Run ID: {active_run.info.run_id}")
```

### Logging Parameters

```python
# Single parameter
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("model_type", "random_forest")

# Multiple parameters
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
})
```

### Logging Metrics

```python
# Single metric
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("loss", 0.05)

# Multiple metrics
mlflow.log_metrics({
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90
})

# Metrics with steps (for training curves)
for epoch in range(100):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

# Log metrics at specific timestamp
import time
mlflow.log_metric("metric", value, step=0, timestamp=int(time.time() * 1000))
```

### Logging Artifacts

```python
# Log single file
mlflow.log_artifact("model.pkl")
mlflow.log_artifact("config.yaml", artifact_path="configs")

# Log entire directory
mlflow.log_artifacts("./output_folder")
mlflow.log_artifacts("./plots", artifact_path="figures")

# Log text directly
mlflow.log_text("Model description here", "description.txt")

# Log dictionary as JSON
mlflow.log_dict({"key": "value"}, "config.json")

# Log figure (matplotlib)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
mlflow.log_figure(fig, "plot.png")

# Log image
mlflow.log_image(image_array, "image.png")
```

### Tags

```python
# Set single tag
mlflow.set_tag("model_version", "v1.0")
mlflow.set_tag("team", "data-science")

# Set multiple tags
mlflow.set_tags({
    "environment": "production",
    "framework": "pytorch"
})
```

### Querying Runs

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_ids=["0", "1"],
    filter_string="metrics.accuracy > 0.9 AND params.model_type = 'xgboost'",
    order_by=["metrics.accuracy DESC"],
    max_results=100
)

# Get specific run
run = mlflow.get_run("run-id")
print(run.info.status)
print(run.data.params)
print(run.data.metrics)

# Get experiment by name
experiment = mlflow.get_experiment_by_name("my-experiment")
```

---

## Python API - MlflowClient (Low-Level)

```python
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
import time

client = MlflowClient()

# Create experiment
exp_id = client.create_experiment("my-experiment")

# Get experiment
exp = client.get_experiment_by_name("my-experiment")

# Create run
run = client.create_run(experiment_id=exp_id)
run_id = run.info.run_id

# Log batch (efficient for many values)
timestamp = int(time.time() * 1000)
metrics = [
    Metric("accuracy", 0.95, timestamp, step=0),
    Metric("loss", 0.05, timestamp, step=0)
]
params = [
    Param("learning_rate", "0.01"),
    Param("epochs", "100")
]
tags = [
    RunTag("model_type", "neural_network")
]
client.log_batch(run_id, metrics=metrics, params=params, tags=tags)

# Set tag
client.set_tag(run_id, "status", "completed")

# Terminate run
client.set_terminated(run_id, status="FINISHED")

# Download artifacts
client.download_artifacts(run_id, "model", "./local_path")

# List artifacts
artifacts = client.list_artifacts(run_id)
```

---

## Model Logging & Loading

### Scikit-learn

```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

with mlflow.start_run():
    # Basic logging
    mlflow.sklearn.log_model(model, "model")

    # With signature and input example (recommended)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )

# Load model
loaded_model = mlflow.sklearn.load_model("runs:/<run-id>/model")
predictions = loaded_model.predict(X_test)

# Load as generic pyfunc
pyfunc_model = mlflow.pyfunc.load_model("runs:/<run-id>/model")
```

### PyTorch

```python
import mlflow.pytorch
import torch

model = MyPyTorchModel()
# Training...

with mlflow.start_run():
    mlflow.pytorch.log_model(
        model,
        "model",
        signature=signature,
        input_example=sample_input
    )

# Load model
loaded_model = mlflow.pytorch.load_model("runs:/<run-id>/model")
```

### TensorFlow/Keras

```python
import mlflow.tensorflow
import mlflow.keras

# For Keras models
with mlflow.start_run():
    mlflow.keras.log_model(keras_model, "model")

# For TensorFlow SavedModel
with mlflow.start_run():
    mlflow.tensorflow.log_model(
        tf_saved_model_dir="./saved_model",
        artifact_path="model"
    )

# Load
loaded_model = mlflow.keras.load_model("runs:/<run-id>/model")
```

### Custom PyFunc Model

```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load artifacts
        import pickle
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log custom model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModel(),
        artifacts={"model_path": "model.pkl"},
        conda_env="conda.yaml"
    )

# Load and use
loaded = mlflow.pyfunc.load_model("runs:/<run-id>/model")
predictions = loaded.predict(data)
```

---

## Autologging

### Enable All Frameworks

```python
import mlflow

# Enable for all supported frameworks
mlflow.autolog()

# With configuration
mlflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    silent=False
)
```

### Framework-Specific

```python
# Scikit-learn
mlflow.sklearn.autolog()
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    max_tuning_runs=5  # For GridSearchCV
)

# PyTorch (with Lightning)
mlflow.pytorch.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()

# XGBoost
mlflow.xgboost.autolog()

# LightGBM
mlflow.lightgbm.autolog()

# Spark
mlflow.spark.autolog()

# LangChain (tracing)
mlflow.langchain.autolog()

# DSPy (tracing + optimization)
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces=True,
    log_traces_from_compile=False
)

# OpenAI
mlflow.openai.autolog()

# Transformers/Hugging Face
mlflow.transformers.autolog()
```

### Disable Autologging

```python
# Disable all
mlflow.autolog(disable=True)

# Disable specific
mlflow.sklearn.autolog(disable=True)
```

---

## Model Registry

### Register Models

```python
import mlflow

# Option 1: Register during logging
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="my-model"
    )

# Option 2: Register existing model
result = mlflow.register_model(
    model_uri="runs:/<run-id>/model",
    name="my-model"
)
print(f"Version: {result.version}")
```

### Manage Model Versions

```python
from mlflow import MlflowClient

client = MlflowClient()

# List registered models
for rm in client.search_registered_models():
    print(rm.name)

# Get specific model
model = client.get_registered_model("my-model")

# Get model version
version = client.get_model_version("my-model", "1")

# Update model description
client.update_registered_model(
    name="my-model",
    description="Production classification model"
)

# Update version description
client.update_model_version(
    name="my-model",
    version="1",
    description="Initial version with 95% accuracy"
)

# Search model versions
versions = client.search_model_versions("name='my-model'")
```

### Stage Transitions

```python
from mlflow import MlflowClient

client = MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="my-model",
    version="1",
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="my-model",
    version="2",
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="my-model",
    version="1",
    stage="Archived"
)

# Available stages: None, Staging, Production, Archived
```

### Load Models by Stage/Version/Alias

```python
import mlflow.pyfunc

# Load latest version
model = mlflow.pyfunc.load_model("models:/my-model/latest")

# Load specific version
model = mlflow.pyfunc.load_model("models:/my-model/1")

# Load by stage
model = mlflow.pyfunc.load_model("models:/my-model/Production")
model = mlflow.pyfunc.load_model("models:/my-model/Staging")

# Load by alias (MLflow 2.3+)
model = mlflow.pyfunc.load_model("models:/my-model@champion")
```

### Model Aliases (MLflow 2.3+)

```python
from mlflow import MlflowClient

client = MlflowClient()

# Set alias
client.set_registered_model_alias("my-model", "champion", "2")
client.set_registered_model_alias("my-model", "challenger", "3")

# Get alias
alias_info = client.get_model_version_by_alias("my-model", "champion")

# Delete alias
client.delete_registered_model_alias("my-model", "champion")
```

---

## MLflow Projects

### MLproject File Structure

```yaml
# MLproject
name: my_project

# Environment options (choose one)
conda_env: conda.yaml
# OR
python_env: python_env.yaml
# OR
docker_env:
  image: my-image:latest
  volumes: ["/local/path:/container/path"]
  environment: ["ENV_VAR=value"]

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.01}
      epochs: {type: int, default: 100}
      data_path: {type: path, default: "./data"}
    command: "python train.py --lr {learning_rate} --epochs {epochs} --data {data_path}"

  validate:
    parameters:
      model_path: path
    command: "python validate.py --model {model_path}"
```

### conda.yaml Example

```yaml
name: my_project_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - scikit-learn=1.2.0
  - pandas=2.0.0
  - pip:
    - mlflow>=2.0
    - xgboost>=1.7.0
```

### Running Projects Programmatically

```python
import mlflow

# Run local project
mlflow.projects.run(
    uri=".",
    entry_point="main",
    parameters={"learning_rate": 0.01, "epochs": 50},
    experiment_name="my-experiment"
)

# Run from Git
mlflow.projects.run(
    uri="https://github.com/mlflow/mlflow-example.git",
    parameters={"alpha": 0.5}
)

# Run with specific environment
mlflow.projects.run(
    uri=".",
    env_manager="conda",  # or "virtualenv", "local"
    parameters={"alpha": 0.5}
)
```

---

## Model Serving & Deployment

### Local REST API

```bash
# Start server
mlflow models serve -m runs:/<run-id>/model -p 5001 --no-conda

# Make predictions
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["col1", "col2"], "data": [[1, 2], [3, 4]]}}'

# DataFrame records format
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"col1": 1, "col2": 2}]}'

# Instances format (for TensorFlow)
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1, 2], [3, 4]]}'
```

### Docker Deployment

```bash
# Build Docker image
mlflow models build-docker -m runs:/<run-id>/model -n my-model:v1

# Run container
docker run -p 5001:8080 my-model:v1

# Build with MLServer (high performance)
mlflow models build-docker -m runs:/<run-id>/model -n my-model:v1 --enable-mlserver
```

### Kubernetes with KServe

```yaml
# inferenceservice.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
spec:
  predictor:
    model:
      modelFormat:
        name: mlflow
      storageUri: s3://bucket/path/to/model
```

```bash
kubectl apply -f inferenceservice.yaml
```

### Cloud Deployments

```bash
# AWS SageMaker
mlflow sagemaker deploy -a my-app-name -m runs:/<run-id>/model \
  --region-name us-east-1 \
  --mode create \
  --instance-type ml.m5.large \
  --instance-count 1

# Azure ML
mlflow azureml deploy -m runs:/<run-id>/model \
  -w my-workspace \
  -s my-subscription \
  -r my-resource-group

# Ray Serve
mlflow deployments create -t ray-serve -m models:/MyModel/1 --name iris:v1
```

---

## MLflow UI/Webapp

### Features

| View | Features |
|------|----------|
| **Experiments** | List, filter, compare experiments |
| **Runs** | View all runs, sort by metrics/params, filter, compare side-by-side |
| **Run Detail** | Parameters table, metrics charts, artifacts browser, tags |
| **Model Registry** | List models, version history, stage transitions, lineage |
| **Traces** | LLM call traces, side-by-side comparison, full-text search |

### Search Syntax

```sql
-- Metrics
metrics.accuracy > 0.9
metrics.loss < 0.1

-- Parameters
params.model_type = "xgboost"
params.learning_rate < 0.01

-- Tags
tags.environment = "production"

-- Attributes
attributes.status = "FINISHED"
attributes.run_name LIKE "train%"

-- Combine
metrics.accuracy > 0.9 AND params.model_type = "xgboost"
```

### Accessing UI

```bash
# Local UI
mlflow ui  # http://localhost:5000

# Remote server
mlflow server --host 0.0.0.0 --port 5000
# Access via http://<server-ip>:5000
```

---

## Environment Variables

### Core Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | Tracking server URI | `http://localhost:5000` |
| `MLFLOW_TRACKING_USERNAME` | Basic auth username | `admin` |
| `MLFLOW_TRACKING_PASSWORD` | Basic auth password | `password` |
| `MLFLOW_EXPERIMENT_NAME` | Default experiment | `my-experiment` |
| `MLFLOW_EXPERIMENT_ID` | Default experiment ID | `0` |
| `MLFLOW_RUN_ID` | Active run ID | `abc123` |

### Artifact Storage

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_S3_ENDPOINT_URL` | S3-compatible endpoint | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `wJalrXUtnFEMI...` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure connection | `DefaultEndpoints...` |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS credentials path | `/path/to/creds.json` |

### Database Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE` | Connection pool size | `10` |
| `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW` | Max overflow connections | `20` |
| `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE` | Connection recycle (seconds) | `3600` |

### Production Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING` | Async trace logging | `true` |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` | HTTP timeout (seconds) | `120` |
| `MLFLOW_LOGGING_LEVEL` | Logging level | `DEBUG` |
| `MLFLOW_TRUNCATE_LONG_VALUES` | Truncate long values | `True` |

---

## Complete Example: End-to-End ML Workflow

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.models import infer_signature
from mlflow import MlflowClient

# Configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# Load data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train with MLflow tracking
with mlflow.start_run(run_name="random-forest-v1"):
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    # Log model with signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:3],
        registered_model_name="iris-classifier"
    )

    # Log additional artifacts
    mlflow.log_text(str(iris.feature_names), "feature_names.txt")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# Transition to production
client = MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",
    stage="Production"
)

# Set alias for easier reference
client.set_registered_model_alias("iris-classifier", "champion", "1")

# Load and use production model
production_model = mlflow.pyfunc.load_model("models:/iris-classifier@champion")
new_predictions = production_model.predict(X_test[:5])
print(f"Predictions: {new_predictions}")
```

---

## Quick Reference Card

### Most Common Commands

```bash
# Start UI
mlflow ui

# Start server
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0

# Serve model
mlflow models serve -m "models:/my-model/Production" -p 5001

# Run project
mlflow run . -P param=value
```

### Most Common Python

```python
import mlflow

# Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

# Track
with mlflow.start_run():
    mlflow.log_params({"key": "value"})
    mlflow.log_metrics({"accuracy": 0.95})
    mlflow.sklearn.log_model(model, "model")

# Autolog
mlflow.autolog()

# Load model
model = mlflow.pyfunc.load_model("models:/my-model/Production")
```

---

## Sources

- [MLflow Official Documentation](https://mlflow.org/docs/latest/)
- [MLflow CLI Reference](https://mlflow.org/docs/latest/cli.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/ml/deployment/)
- [MLflow Autologging](https://mlflow.org/docs/latest/tracking/autolog.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
