# Heart Disease MLOps Assignment

This project implements an end-to-end MLOps workflow for heart disease prediction using the UCI Heart Disease dataset. The solution covers data preparation, exploratory analysis, model training, experiment tracking, packaging, API serving, CI/CD, monitoring, and deployment.

## Problem Statement

Build a machine learning classifier to predict the risk of heart disease from patient health data and prepare the solution as a cloud-ready, monitored API.

## Project Structure

- `data/raw/` - original dataset files
- `data/processed/` - cleaned and prepared datasets
- `src/data/` - dataset loading and inspection scripts
- `src/features/` - cleaning, target preparation, preprocessing, and EDA scripts
- `src/models/` - training, tuning, tracking, and packaging scripts
- `src/api/` - FastAPI application and monitoring demo
- `tests/` - unit tests
- `.github/workflows/` - GitHub Actions CI pipeline
- `k8s/` - Kubernetes deployment manifests and guide
- `artifacts/` - generated plots, summaries, logs, screenshots, and reports
- `models/` - final packaged model and metadata

## Dataset

This project uses the processed Cleveland subset of the UCI Heart Disease dataset.

Required dataset file:

- `data/raw/processed.cleveland.data`

Optional reference file:

- `data/raw/heart-disease.names`

## Environment Setup

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Execution Order

Run the main project workflow in this order:

```powershell
python src\data\data_understanding.py
python src\features\clean_data.py
python src\features\prepare_target.py
python src\features\eda.py
python src\models\train_models.py
python src\models\tune_models.py
python src\models\track_experiments.py
python src\models\package_model.py
python -m pytest tests
```

## MLflow Experiment Tracking

Run experiment tracking:

```powershell
python src\models\track_experiments.py
```

Open the MLflow UI:

```powershell
mlflow ui
```

Then open:

- `http://127.0.0.1:5000`

## FastAPI Inference Service

Run the API locally:

```powershell
uvicorn src.api.main:app --reload
```

Local API endpoints:

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`

Local Swagger UI:

- `http://127.0.0.1:8000/docs`

## Public Deployment

The API is publicly deployed on Render.

Public deployed URL:

- `https://heart-disease-api-lo3h.onrender.com`

Public endpoints:

- Swagger UI: `https://heart-disease-api-lo3h.onrender.com/docs`
- Health endpoint: `https://heart-disease-api-lo3h.onrender.com/health`
- Predict endpoint: `https://heart-disease-api-lo3h.onrender.com/predict`

## API Testing

To test the API locally or on the public deployment:

1. Open the Swagger UI.
2. Expand `POST /predict`.
3. Click `Try it out`.
4. Use the sample JSON request.
5. Click `Execute`.
6. Confirm that the response returns `prediction` and `confidence`.

## Docker and Containerization

The project includes containerization support through the repository configuration.

The intended setup packages:

- Python runtime
- project dependencies
- FastAPI application
- saved model artifacts
- startup configuration

During development, full local container execution could not be completed on the office laptop because Docker-based tooling required additional platform components and administrator-level installation support. Because of this setup limitation, the container configuration was prepared in the repository, while the final working deployment was completed through Render using the cloud build process.

## CI/CD

GitHub Actions workflow file:

- `.github/workflows/ci.yml`

The CI pipeline performs:

- dependency installation
- linting with `flake8`
- unit testing with `pytest`
- model training and packaging validation

## Kubernetes Deployment

Prepared deployment files:

- `k8s/heart-disease-deployment.yaml`
- `k8s/heart-disease-service.yaml`
- `k8s/deployment_guide.md`

These files are included as deployment-ready manifests for Kubernetes-based deployment.

## Key Results

- Final dataset size: `303` rows
- Final target: binary `target`
- Best model: `logistic_regression_tuned`
- Test ROC-AUC: `0.966`
- Test Recall: `0.929`
- Test F1-score: `0.881`

## Important Artifacts

- EDA findings: `artifacts/eda/eda_findings.md`
- Model comparison: `artifacts/model_training/model_comparison.csv`
- Tuned model summary: `artifacts/model_tuning/tuned_model_summary.md`
- Experiment summary: `artifacts/experiment_tracking/experiment_summary.md`
- Packaging summary: `artifacts/model_packaging/packaging_summary.md`
- Monitoring summary: `artifacts/monitoring/monitoring_summary.md`
- Screenshots: `artifacts/screenshots/`
- Final report: `artifacts/report/`

## Repository Link

GitHub repository:

- `https://github.com/2024ac05094-Riyaz-Bits/heart-disease-mlops-assignment`
