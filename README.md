# Heart Disease MLOps Assignment

This project builds an end-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. The workflow covers data understanding, cleaning, exploratory data analysis, feature engineering, model training, experiment tracking, packaging, API serving, CI/CD, and monitoring.

## Problem Statement

Build a machine learning classifier to predict the risk of heart disease from patient health data and prepare the solution as a cloud-ready, monitored API.

## Project Structure

- `data/raw/` - original downloaded dataset files
- `data/processed/` - cleaned and prepared datasets
- `src/data/` - dataset loading and understanding scripts
- `src/features/` - cleaning, target preparation, preprocessing, and EDA scripts
- `src/models/` - model training, tuning, tracking, and packaging scripts
- `src/api/` - FastAPI prediction service and monitoring demo
- `tests/` - unit tests for data pipeline and model packaging
- `.github/workflows/` - GitHub Actions CI pipeline
- `k8s/` - Kubernetes deployment manifests and guide
- `artifacts/` - generated plots, summaries, metrics, and screenshots
- `models/` - final packaged model and metadata

## Dataset

This project uses the UCI Heart Disease dataset, specifically the processed Cleveland file.

Place the dataset file in:

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

Run the project scripts in this order:

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

Open the API docs:

- `http://127.0.0.1:8000/docs`

Available endpoints:

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`

## Local Testing Instructions

This project is being submitted with local API access instructions instead of a public cloud URL.

Use the following steps on the local machine:

1. Open PowerShell in the project root folder.
2. Install dependencies if not already installed:

```powershell
pip install -r requirements.txt
```

3. Start the API:

```powershell
uvicorn src.api.main:app --reload
```

4. Open the Swagger UI:

- `http://127.0.0.1:8000/docs`

5. Test the prediction endpoint:

- expand `POST /predict`
- click `Try it out`
- use the sample JSON request
- click `Execute`

6. Confirm that a JSON response with `prediction` and `confidence` is returned.

## Docker

Docker support is included through the `Dockerfile`.

Build and run when Docker is available:

```powershell
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## CI/CD

GitHub Actions workflow file:

- `.github/workflows/ci.yml`

The pipeline performs:

- dependency installation
- linting with `flake8`
- unit testing with `pytest`
- model training script execution

## Kubernetes Deployment

Deployment files are available in:

- `k8s/heart-disease-deployment.yaml`
- `k8s/heart-disease-service.yaml`
- `k8s/deployment_guide.md`

These manifests are prepared for deployment, but actual Kubernetes execution depends on Docker and `kubectl` availability on the machine.

## Key Results

- Final dataset size: `303` rows
- Final modeling target: binary `target`
- Best model: `logistic_regression_tuned`
- Test ROC-AUC: `0.966`
- Test Recall: `0.929`
- Test F1: `0.881`

## Important Artifacts

- EDA findings: `artifacts/eda/eda_findings.md`
- Model comparison: `artifacts/model_training/model_comparison.csv`
- Tuned model summary: `artifacts/model_tuning/tuned_model_summary.md`
- Experiment summary: `artifacts/experiment_tracking/experiment_summary.md`
- Packaging summary: `artifacts/model_packaging/packaging_summary.md`
- Monitoring summary: `artifacts/monitoring/monitoring_summary.md`
- API screenshots: `artifacts/screenshots/`

## API Access Note

Public deployment was not used for this submission. The API can be tested locally with:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/predict`

## Repository Link

Add your GitHub repository link here before final submission.
