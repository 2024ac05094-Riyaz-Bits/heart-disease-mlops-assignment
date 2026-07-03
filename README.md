# Heart Disease MLOps Project

This repository contains the end-to-end MLOps assignment for heart disease prediction using the UCI Heart Disease dataset.

## Goal

Build a reproducible machine learning pipeline that:

- performs EDA and preprocessing
- trains and compares multiple classification models
- tracks experiments with MLflow
- serves predictions through a FastAPI API
- runs inside Docker
- supports CI/CD, deployment, and monitoring

## Project Structure

- data/raw: original dataset files
- data/processed: cleaned datasets
- notebooks: EDA and experiments
- src: production code
- tests: unit tests
- artifacts: plots and reports
- models: saved model files
- k8s: deployment manifests

## Next Step

Copy the UCI files `processed.cleveland.data` and `heart-disease.names` into `data/raw/`.
