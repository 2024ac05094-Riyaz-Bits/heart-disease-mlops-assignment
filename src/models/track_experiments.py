from __future__ import annotations

from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_models import build_preprocessor, load_training_data

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment_tracking"


def build_experiment_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression_tracking": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(C=0.1, solver="liblinear", max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest_tracking": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                )),
            ]
        ),
    }


def extract_model_params(pipeline: Pipeline) -> dict[str, object]:
    classifier = pipeline.named_steps["classifier"]
    return classifier.get_params()


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_precision_mean": float(cv_scores["test_precision"].mean()),
        "cv_recall_mean": float(cv_scores["test_recall"].mean()),
        "cv_f1_mean": float(cv_scores["test_f1"].mean()),
        "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_pred, y_proba


def save_confusion_matrix(y_test: pd.Series, y_pred: pd.Series, model_name: str) -> Path:
    TRACKING_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Purples",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    output_path = TRACKING_ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_roc_curve(y_test: pd.Series, y_proba: pd.Series, model_name: str, roc_auc: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{model_name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    output_path = TRACKING_ARTIFACT_DIR / f"{model_name}_roc_curve.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_run_summary(results_df: pd.DataFrame) -> Path:
    lines = [
        "# Experiment Tracking Summary",
        "",
        "## Runs logged",
        "",
    ]
    for _, row in results_df.iterrows():
        lines.extend(
            [
                f"- `{row['model']}`",
                f"  - Test ROC-AUC: `{row['test_roc_auc']:.3f}`",
                f"  - Test Recall: `{row['test_recall']:.3f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Each run logs parameters, metrics, model files, and plot artifacts.",
            "- MLflow tracking data is stored locally inside the `mlruns/` folder.",
            "- This satisfies the experiment tracking requirement for the assignment.",
        ]
    )

    output_path = TRACKING_ARTIFACT_DIR / "experiment_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    sns.set_theme(style="whitegrid")
    TRACKING_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("heart_disease_classification")

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_experiment_models()
    rows: list[dict[str, object]] = []

    for model_name, pipeline in models.items():
        with mlflow.start_run(run_name=model_name):
            params = extract_model_params(pipeline)
            metrics, y_pred, y_proba = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="pickle",
            )

            confusion_path = save_confusion_matrix(y_test, y_pred, model_name)
            roc_path = save_roc_curve(y_test, y_proba, model_name, metrics["test_roc_auc"])

            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(roc_path))

            row = {"model": model_name}
            row.update(metrics)
            row["artifact_confusion_matrix"] = str(confusion_path)
            row["artifact_roc_curve"] = str(roc_path)
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(by="test_roc_auc", ascending=False)
    comparison_path = TRACKING_ARTIFACT_DIR / "experiment_results.csv"
    results_df.to_csv(comparison_path, index=False)
    summary_path = save_run_summary(results_df)

    print("Step 9 - Experiment Tracking")
    print("============================\n")
    print("MLflow experiment name: heart_disease_classification")
    print(f"Tracking directory: {MLRUNS_DIR}")

    print("\nRuns logged:")
    for model_name in results_df["model"]:
        print(f"- {model_name}")

    print("\nExperiment results:")
    print(results_df[[
        "model",
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
        "cv_roc_auc_mean",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]].to_string(index=False))

    print("\nOutputs saved:")
    print(f"- {comparison_path}")
    print(f"- {summary_path}")
    print(f"- {TRACKING_ARTIFACT_DIR}")
    print(f"- {MLRUNS_DIR}")

    print("\nStep 9 conclusion:")
    print("- Experiment runs were tracked with MLflow")
    print("- Parameters, metrics, models, and plots were logged")
    print("- The project now has reproducible local experiment tracking")


if __name__ == "__main__":
    main()
