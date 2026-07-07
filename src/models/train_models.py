from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.preprocess import PROCESSED_DATA_PATH

MODEL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model_training"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"
DROP_COLUMNS = ["num"]


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, Pipeline]]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []
    details: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring)

        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": model_name,
                "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
                "cv_precision_mean": cv_scores["test_precision"].mean(),
                "cv_recall_mean": cv_scores["test_recall"].mean(),
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
            }
        )

        details[model_name] = {
            "predictions": y_pred,
            "probabilities": y_proba,
        }

    results_df = pd.DataFrame(results).sort_values(by="test_roc_auc", ascending=False)
    return results_df, details, fitted_models


def save_model_comparison(results_df: pd.DataFrame) -> Path:
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_ARTIFACT_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    return output_path


def save_confusion_matrices(y_test: pd.Series, details: dict[str, dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, len(details), figsize=(12, 5))
    if len(details) == 1:
        axes = [axes]

    for axis, (model_name, model_detail) in zip(axes, details.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            model_detail["predictions"],
            cmap="Blues",
            ax=axis,
            colorbar=False,
        )
        axis.set_title(f"{model_name} confusion matrix")

    fig.tight_layout()
    output_path = MODEL_ARTIFACT_DIR / "confusion_matrices.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_roc_curves(y_test: pd.Series, details: dict[str, dict[str, object]], results_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model_detail in details.items():
        fpr, tpr, _ = roc_curve(y_test, model_detail["probabilities"])
        roc_auc = results_df.loc[results_df["model"] == model_name, "test_roc_auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    output_path = MODEL_ARTIFACT_DIR / "roc_curves.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_model_summary(results_df: pd.DataFrame) -> Path:
    best_model = results_df.iloc[0]
    lines = [
        "# Model Training Summary",
        "",
        "## Models Compared",
        "",
        "- Logistic Regression",
        "- Random Forest",
        "",
        "## Best Baseline Model",
        "",
        f"- Model: `{best_model['model']}`",
        f"- Test ROC-AUC: `{best_model['test_roc_auc']:.3f}`",
        f"- Test Recall: `{best_model['test_recall']:.3f}`",
        f"- Test F1: `{best_model['test_f1']:.3f}`",
        "",
        "## Notes",
        "",
        "- Cross-validation metrics are included in `model_comparison.csv`.",
        "- These are baseline models before hyperparameter tuning.",
        "- Later steps can use this comparison to justify model selection.",
    ]
    output_path = MODEL_ARTIFACT_DIR / "model_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_best_baseline_model(results_df: pd.DataFrame, fitted_models: dict[str, Pipeline]) -> Path:
    best_model_name = results_df.iloc[0]["model"]
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_OUTPUT_DIR / "baseline_best_model.joblib"
    joblib.dump(fitted_models[best_model_name], output_path)
    return output_path


def main() -> None:
    sns.set_theme(style="whitegrid")
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_model_pipelines()
    results_df, details, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)

    comparison_path = save_model_comparison(results_df)
    confusion_path = save_confusion_matrices(y_test, details)
    roc_path = save_roc_curves(y_test, details, results_df)
    summary_path = save_model_summary(results_df)
    model_path = save_best_baseline_model(results_df, fitted_models)

    print("Step 7 - Baseline Feature Engineering and Model Development")
    print("===========================================================\n")
    print("Models trained:")
    for model_name in results_df["model"]:
        print(f"- {model_name}")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nOutputs saved:")
    print(f"- {comparison_path}")
    print(f"- {confusion_path}")
    print(f"- {roc_path}")
    print(f"- {summary_path}")
    print(f"- {model_path}")

    print("\nStep 7 conclusion:")
    print("- Two baseline classifiers were trained and evaluated")
    print("- Cross-validation and test-set metrics were recorded")
    print("- A best baseline model was saved for later comparison")


if __name__ == "__main__":
    main()
