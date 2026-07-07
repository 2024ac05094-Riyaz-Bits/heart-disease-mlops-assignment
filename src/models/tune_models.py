from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_models import build_preprocessor, load_training_data

MODEL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model_tuning"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"


def build_search_spaces() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    preprocessor = build_preprocessor()

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    logistic_params = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    random_forest_params = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    }

    return {
        "logistic_regression_tuned": (logistic_pipeline, logistic_params),
        "random_forest_tuned": (random_forest_pipeline, random_forest_params),
    }


def run_grid_searches(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, GridSearchCV]:
    searches: dict[str, GridSearchCV] = {}

    for model_name, (pipeline, params) in build_search_spaces().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        searches[model_name] = search

    return searches


def collect_results(
    searches: dict[str, GridSearchCV], X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows = []
    details: dict[str, dict[str, object]] = {}

    for model_name, search in searches.items():
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "model": model_name,
                "best_cv_roc_auc": search.best_score_,
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
                "best_params": str(search.best_params_),
            }
        )

        details[model_name] = {
            "predictions": y_pred,
            "probabilities": y_proba,
            "best_params": search.best_params_,
        }

    results_df = pd.DataFrame(rows).sort_values(by="test_roc_auc", ascending=False)
    return results_df, details


def save_results_table(results_df: pd.DataFrame) -> Path:
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_ARTIFACT_DIR / "tuned_model_comparison.csv"
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
            cmap="Greens",
            ax=axis,
            colorbar=False,
        )
        axis.set_title(f"{model_name} confusion matrix")

    fig.tight_layout()
    output_path = MODEL_ARTIFACT_DIR / "tuned_confusion_matrices.png"
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
    ax.set_title("Tuned Model ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    output_path = MODEL_ARTIFACT_DIR / "tuned_roc_curves.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_summary(results_df: pd.DataFrame) -> Path:
    best_model = results_df.iloc[0]
    lines = [
        "# Tuned Model Summary",
        "",
        "## Tuned Models Compared",
        "",
        "- Logistic Regression with GridSearchCV",
        "- Random Forest with GridSearchCV",
        "",
        "## Best Tuned Model",
        "",
        f"- Model: `{best_model['model']}`",
        f"- Best CV ROC-AUC: `{best_model['best_cv_roc_auc']:.3f}`",
        f"- Test ROC-AUC: `{best_model['test_roc_auc']:.3f}`",
        f"- Test Recall: `{best_model['test_recall']:.3f}`",
        f"- Best Parameters: `{best_model['best_params']}`",
        "",
        "## Notes",
        "",
        "- Grid search used ROC-AUC as the selection metric.",
        "- This step strengthens the model selection section of the assignment.",
        "- The saved tuned model is ready for experiment tracking and packaging steps.",
    ]
    output_path = MODEL_ARTIFACT_DIR / "tuned_model_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_best_model(searches: dict[str, GridSearchCV], results_df: pd.DataFrame) -> Path:
    best_model_name = results_df.iloc[0]["model"]
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_OUTPUT_DIR / "tuned_best_model.joblib"
    joblib.dump(searches[best_model_name].best_estimator_, output_path)
    return output_path


def save_best_params(searches: dict[str, GridSearchCV]) -> Path:
    rows = []
    for model_name, search in searches.items():
        row = {"model": model_name}
        row.update(search.best_params_)
        rows.append(row)

    output_path = MODEL_ARTIFACT_DIR / "best_parameters.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    sns.set_theme(style="whitegrid")
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    searches = run_grid_searches(X_train, y_train)
    results_df, details = collect_results(searches, X_test, y_test)

    comparison_path = save_results_table(results_df)
    params_path = save_best_params(searches)
    confusion_path = save_confusion_matrices(y_test, details)
    roc_path = save_roc_curves(y_test, details, results_df)
    summary_path = save_summary(results_df)
    model_path = save_best_model(searches, results_df)

    print("Step 8 - Model Tuning and Final Selection")
    print("=========================================\n")
    print("Tuned models evaluated:")
    for model_name in results_df["model"]:
        print(f"- {model_name}")

    print("\nTuned model comparison:")
    print(results_df.to_string(index=False))

    print("\nOutputs saved:")
    print(f"- {comparison_path}")
    print(f"- {params_path}")
    print(f"- {confusion_path}")
    print(f"- {roc_path}")
    print(f"- {summary_path}")
    print(f"- {model_path}")

    print("\nStep 8 conclusion:")
    print("- Hyperparameter tuning was completed with GridSearchCV")
    print("- Tuned models were compared using ROC-AUC and classification metrics")
    print("- The best tuned model was saved for the next assignment steps")


if __name__ == "__main__":
    main()
