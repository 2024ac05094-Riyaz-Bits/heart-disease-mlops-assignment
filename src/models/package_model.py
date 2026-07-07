from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_models import DROP_COLUMNS, TARGET_COLUMN, load_training_data

MODEL_DIR = PROJECT_ROOT / "models"
PACKAGING_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model_packaging"
SOURCE_MODEL_PATH = MODEL_DIR / "tuned_best_model.joblib"
FINAL_MODEL_PATH = MODEL_DIR / "final_heart_disease_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SAMPLE_INPUT_PATH = MODEL_DIR / "sample_input.json"
PREDICTION_CHECK_PATH = PACKAGING_ARTIFACT_DIR / "prediction_check.json"
SUMMARY_PATH = PACKAGING_ARTIFACT_DIR / "packaging_summary.md"


def load_model():
    if not SOURCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected tuned model not found at: {SOURCE_MODEL_PATH}"
        )
    return joblib.load(SOURCE_MODEL_PATH)


def build_sample_input(X: pd.DataFrame) -> dict[str, object]:
    sample_row = X.iloc[0].to_dict()
    normalized: dict[str, object] = {}
    for key, value in sample_row.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def save_metadata(model, X: pd.DataFrame) -> Path:
    classifier = model.named_steps["classifier"]
    metadata = {
        "model_name": "heart_disease_classifier",
        "model_file": FINAL_MODEL_PATH.name,
        "source_model_file": SOURCE_MODEL_PATH.name,
        "target_column": TARGET_COLUMN,
        "dropped_columns": DROP_COLUMNS,
        "feature_columns": list(X.columns),
        "classifier_type": classifier.__class__.__name__,
        "classifier_parameters": classifier.get_params(),
        "packaging_note": "The preprocessing pipeline is stored inside the saved model object.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return METADATA_PATH


def save_sample_input(sample_input: dict[str, object]) -> Path:
    SAMPLE_INPUT_PATH.write_text(json.dumps(sample_input, indent=2), encoding="utf-8")
    return SAMPLE_INPUT_PATH


def save_prediction_check(model, sample_input: dict[str, object]) -> Path:
    sample_df = pd.DataFrame([sample_input])
    prediction = int(model.predict(sample_df)[0])
    probability = float(model.predict_proba(sample_df)[0][1])
    payload = {
        "sample_input": sample_input,
        "prediction": prediction,
        "confidence_for_positive_class": round(probability, 6),
    }
    PREDICTION_CHECK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PREDICTION_CHECK_PATH


def save_summary(metadata_path: Path, sample_input_path: Path, prediction_path: Path) -> Path:
    lines = [
        "# Model Packaging Summary",
        "",
        "## Packaged Outputs",
        "",
        f"- Final reusable model: `{FINAL_MODEL_PATH}`",
        f"- Metadata file: `{metadata_path}`",
        f"- Sample input file: `{sample_input_path}`",
        f"- Prediction verification file: `{prediction_path}`",
        "",
        "## Reproducibility Notes",
        "",
        "- The saved joblib file contains both preprocessing and classifier steps.",
        "- A sample input JSON file is included to test loading and prediction later.",
        "- Metadata records feature columns and model parameters for reference.",
        "",
        "## Usage",
        "",
        "- Load the model with `joblib.load(...)`.",
        "- Pass a pandas DataFrame with the same feature columns.",
        "- No retraining is required for inference.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SUMMARY_PATH


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGING_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    X, _ = load_training_data()
    sample_input = build_sample_input(X)

    joblib.dump(model, FINAL_MODEL_PATH)
    metadata_path = save_metadata(model, X)
    sample_input_path = save_sample_input(sample_input)
    prediction_path = save_prediction_check(model, sample_input)
    summary_path = save_summary(metadata_path, sample_input_path, prediction_path)

    print("Step 10 - Model Packaging and Reproducibility")
    print("=============================================\n")
    print("Packaged model source:")
    print(f"- {SOURCE_MODEL_PATH}")

    print("\nOutputs saved:")
    print(f"- {FINAL_MODEL_PATH}")
    print(f"- {metadata_path}")
    print(f"- {sample_input_path}")
    print(f"- {prediction_path}")
    print(f"- {summary_path}")

    print("\nStep 10 conclusion:")
    print("- The final model was saved in reusable joblib format")
    print("- The preprocessing pipeline remains included inside the saved model")
    print("- Metadata and sample input files were created for reproducibility")


if __name__ == "__main__":
    main()
