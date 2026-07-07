from __future__ import annotations

import json

import joblib
import pandas as pd

from src.models.package_model import (
    FINAL_MODEL_PATH,
    METADATA_PATH,
    SAMPLE_INPUT_PATH,
    build_sample_input,
    load_model,
)
from src.models.train_models import load_training_data


def test_packaged_model_files_exist() -> None:
    assert FINAL_MODEL_PATH.exists()
    assert METADATA_PATH.exists()
    assert SAMPLE_INPUT_PATH.exists()


def test_build_sample_input_matches_training_columns() -> None:
    X, _ = load_training_data()
    sample_input = build_sample_input(X)

    assert list(sample_input.keys()) == list(X.columns)


def test_final_model_can_predict_from_saved_sample_input() -> None:
    model = joblib.load(FINAL_MODEL_PATH)
    sample_input = json.loads(SAMPLE_INPUT_PATH.read_text(encoding="utf-8"))
    sample_df = pd.DataFrame([sample_input])

    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0][1]

    assert prediction in [0, 1]
    assert 0.0 <= float(probability) <= 1.0


def test_source_model_loads_successfully() -> None:
    model = load_model()
    assert hasattr(model, "predict")
