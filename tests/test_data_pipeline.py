from __future__ import annotations

import pandas as pd

from src.data.load_data import COLUMN_NAMES, TARGET_COLUMN, load_heart_disease_data, summarize_dataset
from src.features.preprocess import add_binary_target, impute_missing_values


def test_load_heart_disease_data_has_expected_columns() -> None:
    df = load_heart_disease_data()

    assert list(df.columns) == COLUMN_NAMES
    assert df.shape[0] == 303


def test_summarize_dataset_reports_expected_structure() -> None:
    df = load_heart_disease_data()
    summary = summarize_dataset(df)

    assert summary["shape"] == (303, 14)
    assert "ca" in summary["missing_values"]
    assert "thal" in summary["missing_values"]
    assert TARGET_COLUMN in summary["target_distribution"] or 0 in summary["target_distribution"]


def test_impute_missing_values_removes_missing_values_from_ca_and_thal() -> None:
    raw_df = load_heart_disease_data()
    imputed_df = impute_missing_values(raw_df)

    assert imputed_df["ca"].isna().sum() == 0
    assert imputed_df["thal"].isna().sum() == 0
    assert pd.api.types.is_integer_dtype(imputed_df["ca"])
    assert pd.api.types.is_integer_dtype(imputed_df["thal"])


def test_add_binary_target_creates_only_zero_and_one() -> None:
    raw_df = load_heart_disease_data()
    imputed_df = impute_missing_values(raw_df)
    prepared_df = add_binary_target(imputed_df)

    assert "target" in prepared_df.columns
    assert set(prepared_df["target"].unique()) == {0, 1}
    assert prepared_df.loc[prepared_df[TARGET_COLUMN] == 0, "target"].eq(0).all()
    assert prepared_df.loc[prepared_df[TARGET_COLUMN] > 0, "target"].eq(1).all()
