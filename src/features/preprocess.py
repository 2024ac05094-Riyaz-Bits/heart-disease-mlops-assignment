from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import TARGET_COLUMN

IMPUTED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "heart_disease_imputed.csv"
)
PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "heart_disease_processed.csv"
)

DISCRETE_FEATURE_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the missing values found during Step 3 inspection."""
    imputed_df = df.copy()

    for column in ["ca", "thal"]:
        imputed_df[column] = imputed_df[column].fillna(imputed_df[column].mode().iloc[0])

    for column in DISCRETE_FEATURE_COLUMNS + [TARGET_COLUMN]:
        imputed_df[column] = imputed_df[column].astype(int)

    return imputed_df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary heart disease target used for modeling."""
    prepared_df = df.copy()
    prepared_df["target"] = (prepared_df[TARGET_COLUMN] > 0).astype(int)
    return prepared_df


def summarize_imputation(raw_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary for the missing-value cleaning step."""
    return {
        "raw_missing_values": raw_df.isna().sum().to_dict(),
        "imputed_missing_values": imputed_df.isna().sum().to_dict(),
        "output_shape": imputed_df.shape,
    }


def summarize_target_preparation(prepared_df: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary for binary-target preparation."""
    return {
        "raw_target_distribution": prepared_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "binary_target_distribution": prepared_df["target"].value_counts().sort_index().to_dict(),
        "output_shape": prepared_df.shape,
    }


def save_dataframe(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Save a processed dataframe to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
