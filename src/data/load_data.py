from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "num"
COLUMN_NAMES = FEATURE_COLUMNS + [TARGET_COLUMN]
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "processed.cleveland.data"
)


def load_heart_disease_data(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the processed Cleveland heart disease dataset."""
    return pd.read_csv(data_path, names=COLUMN_NAMES, na_values="?")


def summarize_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary for Step 3 data inspection."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }


def print_dataset_summary(summary: dict[str, object]) -> None:
    """Print a readable terminal summary for the dataset."""
    print("Dataset shape:", summary["shape"])
    print("\nColumn data types:")
    for column, dtype in summary["dtypes"].items():
        print(f"- {column}: {dtype}")

    print("\nMissing values:")
    for column, count in summary["missing_values"].items():
        print(f"- {column}: {count}")

    print("\nDuplicate rows:", summary["duplicate_rows"])
    print("\nTarget distribution:")
    for value, count in summary["target_distribution"].items():
        print(f"- num={value}: {count}")


if __name__ == "__main__":
    dataset = load_heart_disease_data()
    summary = summarize_dataset(dataset)
    print_dataset_summary(summary)
