from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import TARGET_COLUMN, load_heart_disease_data
from src.features.preprocess import (
    IMPUTED_DATA_PATH,
    PROCESSED_DATA_PATH,
    add_binary_target,
    impute_missing_values,
    save_dataframe,
    summarize_target_preparation,
)


def load_imputed_or_build() -> pd.DataFrame:
    """Use the Step 4 output if it exists; otherwise build it from raw data."""
    if IMPUTED_DATA_PATH.exists():
        return pd.read_csv(IMPUTED_DATA_PATH)

    raw_df = load_heart_disease_data()
    return impute_missing_values(raw_df)


def main() -> None:
    imputed_df = load_imputed_or_build()
    prepared_df = add_binary_target(imputed_df)
    summary = summarize_target_preparation(prepared_df)
    output_path = save_dataframe(prepared_df, PROCESSED_DATA_PATH)

    print("Step 5 - Binary Target Preparation")
    print("==================================\n")

    print(f"Original target column: {TARGET_COLUMN}")
    print("New target column: target")

    print("\nRaw target distribution (num):")
    for value, count in summary["raw_target_distribution"].items():
        print(f"- num={value}: {count}")

    print("\nBinary target distribution (target):")
    for value, count in summary["binary_target_distribution"].items():
        label = "no disease" if value == 0 else "disease present"
        print(f"- target={value} ({label}): {count}")

    print("\nOutput shape:", summary["output_shape"])
    print("Final processed dataset saved to:", output_path)

    print("\nStep 5 conclusion:")
    print("- target=0 represents no heart disease")
    print("- target=1 represents heart disease present")
    print("- The dataset is now finalized for EDA and later modeling")


if __name__ == "__main__":
    main()
