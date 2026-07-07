from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_heart_disease_data
from src.features.preprocess import (
    IMPUTED_DATA_PATH,
    impute_missing_values,
    save_dataframe,
    summarize_imputation,
)


def main() -> None:
    raw_df = load_heart_disease_data()
    imputed_df = impute_missing_values(raw_df)
    summary = summarize_imputation(raw_df, imputed_df)
    output_path = save_dataframe(imputed_df, IMPUTED_DATA_PATH)

    print("Step 4 - Data Cleaning")
    print("======================\n")

    print("Raw missing values:")
    for column, count in summary["raw_missing_values"].items():
        print(f"- {column}: {count}")

    print("\nMissing values after cleaning:")
    for column, count in summary["imputed_missing_values"].items():
        print(f"- {column}: {count}")

    print("\nOutput shape:", summary["output_shape"])
    print("Imputed dataset saved to:", output_path)

    print("\nStep 4 conclusion:")
    print("- Missing values in ca and thal were filled using mode imputation")
    print("- The cleaned dataset is saved without changing the original target column yet")


if __name__ == "__main__":
    main()
