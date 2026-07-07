from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import (
    COLUMN_NAMES,
    TARGET_COLUMN,
    load_heart_disease_data,
    print_dataset_summary,
    summarize_dataset,
)


def main() -> None:
    dataset = load_heart_disease_data()
    summary = summarize_dataset(dataset)

    print("Step 3 - Data Understanding")
    print("===========================\n")
    print("Columns used in the processed Cleveland dataset:")
    for index, column in enumerate(COLUMN_NAMES, start=1):
        print(f"{index}. {column}")

    print()
    print_dataset_summary(summary)

    print("\nMeaning of target column:")
    print(f"- {TARGET_COLUMN} = 0 means no heart disease")
    print(f"- {TARGET_COLUMN} = 1, 2, 3, 4 mean heart disease is present")

    print("\nStep 3 conclusion:")
    print("- The raw data loaded successfully")
    print("- Missing values exist in ca and thal")
    print("- Duplicate rows are not present")
    print("- The target is currently multiclass and will be converted later for binary classification")


if __name__ == "__main__":
    main()
