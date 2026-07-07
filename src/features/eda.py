from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.preprocess import PROCESSED_DATA_PATH

EDA_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"
EDA_FINDINGS_PATH = EDA_OUTPUT_DIR / "eda_findings.md"

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_TARGET_COLUMNS = ["cp", "exang", "thal", "ca"]


def load_processed_data(data_path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def save_missing_values_plot(df: pd.DataFrame) -> Path:
    missing_counts = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="Blues_d", ax=ax)
    ax.set_title("Missing Value Summary")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing Value Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "missing_values_summary.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_class_distribution_plot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="target", palette="Set2", ax=ax)
    ax.set_title("Class Distribution of Target")
    ax.set_xlabel("Target (0 = no disease, 1 = disease present)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "class_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_histograms(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for index, column in enumerate(NUMERIC_COLUMNS):
        sns.histplot(df[column], kde=True, bins=20, color="#2a9d8f", ax=axes[index])
        axes[index].set_title(f"Distribution of {column}")

    axes[-1].axis("off")
    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "numeric_histograms.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr_df = df[NUMERIC_COLUMNS + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_boxplots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for index, column in enumerate(NUMERIC_COLUMNS):
        sns.boxplot(y=df[column], color="#f4a261", ax=axes[index])
        axes[index].set_title(f"Boxplot of {column}")

    axes[-1].axis("off")
    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "numeric_boxplots.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_feature_vs_target_plots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for index, column in enumerate(CATEGORICAL_TARGET_COLUMNS):
        sns.countplot(data=df, x=column, hue="target", palette="Set1", ax=axes[index])
        axes[index].set_title(f"{column} vs target")
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Count")

    fig.tight_layout()
    output_path = EDA_OUTPUT_DIR / "feature_vs_target.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_findings(df: pd.DataFrame) -> str:
    class_counts = df["target"].value_counts().sort_index()
    imbalance_comment = (
        "The target classes are reasonably balanced."
        if abs(class_counts[0] - class_counts[1]) <= 30
        else "There is noticeable class imbalance in the target distribution."
    )

    skewness = df[NUMERIC_COLUMNS].skew().sort_values(ascending=False)
    most_skewed = skewness.index[0]

    corr_with_target = df[NUMERIC_COLUMNS + ["target"]].corr()["target"].drop("target")
    strongest_corr = corr_with_target.abs().sort_values(ascending=False).index[0]
    strongest_corr_value = corr_with_target[strongest_corr]

    feature_risk_notes = []
    for column in CATEGORICAL_TARGET_COLUMNS:
        risk_by_group = df.groupby(column)["target"].mean().sort_values(ascending=False)
        top_group = risk_by_group.index[0]
        top_rate = risk_by_group.iloc[0]
        feature_risk_notes.append(
            f"- `{column}`: category `{top_group}` has the highest disease rate at {top_rate:.2%}."
        )

    findings = f"""# EDA Findings

## Dataset Overview

- Final processed dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.
- The dataset now includes the binary `target` column used for heart disease classification.

## Missing Value Summary

- No missing values remain in the processed dataset after Step 4 cleaning.
- This means the data is ready for consistent plotting and later modeling.

## Class Distribution of Target

- `target = 0`: {class_counts[0]} patients
- `target = 1`: {class_counts[1]} patients
- {imbalance_comment}
- Impact on prediction: the near-balanced classes reduce the risk of a model learning only the majority class.

## Histograms of Numeric Features

- The numeric features show that most patients fall into a middle-age range, while some health indicators have longer tails.
- `{most_skewed}` appears to be the most skewed numeric feature, which suggests a less symmetric distribution.
- Impact on prediction: skewed features may still be informative, but scaling and robust evaluation will matter later.

## Correlation Heatmap

- The strongest linear relationship with `target` among the numeric features is `{strongest_corr}` with correlation {strongest_corr_value:.2f}.
- This does not prove causation, but it suggests that `{strongest_corr}` may be useful for classification.
- Impact on prediction: features with stronger target relationships can contribute more signal to the model.

## Boxplots of Important Numeric Features

- The boxplots help reveal spread and possible outliers in age, cholesterol, resting blood pressure, maximum heart rate, and oldpeak.
- Variables like cholesterol and oldpeak often show wider spread, which may affect model sensitivity.
- Impact on prediction: outliers and spread patterns should be considered when we later scale features and compare models.

## Feature vs Target Plots

{chr(10).join(feature_risk_notes)}

- These category-level differences suggest that some clinical patterns are more associated with heart disease than others.
- Impact on prediction: categorical medical features such as chest pain type, exercise-induced angina, thal status, and vessel count are likely to be useful predictors.
"""
    return findings


def save_findings(findings: str) -> Path:
    EDA_FINDINGS_PATH.write_text(findings, encoding="utf-8")
    return EDA_FINDINGS_PATH


def main() -> None:
    sns.set_theme(style="whitegrid")
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()

    outputs = [
        save_missing_values_plot(df),
        save_class_distribution_plot(df),
        save_histograms(df),
        save_correlation_heatmap(df),
        save_boxplots(df),
        save_feature_vs_target_plots(df),
    ]
    findings_path = save_findings(build_findings(df))

    print("Step 6 - Exploratory Data Analysis")
    print("==================================\n")
    print("Processed dataset loaded from:", PROCESSED_DATA_PATH)
    print("EDA outputs saved:")
    for output in outputs:
        print(f"- {output}")
    print(f"- {findings_path}")
    print("\nStep 6 conclusion:")
    print("- Required EDA plots were generated and saved")
    print("- A written findings file was created for reporting")
    print("- Task 1 visual analysis is now ready for review")


if __name__ == "__main__":
    main()
