# EDA Findings

## Dataset Overview

- Final processed dataset shape: 303 rows x 15 columns.
- The dataset now includes the binary `target` column used for heart disease classification.

## Missing Value Summary

- No missing values remain in the processed dataset after Step 4 cleaning.
- This means the data is ready for consistent plotting and later modeling.

## Class Distribution of Target

- `target = 0`: 164 patients
- `target = 1`: 139 patients
- The target classes are reasonably balanced.
- Impact on prediction: the near-balanced classes reduce the risk of a model learning only the majority class.

## Histograms of Numeric Features

- The numeric features show that most patients fall into a middle-age range, while some health indicators have longer tails.
- `oldpeak` appears to be the most skewed numeric feature, which suggests a less symmetric distribution.
- Impact on prediction: skewed features may still be informative, but scaling and robust evaluation will matter later.

## Correlation Heatmap

- The strongest linear relationship with `target` among the numeric features is `oldpeak` with correlation 0.42.
- This does not prove causation, but it suggests that `oldpeak` may be useful for classification.
- Impact on prediction: features with stronger target relationships can contribute more signal to the model.

## Boxplots of Important Numeric Features

- The boxplots help reveal spread and possible outliers in age, cholesterol, resting blood pressure, maximum heart rate, and oldpeak.
- Variables like cholesterol and oldpeak often show wider spread, which may affect model sensitivity.
- Impact on prediction: outliers and spread patterns should be considered when we later scale features and compare models.

## Feature vs Target Plots

- `cp`: category `4` has the highest disease rate at 72.92%.
- `exang`: category `1` has the highest disease rate at 76.77%.
- `thal`: category `7` has the highest disease rate at 76.07%.
- `ca`: category `3` has the highest disease rate at 85.00%.

- These category-level differences suggest that some clinical patterns are more associated with heart disease than others.
- Impact on prediction: categorical medical features such as chest pain type, exercise-induced angina, thal status, and vessel count are likely to be useful predictors.
