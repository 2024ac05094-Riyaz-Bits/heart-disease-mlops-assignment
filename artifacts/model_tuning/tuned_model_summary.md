# Tuned Model Summary

## Tuned Models Compared

- Logistic Regression with GridSearchCV
- Random Forest with GridSearchCV

## Best Tuned Model

- Model: `logistic_regression_tuned`
- Best CV ROC-AUC: `0.899`
- Test ROC-AUC: `0.966`
- Test Recall: `0.929`
- Best Parameters: `{'classifier__C': 0.1, 'classifier__solver': 'liblinear'}`

## Notes

- Grid search used ROC-AUC as the selection metric.
- This step strengthens the model selection section of the assignment.
- The saved tuned model is ready for experiment tracking and packaging steps.