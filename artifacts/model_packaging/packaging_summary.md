# Model Packaging Summary

## Packaged Outputs

- Final reusable model: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\models\final_heart_disease_model.joblib`
- Metadata file: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\models\model_metadata.json`
- Sample input file: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\models\sample_input.json`
- Prediction verification file: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\artifacts\model_packaging\prediction_check.json`

## Reproducibility Notes

- The saved joblib file contains both preprocessing and classifier steps.
- A sample input JSON file is included to test loading and prediction later.
- Metadata records feature columns and model parameters for reference.

## Usage

- Load the model with `joblib.load(...)`.
- Pass a pandas DataFrame with the same feature columns.
- No retraining is required for inference.