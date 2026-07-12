# Monitoring Summary

## Monitoring Features

- API request logging is enabled through FastAPI middleware.
- Health endpoint activity is counted.
- Prediction request counts and average latency are tracked.
- Metrics are exposed at `/metrics`.

## Artifacts

- Metrics snapshot: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\artifacts\monitoring\metrics_snapshot.json`
- Request log file: `C:\Users\RA7628\Downloads\Bits pilani\Bits traning session\Sem-3\Mlops\Assignment-1\Ass1 ans\heart-disease-mlops-assignment\artifacts\monitoring\api_requests.log`

## Example Prediction Response

```json
{
  "prediction": 0,
  "confidence": 0.425821
}
```