# Submission Checklist

## Deliverables Check

- GitHub repository with code: `Completed`
- `Dockerfile`: `Completed`
- `requirements.txt`: `Completed`
- cleaned dataset: `Completed`
- download instructions/script: `Completed`
- EDA, training, and inference scripts: `Completed`
- `tests/` folder with unit tests: `Completed`
- GitHub Actions workflow YAML: `Completed`
- deployment manifests: `Completed`
- screenshot folder for reporting: `Completed`
- final written report in Markdown: `Completed`
- final written report in PDF: `Completed`
- deployed API URL: `Not applicable for this submission`
- local API access instructions: `Completed`
- short video recording of pipeline: `Pending`

## Production-Readiness Requirements Check

- all scripts execute from clean setup using `requirements.txt`: `Completed`
- model serves correctly through local FastAPI API: `Completed`
- Docker isolated-environment build/test proof: `Partially completed`
- pipeline fails on code or test errors with clear logs: `Completed`

## Notes

- Public deployment was not completed on the office-managed laptop because Docker, WSL, and Kubernetes tooling were restricted.
- Kubernetes manifests and deployment instructions are prepared for execution on another machine.
- Docker support exists in the repository, but if you do not have a successful Docker build screenshot or run proof, add an honest note in the submission.
- The API is being submitted with local testing instructions through `http://127.0.0.1:8000/docs`.
