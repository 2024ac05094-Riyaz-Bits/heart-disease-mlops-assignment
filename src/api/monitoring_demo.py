from __future__ import annotations

from pathlib import Path
import json
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.main import LOG_FILE, app

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "monitoring"
SAMPLE_REQUEST_PATH = PROJECT_ROOT / "artifacts" / "api" / "sample_request.json"
METRICS_OUTPUT_PATH = ARTIFACT_DIR / "metrics_snapshot.json"
SUMMARY_PATH = ARTIFACT_DIR / "monitoring_summary.md"


def main() -> None:
    client = TestClient(app)
    payload = json.loads(SAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))

    client.get("/")
    client.get("/health")
    prediction_response = client.post("/predict", json=payload)
    prediction_response.raise_for_status()
    metrics_response = client.get("/metrics")
    metrics_response.raise_for_status()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.write_text(
        json.dumps(metrics_response.json(), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Monitoring Summary",
        "",
        "## Monitoring Features",
        "",
        "- API request logging is enabled through FastAPI middleware.",
        "- Health endpoint activity is counted.",
        "- Prediction request counts and average latency are tracked.",
        "- Metrics are exposed at `/metrics`.",
        "",
        "## Artifacts",
        "",
        f"- Metrics snapshot: `{METRICS_OUTPUT_PATH}`",
        f"- Request log file: `{LOG_FILE}`",
        "",
        "## Example Prediction Response",
        "",
        f"```json\n{json.dumps(prediction_response.json(), indent=2)}\n```",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("Step 16 - Monitoring and Logging")
    print("================================\n")
    print("Outputs saved:")
    print(f"- {METRICS_OUTPUT_PATH}")
    print(f"- {LOG_FILE}")
    print(f"- {SUMMARY_PATH}")

    print("\nCurrent metrics:")
    print(json.dumps(metrics_response.json(), indent=2))


if __name__ == "__main__":
    main()
