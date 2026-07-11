from __future__ import annotations

from pathlib import Path
import json
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.main import app

SAMPLE_REQUEST_PATH = PROJECT_ROOT / "artifacts" / "api" / "sample_request.json"
SAMPLE_RESPONSE_PATH = PROJECT_ROOT / "artifacts" / "api" / "sample_response.json"


def main() -> None:
    payload = json.loads(SAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))
    client = TestClient(app)
    response = client.post("/predict", json=payload)
    response.raise_for_status()

    SAMPLE_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_RESPONSE_PATH.write_text(
        json.dumps(response.json(), indent=2),
        encoding="utf-8",
    )

    print("Step 13 - API Local Verification")
    print("================================\n")
    print("Request file:")
    print(f"- {SAMPLE_REQUEST_PATH}")
    print("\nResponse saved to:")
    print(f"- {SAMPLE_RESPONSE_PATH}")
    print("\nAPI response:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
