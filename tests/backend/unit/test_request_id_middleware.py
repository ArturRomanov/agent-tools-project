import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app  # noqa: E402
from backend.app.observability.context import get_request_id  # noqa: E402


def test_request_id_generated_and_context_cleared() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]
    assert get_request_id() is None


def test_request_id_preserved_from_header() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"


def test_request_lifecycle_logs_share_same_request_id(caplog) -> None:
    client = TestClient(app)
    caplog.set_level(logging.INFO)

    response = client.get("/health", headers={"X-Request-ID": "rid-log-1"})

    assert response.status_code == 200
    start = [
        r for r in caplog.records if getattr(r, "event", "") == "api.request.start"
    ]
    end = [r for r in caplog.records if getattr(r, "event", "") == "api.request.end"]
    assert start
    assert end
    assert start[-1].request_id == "rid-log-1"
    assert end[-1].request_id == "rid-log-1"
