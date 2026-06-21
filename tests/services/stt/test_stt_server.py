import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "stt"))


class TestTranscribeEndpoint:
    def test_transcribe_happy_path(self, client):
        audio_data = b"\x00\x01\x02\x03" * 100
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", BytesIO(audio_data), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "duration_seconds" in data
        assert data["text"] == "Hello world how are you"

    def test_transcribe_empty_file_returns_422(self, client):
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", BytesIO(b""), "audio/wav")},
        )
        assert response.status_code == 422

    def test_transcribe_oversized_file_returns_413(self, client):
        large_data = b"\x00" * (26 * 1024 * 1024)  # 26MB > 25MB limit
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", BytesIO(large_data), "audio/wav")},
        )
        assert response.status_code == 413

    def test_transcribe_unsupported_format_returns_422(self, client):
        response = client.post(
            "/transcribe",
            files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 422

    def test_transcribe_service_error_returns_500(self):
        async def failing_transcribe(audio_bytes: bytes) -> dict:
            raise RuntimeError("Model failed")

        with patch("stt_service.server.transcribe", failing_transcribe):
            from stt_service.server import app
            from starlette.testclient import TestClient

            error_client = TestClient(app)
            audio_data = b"\x00\x01\x02\x03" * 100
            response = error_client.post(
                "/transcribe",
                files={"file": ("test.wav", BytesIO(audio_data), "audio/wav")},
            )
            assert response.status_code == 500

    def test_health_endpoint(self):
        with patch("stt_service.server.get_model", return_value=None):
            from stt_service.server import app
            from starlette.testclient import TestClient

            health_client = TestClient(app)
            response = health_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "model_size" in data

    def test_cors_headers_present(self, client):
        response = client.options(
            "/transcribe",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in response.headers
