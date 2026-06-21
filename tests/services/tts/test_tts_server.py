import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "tts"))


class TestSynthesizeEndpoint:
    def test_synthesize_happy_path(self, client):
        response = client.post(
            "/synthesize",
            json={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0

    def test_synthesize_empty_text_returns_422(self, client):
        response = client.post(
            "/synthesize",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_synthesize_whitespace_only_returns_422(self, client):
        response = client.post(
            "/synthesize",
            json={"text": "   "},
        )
        assert response.status_code == 422

    def test_synthesize_text_too_long_returns_422(self, client):
        long_text = "a" * 5001
        response = client.post(
            "/synthesize",
            json={"text": long_text},
        )
        assert response.status_code == 422

    def test_synthesize_service_error_returns_500(self):
        async def failing_synthesize(text: str, language: str, speaker: str) -> bytes:
            raise RuntimeError("Error")

        async def dummy_stream(text, language, speaker):
            yield (0, 1, b"")

        with patch("tts_service.server.synthesize", failing_synthesize), patch(
            "tts_service.server.synthesize_stream", dummy_stream
        ):
            from tts_service.server import app
            from starlette.testclient import TestClient

            error_client = TestClient(app)
            response = error_client.post(
                "/synthesize",
                json={"text": "Hello"},
            )
            assert response.status_code == 500

    def test_synthesize_custom_voice_params(self, mock_synthesizer):
        called_with = {}

        async def tracking_synthesize(
            text: str, language: str, speaker: str
        ) -> bytes:
            called_with["text"] = text
            called_with["language"] = language
            called_with["speaker"] = speaker
            return await mock_synthesizer(text, language, speaker)

        async def dummy_stream(text, language, speaker):
            yield (0, 1, b"")

        with patch("tts_service.server.synthesize", tracking_synthesize), patch(
            "tts_service.server.synthesize_stream", dummy_stream
        ):
            from tts_service.server import app
            from starlette.testclient import TestClient

            tc = TestClient(app)
            response = tc.post(
                "/synthesize",
                json={
                    "text": "Bonjour",
                    "language": "French",
                    "speaker": "Marie",
                },
            )
            assert response.status_code == 200
            assert called_with["language"] == "French"
            assert called_with["speaker"] == "Marie"

    def test_health_endpoint(self):
        with patch("tts_service.server.get_model", return_value=None), patch(
            "tts_service.server.get_active_attn_implementation",
            return_value="sdpa",
        ):
            from tts_service.server import app
            from starlette.testclient import TestClient

            health_client = TestClient(app)
            response = health_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "model_name" in data
            assert data["attn_implementation"] == "sdpa"

    def test_cors_headers_present(self, client):
        response = client.options(
            "/synthesize",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in response.headers


class TestSynthesizeStreamEndpoint:
    def test_stream_returns_ndjson(self, client):
        response = client.post(
            "/synthesize/stream",
            json={"text": "Hello world. This is a test."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

    def test_stream_response_structure(self, client):
        response = client.post(
            "/synthesize/stream",
            json={"text": "Hello world. This is a test."},
        )
        lines = [
            line for line in response.text.strip().split("\n") if line.strip()
        ]
        assert len(lines) >= 2  # At least one audio chunk + done

        # Check audio chunks
        for line in lines[:-1]:
            data = json.loads(line)
            assert "index" in data
            assert "total" in data
            assert "audio" in data
            # audio should be base64 encoded
            import base64

            decoded = base64.b64decode(data["audio"])
            assert decoded[:4] == b"RIFF"

        # Check final done line
        done_data = json.loads(lines[-1])
        assert done_data["done"] is True

    def test_stream_empty_text_returns_422(self, client):
        response = client.post(
            "/synthesize/stream",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_stream_whitespace_only_returns_422(self, client):
        response = client.post(
            "/synthesize/stream",
            json={"text": "   "},
        )
        assert response.status_code == 422

    def test_stream_text_too_long_returns_422(self, client):
        long_text = "a" * 5001
        response = client.post(
            "/synthesize/stream",
            json={"text": long_text},
        )
        assert response.status_code == 422

    def test_stream_lines_are_parseable_json(self, client):
        response = client.post(
            "/synthesize/stream",
            json={"text": "First test. Second test."},
        )
        lines = [
            line for line in response.text.strip().split("\n") if line.strip()
        ]
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
