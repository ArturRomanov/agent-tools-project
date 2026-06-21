import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the STT service to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "stt"))


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeTranscriptionInfo:
    def __init__(self, language: str = "en", duration: float = 3.5):
        self.language = language
        self.duration = duration


@pytest.fixture
def mock_whisper_model():
    model = MagicMock()
    model.transcribe.return_value = (
        [FakeSegment("Hello world"), FakeSegment("how are you")],
        FakeTranscriptionInfo(),
    )
    return model


@pytest.fixture
def mock_transcriber(mock_whisper_model):
    async def fake_transcribe(audio_bytes: bytes) -> dict:
        return {
            "text": "Hello world how are you",
            "language": "en",
            "duration_seconds": 3.5,
        }

    return fake_transcribe


@pytest.fixture
def client(mock_transcriber):
    from starlette.testclient import TestClient

    with patch("stt_service.server.transcribe", mock_transcriber):
        from stt_service.server import app

        yield TestClient(app)
