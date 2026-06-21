import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the TTS service to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "tts"))


@pytest.fixture
def mock_tts_model():
    import numpy as np

    model = MagicMock()
    model.config.sampling_rate = 24000
    # Return a short silent audio array
    model.generate_custom_voice.return_value = np.zeros(24000, dtype=np.float32)
    return model


def _make_wav_bytes():
    import struct

    sample_rate = 24000
    num_samples = 100
    data_size = num_samples * 2  # 16-bit samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


@pytest.fixture
def mock_synthesizer():
    wav_bytes = _make_wav_bytes()

    async def fake_synthesize(text: str, language: str, speaker: str) -> bytes:
        return wav_bytes

    return fake_synthesize


@pytest.fixture
def mock_synthesize_stream():
    wav_bytes = _make_wav_bytes()

    async def fake_synthesize_stream(text, language, speaker):
        # Simulate splitting into 2 chunks
        for i in range(2):
            yield (i, 2, wav_bytes)

    return fake_synthesize_stream


@pytest.fixture
def client(mock_synthesizer, mock_synthesize_stream):
    from starlette.testclient import TestClient

    with patch("tts_service.server.synthesize", mock_synthesizer), patch(
        "tts_service.server.synthesize_stream", mock_synthesize_stream
    ):
        from tts_service.server import app

        yield TestClient(app)
