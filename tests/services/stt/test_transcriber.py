import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "stt"))


class TestDetectDevice:
    def test_detect_device_returns_cpu_when_no_cuda(self):
        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.device = "auto"
            from stt_service.transcriber import detect_device

            with patch.dict("sys.modules", {"ctranslate2": None}):
                # When ctranslate2 can't be imported, should fall back to cpu
                result = detect_device()
                assert result == "cpu"

    def test_detect_device_returns_cuda_when_available(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.return_value = ["cuda", "float16", "int8"]

        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.device = "auto"
            from stt_service.transcriber import detect_device

            with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
                with patch("stt_service.transcriber.detect_device") as mock_detect:
                    mock_detect.return_value = "cuda"
                    result = mock_detect()
                    assert result == "cuda"

    def test_detect_device_respects_explicit_setting(self):
        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.device = "cpu"
            from stt_service.transcriber import detect_device

            result = detect_device()
            assert result == "cpu"


class TestDetectComputeType:
    def test_detect_compute_type_auto_cpu(self):
        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.compute_type = "auto"
            from stt_service.transcriber import detect_compute_type

            result = detect_compute_type("cpu")
            assert result == "int8"

    def test_detect_compute_type_auto_cuda(self):
        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.compute_type = "auto"
            from stt_service.transcriber import detect_compute_type

            result = detect_compute_type("cuda")
            assert result == "float16"

    def test_detect_compute_type_respects_explicit(self):
        with patch("stt_service.transcriber.settings") as mock_settings:
            mock_settings.compute_type = "float32"
            from stt_service.transcriber import detect_compute_type

            result = detect_compute_type("cpu")
            assert result == "float32"


class TestGetModel:
    def test_get_model_lazy_loads_once(self):
        import stt_service.transcriber as mod

        mod._model = None  # Reset singleton

        fake_model = MagicMock()
        with patch("stt_service.transcriber.WhisperModel", return_value=fake_model):
            with patch("stt_service.transcriber.detect_device", return_value="cpu"):
                with patch(
                    "stt_service.transcriber.detect_compute_type", return_value="int8"
                ):
                    model1 = mod.get_model()
                    model2 = mod.get_model()
                    assert model1 is model2
                    assert model1 is fake_model

        mod._model = None  # Clean up singleton


class TestTranscribe:
    def test_transcribe_returns_expected_format(self):
        from stt_service.transcriber import _transcribe_sync

        fake_segments = [MagicMock(text="Hello"), MagicMock(text="world")]
        fake_info = MagicMock(language="en", duration=2.5)

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (fake_segments, fake_info)

        import stt_service.transcriber as mod

        original = mod._model
        mod._model = fake_model

        result = _transcribe_sync(b"fake audio data")
        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["duration_seconds"] == 2.5

        mod._model = original

    def test_transcribe_handles_empty_segments(self):
        from stt_service.transcriber import _transcribe_sync

        fake_segments = [MagicMock(text=""), MagicMock(text="  ")]
        fake_info = MagicMock(language="en", duration=1.0)

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (fake_segments, fake_info)

        import stt_service.transcriber as mod

        original = mod._model
        mod._model = fake_model

        result = _transcribe_sync(b"fake audio data")
        assert result["text"] == ""
        assert result["language"] == "en"

        mod._model = original
