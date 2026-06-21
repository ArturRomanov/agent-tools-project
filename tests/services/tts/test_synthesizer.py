import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "tts"))


class TestDetectDevice:
    def test_detect_device_prefers_cuda(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.device = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                from tts_service.synthesizer import detect_device

                result = detect_device()
                assert result == "cuda"

    def test_detect_device_falls_back_to_mps(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.device = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = True
                from tts_service.synthesizer import detect_device

                result = detect_device()
                assert result == "mps"

    def test_detect_device_falls_back_to_cpu(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.device = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False
                from tts_service.synthesizer import detect_device

                result = detect_device()
                assert result == "cpu"


class TestDetectDtype:
    def test_detect_dtype_cuda_uses_bfloat16(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.dtype = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.bfloat16 = "bfloat16_sentinel"
                from tts_service.synthesizer import detect_dtype

                result = detect_dtype("cuda")
                assert result == "bfloat16_sentinel"

    def test_detect_dtype_cpu_uses_float32(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.dtype = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.float32 = "float32_sentinel"
                from tts_service.synthesizer import detect_dtype

                result = detect_dtype("cpu")
                assert result == "float32_sentinel"

    def test_detect_dtype_mps_uses_float32(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.dtype = "auto"
            with patch("tts_service.synthesizer.torch") as mock_torch:
                mock_torch.float32 = "float32_sentinel"
                from tts_service.synthesizer import detect_dtype

                result = detect_dtype("mps")
                assert result == "float32_sentinel"


class TestDetectAttnImplementation:
    def test_explicit_config_value(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.attn_implementation = "flash_attention_2"
            from tts_service.synthesizer import detect_attn_implementation

            result = detect_attn_implementation()
            assert result == "flash_attention_2"

    def test_auto_with_flash_attn_available(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.attn_implementation = "auto"
            flash_mock = MagicMock()
            with patch.dict("sys.modules", {"flash_attn": flash_mock}):
                from tts_service.synthesizer import detect_attn_implementation

                result = detect_attn_implementation()
                assert result == "flash_attention_2"

    def test_auto_without_flash_attn_torch2(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.attn_implementation = "auto"
            with patch.dict("sys.modules", {"flash_attn": None}):
                with patch("tts_service.synthesizer.torch") as mock_torch:
                    mock_torch.__version__ = "2.1.0"
                    from tts_service.synthesizer import detect_attn_implementation

                    result = detect_attn_implementation()
                    assert result == "sdpa"

    def test_auto_without_flash_attn_old_torch(self):
        with patch("tts_service.synthesizer.settings") as mock_settings:
            mock_settings.attn_implementation = "auto"
            with patch.dict("sys.modules", {"flash_attn": None}):
                with patch("tts_service.synthesizer.torch") as mock_torch:
                    mock_torch.__version__ = "1.13.0"
                    from tts_service.synthesizer import detect_attn_implementation

                    result = detect_attn_implementation()
                    assert result == "eager"


class TestGetModel:
    def test_get_model_lazy_loads_once(self):
        import tts_service.synthesizer as mod

        mod._model = None  # Reset singleton

        fake_model = MagicMock()
        with patch(
            "tts_service.synthesizer.Qwen3TTSModel"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = fake_model
            with patch("tts_service.synthesizer.detect_device", return_value="cpu"):
                with patch("tts_service.synthesizer.detect_dtype") as mock_dtype:
                    import torch

                    mock_dtype.return_value = torch.float32
                    with patch(
                        "tts_service.synthesizer.detect_attn_implementation",
                        return_value="sdpa",
                    ):
                        model1 = mod.get_model()
                        model2 = mod.get_model()
                        assert model1 is model2
                        assert model1 is fake_model
                        mock_cls.from_pretrained.assert_called_once()

        mod._model = None  # Clean up singleton
        mod._active_attn_implementation = None


class TestSplitText:
    def test_single_sentence(self):
        from tts_service.synthesizer import split_text

        result = split_text("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        from tts_service.synthesizer import split_text

        text = "First test. Second test. Third test."
        result = split_text(text)
        assert len(result) >= 1
        # All text should be preserved
        assert " ".join(result) == text

    def test_respects_max_chunk_chars(self):
        from tts_service.synthesizer import split_text

        text = "Short. " * 50  # ~350 chars
        result = split_text(text.strip(), max_chunk_chars=100)
        for chunk in result:
            # Each chunk might slightly exceed if a single sentence is longer,
            # but grouped sentences should respect the limit
            assert len(chunk) <= 100 or " " not in chunk

    def test_no_punctuation(self):
        from tts_service.synthesizer import split_text

        result = split_text("Hello world without punctuation")
        assert result == ["Hello world without punctuation"]

    def test_trailing_whitespace(self):
        from tts_service.synthesizer import split_text

        result = split_text("  Hello world.  ")
        assert result == ["Hello world."]

    def test_empty_string(self):
        from tts_service.synthesizer import split_text

        result = split_text("")
        assert result == []

    def test_whitespace_only(self):
        from tts_service.synthesizer import split_text

        result = split_text("   ")
        assert result == []

    def test_question_and_exclamation(self):
        from tts_service.synthesizer import split_text

        text = "Is this test? Yes it is test! Great."
        result = split_text(text)
        assert " ".join(result) == text

    def test_long_sentences_cause_multiple_chunks(self):
        from tts_service.synthesizer import split_text

        s1 = "A" * 300 + "."
        s2 = "B" * 300 + "."
        text = s1 + " " + s2
        result = split_text(text, max_chunk_chars=500)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2


class TestSynthesize:
    def test_synthesize_returns_valid_wav_bytes(self):
        import numpy as np

        from tts_service.synthesizer import _synthesize_sync

        fake_model = MagicMock()
        fake_model.config.sampling_rate = 24000
        fake_model.generate_custom_voice.return_value = (
            [np.zeros(24000, dtype=np.float32)], 24000
        )

        import tts_service.synthesizer as mod

        original = mod._model
        mod._model = fake_model

        result = _synthesize_sync("Hello", "English", "Ryan")
        # WAV files start with RIFF header
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

        mod._model = original

    def test_synthesize_passes_correct_params(self):
        import numpy as np

        from tts_service.synthesizer import _synthesize_sync

        fake_model = MagicMock()
        fake_model.config.sampling_rate = 24000
        fake_model.generate_custom_voice.return_value = (
            [np.zeros(1000, dtype=np.float32)], 24000
        )

        import tts_service.synthesizer as mod

        original = mod._model
        mod._model = fake_model

        _synthesize_sync("Bonjour", "French", "Marie")
        fake_model.generate_custom_voice.assert_called_once_with(
            "Bonjour", "Marie", "French"
        )

        mod._model = original


class TestSynthesizeStream:
    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_correct_chunks(self):
        import numpy as np

        fake_model = MagicMock()
        fake_model.config.sampling_rate = 24000
        fake_model.generate_custom_voice.return_value = (
            [np.zeros(1000, dtype=np.float32)], 24000
        )

        import tts_service.synthesizer as mod

        original = mod._model
        mod._model = fake_model

        text = "First test. Second test. Third test."
        results = []
        async for index, total, wav_bytes in mod.synthesize_stream(
            text, "English", "Ryan"
        ):
            results.append((index, total, wav_bytes))

        assert len(results) >= 1
        # All chunks should have same total
        totals = {r[1] for r in results}
        assert len(totals) == 1
        # Indices should be sequential starting from 0
        indices = [r[0] for r in results]
        assert indices == list(range(len(results)))
        # Each chunk should be valid WAV bytes
        for _, _, wav_bytes in results:
            assert wav_bytes[:4] == b"RIFF"

        mod._model = original
