import asyncio
import gc
import io
import re
import threading
from collections.abc import AsyncGenerator

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from tts_service.config import settings

_model = None
_model_lock = threading.Lock()
_active_attn_implementation: str | None = None


def detect_attn_implementation() -> str:
    if settings.attn_implementation != "auto":
        return settings.attn_implementation
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        pass
    if hasattr(torch, "__version__") and tuple(
        int(x) for x in torch.__version__.split(".")[:2]
    ) >= (2, 0):
        return "sdpa"
    return "eager"


def detect_device() -> str:
    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    if settings.dtype != "auto":
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        return dtype_map.get(settings.dtype, torch.float32)
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_model() -> Qwen3TTSModel:
    global _model, _active_attn_implementation
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        device = detect_device()
        dtype = detect_dtype(device)
        attn_impl = detect_attn_implementation()
        _active_attn_implementation = attn_impl
        _model = Qwen3TTSModel.from_pretrained(
            settings.model_name,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        return _model


def get_active_attn_implementation() -> str | None:
    return _active_attn_implementation


def split_text(text: str, max_chunk_chars: int = 500) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = sentences[0]
    for sentence in sentences[1:]:
        if len(current) + 1 + len(sentence) <= max_chunk_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    chunks.append(current)
    return chunks


def _synthesize_sync(text: str, language: str, speaker: str) -> bytes:
    model = get_model()
    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(text, speaker, language)
        audio_array = wavs[0]
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_array, samplerate=sample_rate, format="WAV")
        result = buf.getvalue()
    del wavs, buf
    gc.collect()
    return result


async def synthesize(text: str, language: str, speaker: str) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _synthesize_sync, text, language, speaker)


async def synthesize_stream(
    text: str, language: str, speaker: str
) -> AsyncGenerator[tuple[int, int, bytes], None]:
    chunks = split_text(text)
    total = len(chunks)
    loop = asyncio.get_running_loop()
    for index, chunk in enumerate(chunks):
        wav_bytes = await loop.run_in_executor(
            None, _synthesize_sync, chunk, language, speaker
        )
        yield (index, total, wav_bytes)
