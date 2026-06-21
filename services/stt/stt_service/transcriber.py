import asyncio
import io
import threading
from functools import lru_cache

from faster_whisper import WhisperModel

from stt_service.config import settings

_model = None
_model_lock = threading.Lock()


def detect_device() -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def detect_compute_type(device: str) -> str:
    if settings.compute_type != "auto":
        return settings.compute_type
    if device == "cuda":
        return "float16"
    return "int8"


def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        device = detect_device()
        compute_type = detect_compute_type(device)
        _model = WhisperModel(
            settings.model_size,
            device=device,
            compute_type=compute_type,
            download_root=settings.models_dir,
        )
        return _model


def _transcribe_sync(audio_bytes: bytes) -> dict:
    model = get_model()
    segments, info = model.transcribe(io.BytesIO(audio_bytes))
    text_parts = []
    for segment in segments:
        part = segment.text.strip()
        if part:
            text_parts.append(part)
    text = " ".join(text_parts)
    return {
        "text": text,
        "language": info.language,
        "duration_seconds": round(info.duration, 2),
    }


async def transcribe(audio_bytes: bytes) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_bytes)
