from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from stt_service.config import settings
from stt_service.transcriber import transcribe, get_model

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "application/octet-stream",
}

app = FastAPI(title="STT Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio format: {file.content_type}",
        )

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty audio file")

    max_bytes = settings.max_audio_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_audio_mb}MB",
        )

    try:
        result = await transcribe(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@app.get("/health")
async def health():
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_size": settings.model_size,
        "device": settings.device,
        "compute_type": settings.compute_type,
    }
