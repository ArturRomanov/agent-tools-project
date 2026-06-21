import base64
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from tts_service.config import settings
from tts_service.synthesizer import (
    get_active_attn_implementation,
    get_model,
    synthesize,
    synthesize_stream,
)


class SynthesizeRequest(BaseModel):
    text: str
    language: str | None = None
    speaker: str | None = None


app = FastAPI(title="TTS Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/synthesize")
async def synthesize_text(request: SynthesizeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty")

    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Text too long. Maximum is {settings.max_text_chars} characters",
        )

    language = request.language or settings.language
    speaker = request.speaker or settings.speaker

    try:
        wav_bytes = await synthesize(text, language, speaker)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/synthesize/stream")
async def synthesize_stream_endpoint(request: SynthesizeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty")

    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Text too long. Maximum is {settings.max_text_chars} characters",
        )

    language = request.language or settings.language
    speaker = request.speaker or settings.speaker

    async def generate():
        async for index, total, wav_bytes in synthesize_stream(
            text, language, speaker
        ):
            line = json.dumps(
                {
                    "index": index,
                    "total": total,
                    "audio": base64.b64encode(wav_bytes).decode(),
                }
            )
            yield line + "\n"
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": settings.model_name,
        "device": settings.device,
        "dtype": settings.dtype,
        "attn_implementation": get_active_attn_implementation(),
    }
