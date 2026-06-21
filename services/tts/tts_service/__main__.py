import uvicorn

from tts_service.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "tts_service.server:app",
        host="0.0.0.0",
        port=8004,
        log_level="info",
    )
