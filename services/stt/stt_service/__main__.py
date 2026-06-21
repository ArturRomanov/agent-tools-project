import uvicorn

from stt_service.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "stt_service.server:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
    )
