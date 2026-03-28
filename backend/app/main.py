import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.middleware_request_id import RequestContextMiddleware
from .api.routes_chat import router as chat_router
from .api.routes_health import router as health_router
from .api.routes_rag import router as rag_router
from .config import configure_logging, get_settings

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(rag_router)
logger.info(
    "backend startup configured",
    extra={
        "event": "app.startup",
        "log_level": settings.log_level,
        "log_format": settings.log_format,
        "log_payload_mode": settings.log_payload_mode,
    },
)


def start_server():
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start_server()
