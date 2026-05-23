import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.middleware_request_id import RequestContextMiddleware
from .api.routes_chat import router as chat_router
from .api.routes_health import router as health_router
from .api.routes_rag import router as rag_router
from .config import configure_logging, get_settings
from .mcp.client import MCPClient

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    web_search_client = MCPClient(
        name="web-search", server_url=settings.mcp_web_search_url
    )
    rag_client = MCPClient(name="rag", server_url=settings.mcp_rag_url)

    connected_clients: list[MCPClient] = []
    contexts = []

    for client in [web_search_client, rag_client]:
        try:
            ctx = client.connect()
            connected = await ctx.__aenter__()
            contexts.append(ctx)
            connected_clients.append(connected)
            logger.info(
                "mcp.lifespan.server_connected",
                extra={
                    "event": "mcp.lifespan.server_connected",
                    "server": client.name,
                    "tools": connected.tool_names(),
                },
            )
        except Exception:
            logger.warning(
                "mcp.lifespan.server_unavailable",
                extra={
                    "event": "mcp.lifespan.server_unavailable",
                    "server": client.name,
                    "url": client.server_url,
                },
                exc_info=True,
            )

    app.state.mcp_clients = connected_clients if connected_clients else None
    logger.info(
        "mcp.lifespan.ready",
        extra={
            "event": "mcp.lifespan.ready",
            "connected_servers": len(connected_clients),
        },
    )

    yield

    for ctx in reversed(contexts):
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("mcp.lifespan.disconnect_error", exc_info=True)


app = FastAPI(lifespan=lifespan)
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
