from __future__ import annotations

import contextlib
import logging
from contextlib import AbstractContextManager

from app.config.settings import Settings

logger = logging.getLogger(__name__)

_langfuse_client: object | None = None
_langfuse_available: bool = False


def init_langfuse(settings: Settings) -> bool:
    """Initialize the Langfuse singleton client.

    Returns True if successfully initialized, False otherwise.
    """
    global _langfuse_client, _langfuse_available

    if not settings.langfuse_enabled:
        logger.info("langfuse.init.disabled")
        _langfuse_available = False
        return False

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.warning(
            "langfuse.init.missing_keys",
            extra={
                "event": "langfuse.init.missing_keys",
                "detail": "LANGFUSE_ENABLED=true but secret/public keys are empty",
            },
        )
        _langfuse_available = False
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        _langfuse_available = True
        logger.info(
            "langfuse.init.success",
            extra={
                "event": "langfuse.init.success",
                "host": settings.langfuse_host,
            },
        )
        return True
    except Exception:
        logger.warning("langfuse.init.failed", exc_info=True)
        _langfuse_client = None
        _langfuse_available = False
        return False


def create_langfuse_handler(
    trace_id: str | None = None,
):
    """Create a per-request CallbackHandler for LangChain.

    Returns None if Langfuse is not available.
    """
    if not _langfuse_available or _langfuse_client is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        kwargs: dict = {}
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id.replace("-", "")}
        handler = CallbackHandler(**kwargs)
        return handler
    except Exception:
        logger.warning("langfuse.handler.creation_failed", exc_info=True)
        return None


def langfuse_attributes(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> AbstractContextManager:
    """Return a context manager that propagates trace-level attributes.

    Returns a no-op context when Langfuse is not available.
    """
    if not _langfuse_available:
        return contextlib.nullcontext()

    try:
        from langfuse import propagate_attributes

        kwargs: dict = {}
        if session_id is not None:
            kwargs["session_id"] = session_id
        if user_id is not None:
            kwargs["user_id"] = user_id
        if tags is not None:
            kwargs["tags"] = tags
        if metadata is not None:
            kwargs["metadata"] = metadata
        return propagate_attributes(**kwargs)
    except Exception:
        logger.warning("langfuse.propagate_attributes.failed", exc_info=True)
        return contextlib.nullcontext()


def shutdown_langfuse() -> None:
    """Flush pending events and shutdown the Langfuse client."""
    global _langfuse_client, _langfuse_available

    if _langfuse_client is None:
        return

    try:
        _langfuse_client.flush()
        _langfuse_client.shutdown()
        logger.info("langfuse.shutdown.success")
    except Exception:
        logger.warning("langfuse.shutdown.error", exc_info=True)
    finally:
        _langfuse_client = None
        _langfuse_available = False
