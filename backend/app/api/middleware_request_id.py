from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.context import clear_request_context, set_request_context
from app.observability.logging_utils import log_event

_LOGGER = logging.getLogger(__name__)
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        set_request_context(request_id, request.url.path, request.method)
        started_at = time.perf_counter()

        log_event(
            _LOGGER,
            "api.request.start",
            route=request.url.path,
            method=request.method,
        )

        try:
            response = await call_next(request)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            response.headers[REQUEST_ID_HEADER] = request_id
            log_event(
                _LOGGER,
                "api.request.end",
                route=request.url.path,
                method=request.method,
                duration_ms=duration_ms,
                status=response.status_code,
            )
            return response
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            log_event(
                _LOGGER,
                "api.request.error",
                route=request.url.path,
                method=request.method,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            clear_request_context()
