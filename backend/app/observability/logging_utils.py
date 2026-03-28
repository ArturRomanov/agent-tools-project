from __future__ import annotations

import logging
import re
from typing import Any

from app.observability.context import get_request_id

_BEARER_PATTERN = re.compile(r"bearer\s+[a-zA-Z0-9_\-\.=]+", re.IGNORECASE)
_API_KEY_PATTERN = re.compile(r"(api[_-]?key\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE)


def sanitize_text(text: str, mode: str, max_chars: int) -> str:
    if mode == "metadata":
        return ""

    cleaned = " ".join(text.split())
    cleaned = _BEARER_PATTERN.sub("bearer [REDACTED]", cleaned)
    cleaned = _API_KEY_PATTERN.sub(r"\1[REDACTED]", cleaned)

    if mode != "full" and len(cleaned) > max_chars:
        return f"{cleaned[:max_chars]}...[truncated]"
    return cleaned


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    request_id = get_request_id()
    extra = {"event": event, **fields}
    if request_id:
        extra["request_id"] = request_id
    logger.info(event, extra=extra)


def summarize_sources(sources: list[Any]) -> dict[str, Any]:
    domains = set()
    for source in sources:
        url = getattr(source, "url", "") or ""
        if "://" in url:
            host = url.split("://", maxsplit=1)[1].split("/", maxsplit=1)[0]
            if host:
                domains.add(host)
    return {"source_count": len(sources), "domains": sorted(domains)}
