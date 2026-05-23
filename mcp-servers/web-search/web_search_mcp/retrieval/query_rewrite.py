from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def rewrite_query(query: str) -> str:
    """Apply lightweight deterministic rewrite rules for search quality."""
    cleaned = _WHITESPACE_RE.sub(" ", query.strip())
    if not cleaned:
        return cleaned

    lowered_tokens = [token.lower() for token in _WORD_RE.findall(cleaned)]
    if not lowered_tokens:
        return cleaned

    # Common wording fix: singular "new" in information-seeking queries.
    if "new" in lowered_tokens and "news" not in lowered_tokens:
        cleaned = re.sub(r"\bnew\b", "news", cleaned, flags=re.IGNORECASE)

    return cleaned
