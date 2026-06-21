from __future__ import annotations

from typing import Literal

Freshness = Literal["auto", "any", "day", "week", "month"]
FreshnessBucket = Literal["any", "day", "week", "month"]


_DAY_HINTS = ("today", "breaking", "just in", "this morning", "now")
_WEEK_HINTS = ("recent", "latest", "news", "update", "updates", "this week")
_MONTH_HINTS = ("this month", "monthly")


def detect_freshness_bucket(query: str, freshness: Freshness = "auto") -> FreshnessBucket:
    if freshness != "auto":
        return freshness

    lowered = query.lower()
    if any(hint in lowered for hint in _DAY_HINTS):
        return "day"
    if any(hint in lowered for hint in _MONTH_HINTS):
        return "month"
    if any(hint in lowered for hint in _WEEK_HINTS):
        return "week"
    return "any"


def freshness_to_timelimit(bucket: FreshnessBucket) -> str | None:
    if bucket == "day":
        return "d"
    if bucket == "week":
        return "w"
    if bucket == "month":
        return "m"
    return None
