from __future__ import annotations

from app.memory.models import MemoryMode


class PolicyGuard:
    @staticmethod
    def normalize_mode(mode: str | None) -> MemoryMode:
        if mode in {"off", "session", "long_term"}:
            return mode
        return "off"

    @staticmethod
    def should_persist(mode: MemoryMode) -> bool:
        return mode in {"session", "long_term"}

    @staticmethod
    def should_retrieve_long_term(mode: MemoryMode) -> bool:
        return mode == "long_term"
