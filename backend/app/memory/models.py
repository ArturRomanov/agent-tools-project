from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MemoryMode = Literal["off", "session", "long_term"]


@dataclass(frozen=True)
class TurnRecord:
    id: str
    session_id: str
    role: str
    content: str
    token_estimate: int
    sequence_no: int
    created_at: str
    archived: bool = False


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    session_id: str | None
    user_scope: str
    memory_type: str
    content: str
    salience: float
    confidence: float
    created_at: str
    is_pinned: bool = False


@dataclass(frozen=True)
class SessionSummaryRecord:
    id: str
    session_id: str
    summary_text: str
    covered_until_turn: int
    created_at: str
    quality_score: float


@dataclass(frozen=True)
class ContextPack:
    session_id: str
    memory_mode: MemoryMode
    user_scope: str
    context_text: str
    used_memory_item_ids: list[str] = field(default_factory=list)
    summary_used: bool = False
    checkpoint_state: dict[str, object] | None = None


@dataclass(frozen=True)
class PersistenceResult:
    checkpoint_id: str | None
    summarized: bool
    stored_memory_item_ids: list[str]


@dataclass(frozen=True)
class BudgetDecision:
    over_budget: bool
    total_estimated_tokens: int
    target_tokens: int
    turns_to_summarize: int
