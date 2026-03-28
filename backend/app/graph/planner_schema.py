from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PlannerDecision(BaseModel):
    action: Literal["call_tool", "final_answer"]
    tool_name: str | None = None
    tool_input: str | None = None
    final_answer: str | None = None
    reasoning_hint: str | None = None
