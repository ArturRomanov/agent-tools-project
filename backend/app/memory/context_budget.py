from __future__ import annotations

from app.memory.models import BudgetDecision, TurnRecord


class ContextBudgetManager:
    def __init__(self, context_limit_tokens: int, keep_recent_turns: int) -> None:
        self._context_limit_tokens = context_limit_tokens
        self._keep_recent_turns = keep_recent_turns

    @staticmethod
    def estimate_tokens(text: str) -> int:
        cleaned = text.strip()
        if not cleaned:
            return 0
        return max(1, len(cleaned) // 4)

    def evaluate(
        self,
        user_query: str,
        recent_turns: list[TurnRecord],
        retrieved_memory_text: str,
        summary_text: str,
    ) -> BudgetDecision:
        query_budget = self.estimate_tokens(user_query)
        turns_budget = sum(turn.token_estimate for turn in recent_turns)
        memory_budget = self.estimate_tokens(retrieved_memory_text)
        summary_budget = self.estimate_tokens(summary_text)
        total = query_budget + turns_budget + memory_budget + summary_budget
        over_budget = total > self._context_limit_tokens

        turns_to_summarize = 0
        if over_budget and len(recent_turns) > self._keep_recent_turns:
            turns_to_summarize = len(recent_turns) - self._keep_recent_turns

        return BudgetDecision(
            over_budget=over_budget,
            total_estimated_tokens=total,
            target_tokens=self._context_limit_tokens,
            turns_to_summarize=turns_to_summarize,
        )
