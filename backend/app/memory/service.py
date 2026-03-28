from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.config.settings import Settings, get_settings
from app.memory.checkpoint_manager import CheckpointManager
from app.memory.context_budget import ContextBudgetManager
from app.memory.extractor import MemoryExtractor
from app.memory.long_term import LongTermMemoryStore
from app.memory.models import ContextPack, MemoryMode, PersistenceResult
from app.memory.policy import PolicyGuard
from app.memory.retriever import SessionSummaryStore
from app.memory.scorer import MemoryScorer
from app.memory.session_manager import SessionManager
from app.memory.summarizer import SessionSummarizer
from app.memory.turn_store import TurnStore
from app.storage import SQLiteStore


class MemoryService:
    def __init__(
        self,
        settings: Settings | None = None,
        sqlite_store: SQLiteStore | None = None,
        long_term_store: LongTermMemoryStore | None = None,
        summarizer: SessionSummarizer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = sqlite_store or SQLiteStore(self._settings)
        self._store.initialize()

        self._sessions = SessionManager(self._store)
        self._turns = TurnStore(self._store)
        self._summaries = SessionSummaryStore(self._store)
        self._checkpoints = CheckpointManager(self._store)
        self._summarizer = summarizer or SessionSummarizer()
        self._extractor = MemoryExtractor()
        self._scorer = MemoryScorer()
        self._budget = ContextBudgetManager(
            context_limit_tokens=self._settings.memory_context_limit_tokens,
            keep_recent_turns=self._settings.memory_keep_recent_turns,
        )
        self._long_term = long_term_store or LongTermMemoryStore(
            sqlite_store=self._store,
            settings=self._settings,
        )

    async def prepare_context(
        self,
        query: str,
        session_id: str | None,
        memory_mode: str | None,
        checkpoint_id: str | None,
        user_scope: str = "default",
        request_id: str | None = None,
    ) -> ContextPack:
        normalized_mode: MemoryMode = PolicyGuard.normalize_mode(memory_mode)
        if not PolicyGuard.should_persist(normalized_mode):
            resolved_session_id = session_id or str(uuid4())
            return ContextPack(
                session_id=resolved_session_id,
                memory_mode=normalized_mode,
                user_scope=user_scope,
                context_text="",
            )

        resolved_session_id = self._sessions.get_or_create(
            session_id=session_id,
            user_scope=user_scope,
            memory_mode_default=normalized_mode,
        )

        recent_turns = self._turns.recent_turns(
            session_id=resolved_session_id,
            limit=self._settings.memory_recent_turn_window,
        )

        summary_record = self._summaries.latest(resolved_session_id)
        summary_text = summary_record.summary_text if summary_record else ""

        used_memory_item_ids: list[str] = []
        long_term_lines: list[str] = []
        if PolicyGuard.should_retrieve_long_term(normalized_mode):
            try:
                retrieved = await self._long_term.retrieve(
                    query=query,
                    user_scope=user_scope,
                    limit=self._settings.memory_top_k,
                )
                for memory, score in retrieved:
                    used_memory_item_ids.append(memory.id)
                    long_term_lines.append(f"- ({memory.memory_type}) {memory.content}")
                    self._long_term.log_access(
                        session_id=resolved_session_id,
                        request_id=request_id,
                        memory_item_id=memory.id,
                        score=score,
                        selected=True,
                    )
            except Exception:
                # Degrade gracefully to session-only memory.
                long_term_lines = []

        recent_lines = [f"- {turn.role}: {turn.content}" for turn in recent_turns]
        context_sections: list[str] = []
        if summary_text:
            context_sections.append("Session summary:\n" + summary_text)
        if recent_lines:
            context_sections.append("Recent turns:\n" + "\n".join(recent_lines))
        if long_term_lines:
            context_sections.append("Relevant long-term memory:\n" + "\n".join(long_term_lines))

        checkpoint_state = None
        if checkpoint_id:
            checkpoint_state = self._checkpoints.load_checkpoint(checkpoint_id)

        return ContextPack(
            session_id=resolved_session_id,
            memory_mode=normalized_mode,
            user_scope=user_scope,
            context_text="\n\n".join(context_sections).strip(),
            used_memory_item_ids=used_memory_item_ids,
            summary_used=bool(summary_text),
            checkpoint_state=checkpoint_state,
        )

    async def persist_after_run(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        memory_mode: str,
        user_scope: str,
        graph_state: dict[str, object],
    ) -> PersistenceResult:
        normalized_mode: MemoryMode = PolicyGuard.normalize_mode(memory_mode)
        if not PolicyGuard.should_persist(normalized_mode):
            return PersistenceResult(
                checkpoint_id=None,
                summarized=False,
                stored_memory_item_ids=[],
            )

        user_turn_id = self._turns.add_turn(session_id=session_id, role="user", content=user_query)
        self._turns.add_turn(session_id=session_id, role="assistant", content=assistant_answer)

        summary_text = self._summaries.latest(session_id)
        summary_content = summary_text.summary_text if summary_text else ""
        active_turns = self._turns.recent_turns(
            session_id,
            self._settings.memory_recent_turn_window,
        )
        budget_decision = self._budget.evaluate(
            user_query=user_query,
            recent_turns=active_turns,
            retrieved_memory_text="",
            summary_text=summary_content,
        )

        summarized = False
        if budget_decision.turns_to_summarize > 0:
            candidate_turns = self._turns.oldest_active_turns(
                session_id=session_id,
                limit=budget_decision.turns_to_summarize,
            )
            summarized_text = await self._summarizer.summarize_turns(candidate_turns)
            if summarized_text.strip():
                with self._store.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO session_summaries (
                            id,
                            session_id,
                            summary_text,
                            covered_until_turn,
                            created_at,
                            quality_score
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid4()),
                            session_id,
                            summarized_text,
                            max(turn.sequence_no for turn in candidate_turns),
                            datetime.now(UTC).isoformat(),
                            0.75,
                        ),
                    )
                self._turns.archive_turns([turn.id for turn in candidate_turns])
                summarized = True

        stored_memory_ids: list[str] = []
        if PolicyGuard.should_retrieve_long_term(normalized_mode):
            for memory_type, content in self._extractor.extract(user_query, assistant_answer):
                salience, confidence = self._scorer.score(memory_type, content)
                try:
                    memory_id = await self._long_term.store_memory(
                        session_id=session_id,
                        user_scope=user_scope,
                        memory_type=memory_type,
                        content=content,
                        salience=salience,
                        confidence=confidence,
                        source_turn_id=user_turn_id,
                    )
                    stored_memory_ids.append(memory_id)
                except Exception:
                    # Keep chat response successful when memory indexing fails.
                    continue

        checkpoint_id = self._checkpoints.save_checkpoint(
            session_id=session_id,
            run_id=str(uuid4()),
            graph_node="finalize",
            state=graph_state,
        )

        return PersistenceResult(
            checkpoint_id=checkpoint_id,
            summarized=summarized,
            stored_memory_item_ids=stored_memory_ids,
        )
