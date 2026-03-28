from __future__ import annotations

from app.graph.state import AgentState


async def finalize_answer(state: AgentState) -> AgentState:
    return {
        "should_continue": False,
        "final_answer": state.get("final_answer"),
    }
