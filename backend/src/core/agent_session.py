"""
Microsoft Agent Framework session helpers.

Uses ``AgentSession`` for multi-turn LLM agents (Requirements Analyst, Ranker)
so conversation history is managed by the framework per docs/agent_patterns.py.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from src.core.state import RecommendationState

try:
    from agent_framework import AgentSession
except ImportError:  # pragma: no cover - optional in minimal test envs
    AgentSession = None  # type: ignore[misc, assignment]


def ensure_session_id(state: RecommendationState) -> str:
    """Assign a stable session id for this recommendation conversation."""
    if not state.agent_session_id:
        state.agent_session_id = str(uuid.uuid4())
    return state.agent_session_id


def load_session(state: RecommendationState) -> Any:
    """Restore or create an ``AgentSession`` from pipeline state."""
    if AgentSession is None:
        return None
    ensure_session_id(state)
    if state.agent_session_data:
        return AgentSession.from_dict(state.agent_session_data)
    return AgentSession(session_id=state.agent_session_id)


def save_session(state: RecommendationState, session: Any) -> None:
    """Persist ``AgentSession`` back into ``RecommendationState`` for the next turn."""
    if session is None or AgentSession is None:
        return
    state.agent_session_id = session.session_id or state.agent_session_id
    state.agent_session_data = session.to_dict()


def run_kwargs(state: RecommendationState) -> dict[str, Any]:
    """Keyword args for ``await agent.run(prompt, **run_kwargs(state))``."""
    session = load_session(state)
    return {"session": session} if session is not None else {}
