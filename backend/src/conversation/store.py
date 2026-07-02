"""
In-memory conversation state, scoped per chat session.

Keeps a small sliding window of the most recent turns so the orchestrator can
interpret follow-up messages against prior context. Deliberately lightweight: a
process-local dict of bounded deques, no persistence. This lives outside the
retrieval/ranking pipeline so conversation memory stays a separate concern.
"""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List

WINDOW_TURNS = 4  # most recent messages (user/assistant) retained per session


@dataclass(frozen=True)
class Turn:
    """A single message in a conversation (one party, one message)."""

    role: str  # "user" or "assistant"
    content: str


class ConversationStore:
    """Per-session sliding window of recent turns (in-memory, thread-safe)."""

    def __init__(self, window_turns: int = WINDOW_TURNS) -> None:
        self._window = window_turns
        self._sessions: Dict[str, Deque[Turn]] = {}
        self._follow_ups: Dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def recent(self, session_id: str) -> List[Turn]:
        """Recent turns for a session, oldest first (empty if unknown session)."""
        with self._lock:
            turns = self._sessions.get(session_id)
            return list(turns) if turns else []

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a turn, evicting the oldest once the window is full."""
        content = (content or "").strip()
        if not content:
            return
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                turns = deque(maxlen=self._window)
                self._sessions[session_id] = turns
            turns.append(Turn(role=role, content=content))

    def clear(self, session_id: str) -> None:
        """Drop all stored turns for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._follow_ups.pop(session_id, None)

    def set_follow_ups(self, session_id: str, questions: list[str]) -> None:
        """Store clickable follow-ups from the latest recommendation turn."""
        cleaned = [q.strip() for q in questions if q and q.strip()]
        if not cleaned:
            return
        with self._lock:
            self._follow_ups[session_id] = cleaned[:5]

    def get_follow_ups(self, session_id: str) -> list[str]:
        """Follow-ups saved for a session (empty if none)."""
        with self._lock:
            return list(self._follow_ups.get(session_id, []))


# Process-wide store used by the pipeline.
conversation_store = ConversationStore()


def session_id_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Derive a stable session id for a chat conversation.

    OpenAI-style requests carry no session id and clients (e.g. Open WebUI)
    re-send the full history each turn, so we key the session on a hash of the
    first user message — stable across the turns of one conversation.
    """
    anchor = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            anchor = content if isinstance(content, str) else str(content)
            break
    digest = hashlib.sha1(anchor.strip().encode("utf-8")).hexdigest()
    return f"sess-{digest[:16]}"
