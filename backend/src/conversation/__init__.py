"""
Conversation memory for multi-turn chats.

Lightweight, in-memory sliding window per session. Recent turns are fed into
the orchestrator prompt so follow-up messages can be interpreted in context.
"""

from src.conversation.store import (
    ConversationStore,
    Turn,
    conversation_store,
    session_id_from_messages,
    WINDOW_TURNS,
)
from src.conversation.history import format_turns

__all__ = [
    "ConversationStore",
    "Turn",
    "conversation_store",
    "session_id_from_messages",
    "WINDOW_TURNS",
    "format_turns",
]
