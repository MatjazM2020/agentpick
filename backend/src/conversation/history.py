"""Format in-memory conversation turns for the orchestrator prompt."""

from __future__ import annotations

from src.conversation.store import Turn


def format_turns(turns: list[Turn]) -> str:
    """Render turns oldest-first as User/Assistant lines."""
    lines: list[str] = []
    for turn in turns:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)
