"""Helpers for Microsoft Agent Framework ``Agent.run`` return values."""

from typing import Any


def agent_run_text(result: Any) -> str:
    """
    Normalize ``await agent.run(...)`` output to a single string for JSON parsing.

    The framework returns ``AgentResponse`` (with a ``.text`` property), not a bare str.
    """
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    messages = getattr(result, "messages", None)
    if messages:
        parts: list[str] = []
        for msg in messages:
            fragment = getattr(msg, "text", None)
            if isinstance(fragment, str):
                parts.append(fragment)
        if parts:
            return "".join(parts)
    raise TypeError(
        f"agent.run() returned {type(result).__name__!r}; expected str or an object with .text"
    )
