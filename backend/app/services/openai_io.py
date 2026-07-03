"""
OpenAI wire-format helpers.

Translates between the OpenAI chat API (what Open WebUI speaks) and the Agent
Framework: builds agent message inputs, streams responses as SSE chunks, and
shapes non-streaming completion payloads.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

from agent_framework import Message


def _text(content: Any) -> str:
    """Flatten OpenAI message content (str or multimodal parts) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content).strip() if content is not None else ""


def to_agent_messages(messages: list[dict]) -> list[Message]:
    """Map OpenAI user/assistant turns to Agent Framework messages.

    Client system prompts are dropped — AgentPick's own instructions define its
    behavior. Full history is passed each turn (Open WebUI is stateless).
    """
    out: list[Message] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text(m.get("content"))
        if text:
            out.append(Message(role=role, contents=[text]))
    return out


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _text(m.get("content"))
    return ""


def is_openwebui_task(text: str) -> bool:
    """Open WebUI background requests (title/tags/follow-up generation) start with '### Task:'."""
    return text.strip().lower().startswith("### task:")


def completion_response(text: str, completion_id: str, model: str, created: int) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunk(base: dict, delta: dict, finish=None) -> bytes:
    frame = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode("utf-8")


async def sse_from_stream(
    text_chunks: AsyncIterator[str], completion_id: str, model: str, created: int
) -> AsyncIterator[bytes]:
    """Wrap an async text stream as OpenAI chat.completion.chunk SSE frames."""
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _chunk(base, {"role": "assistant", "content": ""})
    async for piece in text_chunks:
        if piece:
            yield _chunk(base, {"content": piece})
    yield _chunk(base, {}, finish="stop")
    yield b"data: [DONE]\n\n"


async def sse_from_text(
    text: str, completion_id: str, model: str, created: int
) -> AsyncIterator[bytes]:
    """SSE stream for a single pre-computed message (used for background tasks)."""

    async def _one() -> AsyncIterator[str]:
        yield text

    async for frame in sse_from_stream(_one(), completion_id, model, created):
        yield frame
