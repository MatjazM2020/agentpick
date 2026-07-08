"""The systems under evaluation.

Two systems:

- ``agent``     — the full AgentPick system (LLM orchestrator + catalog tools).
- ``llm_only``  — the same LLM without any catalog access, answering from
                  parametric knowledge (baseline).

Each system is an async ``messages -> answer text`` callable, where
``messages`` is an OpenAI-style list of ``{"role", "content"}`` dicts (one
user turn for most questions, a full dialogue for multi-turn questions), so
the runner can score all of them with the same extraction and metrics
pipeline.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from agent_framework import Agent, Message

from src.agent import complete_reply, get_client, request_scope

LLM_ONLY_INSTRUCTIONS = """You are an expert assistant that helps users choose the right \
open-source language model from the Hugging Face hub. You have NO catalog or search access; \
answer from your own knowledge.

- Only recommend open-weight models whose weights are downloadable from the Hugging Face hub \
(e.g. Llama, Qwen, Mistral, Gemma, DeepSeek, Phi families). Never recommend closed, \
API-only models (GPT-4/o-series, Claude, Gemini).
- Prefer popular, widely downloaded models (roughly 10k+ downloads) over obscure ones, and \
only name models you are confident actually exist on the hub.
- For recommendations, give a short ranked list: "1. org/model — one sentence on why it fits". \
Usually 1-3 picks. Always use exact Hugging Face model ids ("org/model").
- If the request is underspecified, still give a couple of solid options, then end with ONE \
short clarifying question.
- If no model can satisfy the constraints, say so plainly."""


_llm_only_agent: Optional[Agent] = None


def _llm_only() -> Agent:
    global _llm_only_agent
    if _llm_only_agent is None:
        _llm_only_agent = Agent(
            client=get_client(),
            name="LLMOnlyBaseline",
            instructions=LLM_ONLY_INSTRUCTIONS,
        )
    return _llm_only_agent


def _to_agent_messages(messages: list[dict]) -> list[Message]:
    return [Message(role=m["role"], contents=[m["content"]]) for m in messages]


async def run_agent(messages: list[dict]) -> str:
    """Full AgentPick system (tools + catalog)."""
    return await complete_reply(_to_agent_messages(messages))


async def run_llm_only(messages: list[dict]) -> str:
    """Baseline: the same LLM without catalog tools."""
    agent_messages = _to_agent_messages(messages)
    with request_scope(agent_messages):
        result = await _llm_only().run(agent_messages)
    return (result.text or "").strip()


SYSTEMS: dict[str, Callable[[list[dict]], Awaitable[str]]] = {
    "agent": run_agent,
    "llm_only": run_llm_only,
}
