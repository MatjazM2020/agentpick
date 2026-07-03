"""
The AgentPick recommendation agent.

A single conversational agent (Microsoft Agent Framework) with catalog tools.
The framework's function-calling loop is the agentic core: the model decides
when to search, filter, or read a model card, iterates on the results, and then
writes a specialized answer. See docs/agent_patterns.py (Patterns 1, 2, 5).
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Callable, Optional

from agent_framework import Agent, ChatContext, ChatMiddleware, FunctionInvocationConfiguration
from agent_framework.openai import OpenAIChatClient

from src.core import config
from src.core.agent_activity_log import log_llm_call
from src.tools import TOOLS

logger = logging.getLogger(__name__)

INSTRUCTIONS = """You are AgentPick, an expert assistant that helps users choose the right \
open-source language model from the Hugging Face catalog.

You have tools that read the real catalog. Use them to ground every recommendation:
- search_models: semantic search for fuzzy or task-based needs ("summarize contracts").
- filter_models: precise constraints and rankings (parameter size, pipeline type, a tag,
  a name substring like "instruct"/"coder", most downloaded, smallest, largest).
- get_model_details: read a specific model's full card to compare or verify before recommending.

How to work:
- Pick the right tool for the request. Combine tools when it helps (e.g. filter to a
  shortlist, then read one or two cards before deciding). You may call tools several times.
- NEVER invent models, parameter counts, benchmarks, or capabilities. Only state facts that
  appear in tool results. Always use exact model ids returned by the tools.
- If a structured query returns nothing, say plainly that no catalog model satisfies the
  constraints, explain why, and offer the closest realistic trade-off.

How to answer:
- Be concise and specialized, like a knowledgeable colleague — not a marketing page.
- For recommendations, give a one-line framing then a short ranked list:
  "1. org/model — one grounded sentence on why it fits". Usually 1-3 picks.
- When the request is underspecified (e.g. "best model for a student"), still give a couple
  of solid options, then end with ONE short clarifying question to narrow it down.
- For general questions about choosing or running models, answer directly; only call tools
  when catalog data actually helps.
- Never dump full model cards; summarize the relevant parts."""


# ---------------------------------------------------------------------------
# Logging middleware — records every LLM call the framework makes
# ---------------------------------------------------------------------------

class _ActivityLoggingMiddleware(ChatMiddleware):
    """Logs each chat-client call (one per agent loop turn, including tool-result turns)."""

    def __init__(self) -> None:
        self._turn_counter: dict[int, int] = {}

    async def process(self, context: ChatContext, call_next: Callable) -> None:
        task_id = id(context)
        turn = self._turn_counter.get(task_id, 0) + 1
        self._turn_counter[task_id] = turn

        # Rough token estimate from message text lengths (~4 chars per token)
        total_chars = sum(
            len(getattr(m, "text", "") or "")
            for m in (context.messages or [])
        )
        est_tokens = total_chars // 4 or None

        t0 = time.monotonic()
        await call_next()
        elapsed_ms = (time.monotonic() - t0) * 1000

        log_llm_call(turn, est_tokens)
        logger.debug("[agent] LLM turn=%d, est_input_tokens=%s, %.0fms", turn, est_tokens, elapsed_ms)


_middleware = _ActivityLoggingMiddleware()


# ---------------------------------------------------------------------------
# Client and agent (process-wide singletons)
# ---------------------------------------------------------------------------

_client: Optional[OpenAIChatClient] = None
_agent: Optional[Agent] = None
_plain_agent: Optional[Agent] = None


def get_client() -> OpenAIChatClient:
    """Process-wide OpenAI chat client with a bounded tool-calling loop."""
    global _client
    if _client is not None:
        return _client
    kwargs: dict = {
        "model": config.CHAT_MODEL,
        "function_invocation_configuration": FunctionInvocationConfiguration(
            max_iterations=config.MAX_TOOL_ITERATIONS
        ),
        "middleware": [_middleware],
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    try:
        _client = OpenAIChatClient(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize OpenAI client: {e}. Set OPENAI_API_KEY "
            "(and optionally OPENAI_CHAT_MODEL_ID / OPENAI_BASE_URL)."
        ) from e
    return _client


def get_agent() -> Agent:
    """The recommendation agent (tools + specialized instructions)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            client=get_client(),
            name="AgentPick",
            instructions=INSTRUCTIONS,
            tools=TOOLS,
        )
    return _agent


def _plain() -> Agent:
    """A tool-less agent for Open WebUI background tasks (titles, tags, follow-ups)."""
    global _plain_agent
    if _plain_agent is None:
        _plain_agent = Agent(client=get_client(), name="AgentPickTasks")
    return _plain_agent


# ---------------------------------------------------------------------------
# Request runners
# ---------------------------------------------------------------------------

async def stream_reply(messages) -> AsyncIterator[str]:
    """Yield text chunks of the agent's answer as they are generated."""
    async for update in get_agent().run(messages, stream=True):
        if update.text:
            yield update.text


async def complete_reply(messages) -> str:
    """Full agent answer (non-streaming)."""
    result = await get_agent().run(messages)
    return (result.text or "").strip()


async def complete_task(messages) -> str:
    """Answer an Open WebUI background task with a plain (tool-less) completion."""
    result = await _plain().run(messages)
    return (result.text or "").strip()
