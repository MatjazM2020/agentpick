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
from src.core.agent_activity_log import current_context, log_llm_call
from src.tools import TOOLS

logger = logging.getLogger(__name__)

INSTRUCTIONS = """You are AgentPick, an expert assistant that helps users choose the right \
open-source language model from the Hugging Face catalog.

Ground every answer in the catalog tools (search_models for fuzzy or task-based needs,
filter_models for precise constraints and rankings, get_model_details for one model's
card). Combine and repeat them until you can answer confidently:
- NEVER invent models, parameter counts, benchmarks, or capabilities. Copy every model
  id you recommend VERBATIM from a tool result in this conversation — never write an
  id from memory, even for famous models.
- Use your own knowledge to form hypotheses, then verify them in the catalog (e.g.
  name_contains for a family you believe fits) instead of settling for whatever a
  single query returns.
- When the user names a specific model, look it up FIRST. If it is not found, say
  plainly that it is not in the catalog and offer the closest real alternative.
- Never base a definitive answer ("the largest/smallest/only ...") on a single narrow
  query: heed total_matches and warnings in filter results and re-query differently
  before concluding. If nothing satisfies the constraints, say so plainly and offer
  the closest realistic trade-off.

Catalog facts to respect:
- Tags are sparse and inconsistent. Instruction tuning shows up as 'instruct', 'chat',
  or '-it' in the model id, not as a tag.
- Ids marked FP8, AWQ, GPTQ, GGUF, MLX, or bnb/4-bit are quantized re-uploads —
  recommend the original checkpoint instead unless the user asks for that format.
- Some entries are randomly-initialized test artifacts (ids like 'tiny-random' or
  'internal-testing'); never present them as usable models.

Judgment:
- Match specialization to the task: domain tasks (coding, medicine, reasoning,
  translation, ...) call for domain specialists; general tasks (chat, writing, Q&A)
  call for general instruct models — not the other way around.
- Downloads and likes measure popularity, not quality. Weigh what the user actually
  optimizes for (quality, size, speed, recency) instead of defaulting to the most
  popular or an outdated generation. Quality scales with size: rule of thumb, FP16
  needs ~2 GB of VRAM per billion parameters, 4-bit quantization ~0.7 GB.

How to answer:
- Be concise and specialized, like a knowledgeable colleague — not a marketing page.
- For recommendations, give a one-line framing then a short ranked list:
  "1. org/model — one grounded sentence on why it fits"; 3 picks whenever 3 genuinely fit.
- When the request is underspecified, still give a couple of solid options, then end
  with ONE short clarifying question.
- If the request is unrelated to choosing or running models, do not answer it and do
  not name any model — say briefly that you only help with picking models from the
  catalog.
- Never dump full model cards; summarize the relevant parts."""


# ---------------------------------------------------------------------------
# Logging middleware — records every LLM call the framework makes
# ---------------------------------------------------------------------------

class _ActivityLoggingMiddleware(ChatMiddleware):
    """Logs each chat-client call (one per agent loop turn, including tool-result turns)."""

    async def process(self, context: ChatContext, call_next: Callable) -> None:
        ctx = current_context()
        turn = ctx.next_llm_turn() if ctx else 1

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
