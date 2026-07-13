"""The systems under evaluation.

Four systems, ordered from most to least agentic:

- ``agent``        — the full AgentPick system: LLM orchestrator + catalog
                     tools in an open-ended loop (the model reacts to tool
                     results and decides what to run next).
- ``single_round`` — LLM-parameterized retrieval without a loop, as a fixed
                     code pipeline: one planning completion translates the
                     request into query parameters (JSON), code runs exactly
                     one structured filter (PostgreSQL) and one semantic
                     search (Qdrant) with them, and a second completion
                     answers from those results. The LLM adapts the queries
                     to the request but never sees results before they are
                     final — the agent − single_round delta isolates the
                     agentic loop itself.
- ``qdrant_only``  — fixed vanilla RAG: code (not the LLM) embeds the user's
                     words, retrieves top-k from Qdrant, and the LLM answers
                     from those results in one completion. No structured
                     store, no adaptivity.
- ``llm_only``     — the same LLM without any catalog access, answering from
                     parametric knowledge.

Each system is an async ``messages -> answer text`` callable, where
``messages`` is an OpenAI-style list of ``{"role", "content"}`` dicts (one
user turn for most questions, a full dialogue for multi-turn questions), so
the runner can score all of them with the same extraction and metrics
pipeline.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable, Optional

from agent_framework import Agent, Message

from src import catalog
from src.agent import complete_reply, get_client, request_scope
from src.core.agent_activity_log import current_context, log_tool_call

TOP_K = 8  # same as the agent's search_models / filter_models default limit

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
    with request_scope(agent_messages, system="llm_only"):
        result = await _llm_only().run(agent_messages)
    return (result.text or "").strip()


# ---------------------------------------------------------------------------
# Shared retrieval plumbing for the non-agentic catalog baselines
# ---------------------------------------------------------------------------

def _joined_user_turns(messages: list[dict]) -> str:
    """Fallback retrieval query: all user turns joined, decided by code."""
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


async def _run_query(name: str, log_args: dict, fn):
    """Run one code-decided retrieval in a thread, logged like an agent tool
    call. ``fn`` is a no-arg blocking callable returning either a list of
    model dicts or a filter_models-style dict with a "models" list. Returns
    None on error."""
    ctx = current_context()
    label = ctx.next_tool(name) if ctx else name
    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(fn)
    except Exception as e:
        log_tool_call(label, log_args, None, (time.monotonic() - t0) * 1000, error=str(e))
        return None
    models = result["models"] if isinstance(result, dict) else result
    log_tool_call(label, log_args, len(models), (time.monotonic() - t0) * 1000)
    return result


def _context_block(payload) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False) if payload else "(no results)"


_ANSWER_RULES = """You cannot run further queries; ground your answer in these results only:
- NEVER invent models, parameter counts, benchmarks, or capabilities. Copy every model
  id you recommend VERBATIM from the results below — never write an id from memory,
  even for famous models.
- For recommendations, give a one-line framing then a short ranked list:
  "1. org/model — one grounded sentence on why it fits"; 3 picks whenever 3 genuinely fit.
- If none of the results satisfies the user's constraints, say so plainly — do not
  recommend a model that only partially fits.
- If the request is underspecified, the LAST sentence of the reply MUST be ONE direct
  clarifying question ending in "?".
- If the request is unrelated to choosing or running models, do not answer it and do
  not name any model — say briefly that you only help with picking models from the
  catalog."""


# ---------------------------------------------------------------------------
# single_round — LLM-parameterized retrieval, fixed plan → fetch → answer pipeline
# ---------------------------------------------------------------------------

SINGLE_ROUND_PLAN_INSTRUCTIONS = """You translate a user's request for an open-source \
language model into catalog query parameters. Reply with ONLY a JSON object — no prose, \
no code fences:

{"search_query": "<text for a semantic search over model cards>",
 "filter": {"task_type": "<pipeline tag, e.g. text-generation, or null>",
            "name_contains": "<substring of the model id, or null>",
            "min_params_b": <number or null>,
            "max_params_b": <number or null>,
            "sort_by": "<one of: downloads, likes, smallest, largest, newest>"}}

Guidance:
- Instruction tuning shows up as 'instruct', 'chat', or '-it' in the model id, not as a
  tag — use name_contains to require it.
- Respect explicit size windows: "at least 7B" means min_params_b=7; "under 4B" means
  max_params_b=4.
- Superlatives (largest, smallest, most efficient, ...) need sort_by=largest or
  sort_by=smallest plus the relevant filters; otherwise prefer sort_by=downloads.
- In a dialogue, parameterize the user's current need in light of all previous turns."""

SINGLE_ROUND_INSTRUCTIONS = """You are an expert assistant that helps users choose the right \
open-source language model from the Hugging Face catalog.

Below are the results of two catalog queries whose parameters were chosen for this
request: a structured metadata filter and a semantic search. """ + _ANSWER_RULES + """

Catalog context:
"""

_PLAN_SORTS = ("downloads", "likes", "smallest", "largest", "newest")

_planner_agent: Optional[Agent] = None


def _planner() -> Agent:
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = Agent(
            client=get_client(),
            name="SingleRoundPlanner",
            instructions=SINGLE_ROUND_PLAN_INSTRUCTIONS,
        )
    return _planner_agent


def _parse_plan(text: str) -> dict:
    """The planner's JSON reply as a dict ({} when unparseable)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        plan = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return plan if isinstance(plan, dict) else {}


def _filter_kwargs(plan: dict) -> dict:
    """Validated catalog.filter_models kwargs from the plan (empty = defaults)."""
    raw = plan.get("filter")
    if not isinstance(raw, dict):
        return {}
    kwargs: dict = {}
    for key in ("task_type", "name_contains"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            kwargs[key] = value.strip()
    for key in ("min_params_b", "max_params_b"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            kwargs[key] = float(value)
    sort_by = raw.get("sort_by")
    if isinstance(sort_by, str) and sort_by.strip().lower() in _PLAN_SORTS:
        kwargs["sort_by"] = sort_by.strip().lower()
    return kwargs


async def run_single_round(messages: list[dict]) -> str:
    """Baseline: LLM-parameterized retrieval without a loop — one planning
    completion chooses the query parameters, code runs one filter and one
    search with them, and a second completion answers from the results."""
    agent_messages = _to_agent_messages(messages)
    with request_scope(agent_messages, system="single_round"):
        plan_text = (await _planner().run(agent_messages)).text or ""
        plan = _parse_plan(plan_text)
        query = plan.get("search_query")
        if not (isinstance(query, str) and query.strip()):
            query = _joined_user_turns(messages)
        filter_kwargs = _filter_kwargs(plan)
        searched, filtered = await asyncio.gather(
            _run_query(
                "search_models",
                {"query": query, "limit": TOP_K},
                lambda: catalog.semantic_search(query, TOP_K),
            ),
            _run_query(
                "filter_models",
                {**filter_kwargs, "limit": TOP_K},
                lambda: catalog.filter_models(limit=TOP_K, **filter_kwargs),
            ),
        )
        context = (
            "Structured filter results:\n" + _context_block(filtered)
            + "\n\nSemantic search results:\n" + _context_block(searched)
        )
        answerer = Agent(
            client=get_client(),
            name="SingleRoundBaseline",
            instructions=SINGLE_ROUND_INSTRUCTIONS + context,
        )
        result = await answerer.run(agent_messages)
    return (result.text or "").strip()


# ---------------------------------------------------------------------------
# qdrant_only — fixed vanilla RAG, zero adaptivity
# ---------------------------------------------------------------------------

QDRANT_ONLY_INSTRUCTIONS = """You are an expert assistant that helps users choose the right \
open-source language model from the Hugging Face catalog.

Below are the results of one semantic search over the user's words in the model
catalog. """ + _ANSWER_RULES + """

Search results:
"""


async def run_qdrant_only(messages: list[dict]) -> str:
    """Baseline: fixed vanilla RAG — one code-decided semantic search, one completion."""
    agent_messages = _to_agent_messages(messages)
    with request_scope(agent_messages, system="qdrant_only"):
        query = _joined_user_turns(messages)
        searched = await _run_query(
            "search_models",
            {"query": query, "limit": TOP_K},
            lambda: catalog.semantic_search(query, TOP_K),
        )
        agent = Agent(
            client=get_client(),
            name="QdrantOnlyBaseline",
            instructions=QDRANT_ONLY_INSTRUCTIONS + _context_block(searched),
        )
        result = await agent.run(agent_messages)
    return (result.text or "").strip()


SYSTEMS: dict[str, Callable[[list[dict]], Awaitable[str]]] = {
    "agent": run_agent,
    "llm_only": run_llm_only,
    "single_round": run_single_round,
    "qdrant_only": run_qdrant_only,
}
