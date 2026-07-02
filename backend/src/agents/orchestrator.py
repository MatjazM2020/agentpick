"""
Orchestrator — single tool-using agent with a bounded LLM–tool loop.

Interprets user intent (including multi-turn follow-ups), searches the catalog,
and finalizes recommendations. Requirements extraction and query contextualization
are folded into this agent rather than separate pipeline stages.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Annotated, Any, Optional

from agent_framework import Agent, tool
from pydantic import Field

from src.agents import ranker, retriever
from src.conversation.history import format_turns
from src.conversation.store import Turn
from src.core import postgres
from src.core.agent_factory import get_chat_client
from src.core.agent_output import agent_run_text
from src.core.agent_session import run_kwargs, save_session
from src.core.agent_activity_log import log_activity
from src.core.config import AgentConfig, RankerConfig, RetrieverConfig
from src.core.state import RecommendationState

logger = logging.getLogger(__name__)

_client_config_lock = threading.Lock()

_ORCHESTRATOR_INSTRUCTIONS = """You are the Orchestrator for an ML model recommendation system on Hugging Face.

You help users find language models by calling tools. You may also reply in plain text
when clarification or a redirect is needed — do not call tools in that case.

Tools:
- search_models: semantic search in Qdrant (task/domain queries; supports hybrid popularity thresholds)
- get_popular_models: top models by downloads/likes from PostgreSQL (pure popularity queries)
- finalize_recommendations: hard-filter, re-rank, and explain the current candidate pool

Intent and conversation:
- Use the recent conversation when the latest message is a follow-up (e.g. "make it faster",
  "what about that model") — fold prior task/domain/constraints into your search query.
- Extract task type, domain, hardware, license, and popularity thresholds from the request
  and pass them as tool arguments. Do not invent constraints the user did not mention.
- Off-topic (greetings, chit-chat): reply in plain text only; invite the user to describe models.
- Underspecified but in-domain: prefer search_models with a best-guess query over blocking.

Routing:
- Pure popularity ("most downloaded", "top liked"): get_popular_models.
- Task + popularity thresholds (e.g. popular summarization model with >1M downloads):
  search_models with min_downloads/min_likes (hybrid enrich runs automatically).
- Empty or weak search results: retry search_models once with relax_filters=true or a broader query.
- When the candidate pool looks sufficient, call finalize_recommendations exactly once before finishing.

Recommendation count (top_k):
- Default: call finalize_recommendations once — the system returns 3 options.
- Single pick: when the user asks for one recommendation, the best choice among prior
  options, or a follow-up like "which of those is best?", call finalize_recommendations
  once — the system returns 1 pick automatically.

Never invent model capabilities — tools return real catalog data only."""


def _summarize_candidates(models: list[dict], limit: int = 5) -> dict[str, Any]:
    return {
        "count": len(models),
        "top": [
            {"model_id": m["id"], "score": round(float(m.get("score", 0)), 4)}
            for m in models[:limit]
            if m.get("id")
        ],
    }


def _apply_popularity_thresholds(
    state: RecommendationState,
    min_downloads: Optional[int],
    min_likes: Optional[int],
) -> None:
    if min_downloads is None and min_likes is None:
        return
    state.popularity = {
        "mode": "hybrid",
        "sort_by": (state.popularity or {}).get("sort_by"),
        "min_downloads": min_downloads,
        "min_likes": min_likes,
    }


def build_tools(
    state: RecommendationState,
    agents: dict,
    ranker_config: RankerConfig,
    retriever_config: RetrieverConfig,
    activity: Optional[dict[str, int]] = None,
) -> tuple[list[Any], dict[str, bool]]:
    """Build per-request tools that mutate ``state``. Returns (tools, status_flags)."""
    status = {"finalized": False}
    default_top_k = max(1, min(5, int(state.recommendation_top_k or 3)))
    if activity is None:
        activity = {"tool_calls": 0}

    def _record_tool(name: str, detail: str = "") -> None:
        activity["tool_calls"] = activity.get("tool_calls", 0) + 1
        n = activity["tool_calls"]
        suffix = f" | {detail}" if detail else ""
        log_activity(f"tool #{n}: {name}{suffix}")

    @tool(approval_mode="never_require")
    def search_models(
        query: Annotated[str, Field(description="Standalone embedding search query")],
        relax_filters: Annotated[
            bool, Field(description="Broaden Qdrant search after weak or empty results")
        ] = False,
        task_type: Annotated[
            Optional[str], Field(description="Optional pipeline_tag hint, e.g. summarization")
        ] = None,
        license: Annotated[
            Optional[str], Field(description="Optional license filter, e.g. apache-2.0")
        ] = None,
        domain: Annotated[
            Optional[str], Field(description="Optional domain, e.g. legal, mathematics")
        ] = None,
        hardware: Annotated[
            Optional[str], Field(description="Optional hardware, e.g. cpu_only, gpu")
        ] = None,
        min_downloads: Annotated[
            Optional[int], Field(description="Minimum downloads (hybrid filter after search)")
        ] = None,
        min_likes: Annotated[
            Optional[int], Field(description="Minimum likes (hybrid filter after search)")
        ] = None,
    ) -> str:
        """Semantic search over Hugging Face models in Qdrant."""
        _record_tool("search_models", f"query={query.strip()[:80]!r}")
        state.intent_summary = query.strip()
        if task_type:
            state.task_type = task_type
        if license:
            state.constraints["license"] = license
        if domain:
            state.constraints["domain"] = domain
        if hardware:
            state.constraints["hardware"] = hardware
        _apply_popularity_thresholds(state, min_downloads, min_likes)
        state.requirements_extracted = True

        try:
            retriever.run(state, retriever_config, refine=relax_filters)
        except Exception as e:
            logger.error(f"[Orchestrator] search_models failed: {e}")
            state.agent_logs.append(f"Orchestrator tool search_models error: {e}")
            return json.dumps({"error": str(e), "count": 0})

        pop = state.popularity or {}
        if (
            pop.get("mode") == "hybrid"
            or pop.get("min_downloads")
            or pop.get("min_likes")
        ):
            postgres.enrich_and_filter(state)

        state.agent_logs.append(
            f"Orchestrator tool search_models: {len(state.retrieved_models)} candidates"
        )
        return json.dumps(_summarize_candidates(state.retrieved_models))

    @tool(approval_mode="never_require")
    def get_popular_models(
        sort_by: Annotated[str, Field(description="downloads or likes")] = "downloads",
        min_downloads: Annotated[Optional[int], Field(description="Minimum downloads")] = None,
        min_likes: Annotated[Optional[int], Field(description="Minimum likes")] = None,
        task_type: Annotated[
            Optional[str], Field(description="Optional pipeline_tag filter")
        ] = None,
        limit: Annotated[int, Field(description="Max candidates to return", ge=1, le=50)] = 30,
    ) -> str:
        """Fetch top models from PostgreSQL ordered by popularity."""
        _record_tool("get_popular_models", f"sort_by={sort_by}")
        state.popularity = {
            "mode": "popularity_only",
            "sort_by": sort_by,
            "min_downloads": min_downloads,
            "min_likes": min_likes,
        }
        if task_type:
            state.task_type = task_type
        state.requirements_extracted = True

        try:
            candidates = postgres.query_top_models(
                task_type=task_type or state.task_type,
                tags=state.constraints.get("tags"),
                sort_by=sort_by,
                min_downloads=min_downloads,
                min_likes=min_likes,
                limit=min(limit, retriever_config.top_k_models),
            )
        except postgres.PostgresUnavailable as e:
            logger.warning(f"[Orchestrator] get_popular_models unavailable: {e}")
            state.agent_logs.append(f"Orchestrator tool get_popular_models unavailable: {e}")
            return json.dumps({"error": str(e), "count": 0})

        state.retrieved_models = candidates
        state.retrieval_complete = True
        state.agent_logs.append(
            f"Orchestrator tool get_popular_models: {len(candidates)} candidates"
        )
        return json.dumps(_summarize_candidates(candidates))

    async def _finalize(top_k: int) -> str:
        if not state.retrieved_models:
            return json.dumps(
                {"error": "no_candidates", "message": "Search or fetch popular models first."}
            )

        _record_tool("finalize_recommendations", f"top_k={top_k}")

        cfg = RankerConfig(
            candidate_pool_size=ranker_config.candidate_pool_size,
            top_k=top_k,
            max_retries=ranker_config.max_retries,
        )
        try:
            await ranker.run(state, agents["ranker"], cfg)
        except Exception as e:
            logger.error(f"[Orchestrator] finalize_recommendations failed: {e}")
            state.agent_logs.append(f"Orchestrator tool finalize_recommendations error: {e}")
            return json.dumps({"error": str(e)})

        status["finalized"] = True
        state.agent_logs.append(
            f"Orchestrator tool finalize_recommendations: "
            f"{len(state.final_recommendations)} picks"
        )
        return json.dumps(
            {
                "recommendations": [
                    {"model_id": r.model_id, "score": r.score}
                    for r in state.final_recommendations
                ],
                "follow_up_questions": state.follow_up_questions,
            }
        )

    @tool(approval_mode="never_require")
    async def finalize_recommendations() -> str:
        """Hard-filter, re-rank, and explain the current candidate pool."""
        return await _finalize(default_top_k)

    return [search_models, get_popular_models, finalize_recommendations], status


def _build_prompt(
    state: RecommendationState,
    session_turns: Optional[list[Turn]] = None,
) -> str:
    parts: list[str] = []
    if session_turns:
        formatted = format_turns(session_turns)
        if formatted.strip():
            parts.append(
                "Recent conversation (oldest first):\n"
                f"\"\"\"{formatted}\"\"\"\n"
            )

    user_context = state.natural_language_context_for_requirements()
    parts.append(f"Current user request:\n\"\"\"{user_context}\"\"\"")

    top_k = max(1, min(5, int(state.recommendation_top_k or 3)))
    if top_k == 1:
        count_hint = (
            "The user wants a single best recommendation — "
            "call finalize_recommendations when ready."
        )
    else:
        count_hint = (
            "Return three options — call finalize_recommendations when ready."
        )

    parts.append(
        f"\n{count_hint}\n"
        "Decide whether to clarify (plain text only), search, retry search, "
        "fetch popular models, or finalize recommendations. "
        "Call finalize_recommendations when you have a good candidate pool."
    )
    return "\n".join(parts)


def _set_max_iterations(client: Any, max_steps: int) -> Optional[Any]:
    """Set client max_iterations under lock; return previous value for restore."""
    fic = getattr(client, "function_invocation_configuration", None)
    if fic is None:
        return None
    with _client_config_lock:
        if hasattr(fic, "get") and callable(fic.get):
            prev = fic.get("max_iterations")
            fic["max_iterations"] = max_steps
            return prev
        prev = getattr(fic, "max_iterations", None)
        fic.max_iterations = max_steps
        return prev


def _restore_max_iterations(client: Any, prev: Any) -> None:
    fic = getattr(client, "function_invocation_configuration", None)
    if fic is None:
        return
    with _client_config_lock:
        if prev is None:
            if hasattr(fic, "pop"):
                fic.pop("max_iterations", None)
            return
        if hasattr(fic, "__setitem__"):
            fic["max_iterations"] = prev
        else:
            fic.max_iterations = prev


def _clarification_from_text(text: str, state: RecommendationState) -> RecommendationState:
    state.stopped_for_query_refinement = True
    state.refinement_assistant_text = text
    state.final_recommendations = []
    state.scored_models = []
    state.agent_logs.append("Orchestrator: completed with clarification/redirect text")
    return state


def _empty_pool_message(state: RecommendationState) -> str:
    if state.retrieved_models:
        return ""
    return (
        "Search did not return any models for this request. "
        "Try narrowing your task description or relaxing popularity thresholds."
    )


async def _force_finalize(
    state: RecommendationState,
    agents: dict,
    ranker_config: RankerConfig,
    status: dict[str, bool],
    top_k: int = 3,
) -> bool:
    """Run ranker when the loop ended without finalize but candidates exist."""
    if status["finalized"] or not state.retrieved_models:
        return False

    cfg = RankerConfig(
        candidate_pool_size=ranker_config.candidate_pool_size,
        top_k=top_k,
        max_retries=ranker_config.max_retries,
    )
    try:
        await ranker.run(state, agents["ranker"], cfg)
        status["finalized"] = True
        state.agent_logs.append(
            "Orchestrator: auto-finalized after loop (candidates present, no finalize tool call)"
        )
        return True
    except Exception as e:
        logger.error(f"[Orchestrator] auto-finalize failed: {e}")
        state.agent_logs.append(f"Orchestrator auto-finalize failed: {e}")
        return False


async def run_agentic(
    state: RecommendationState,
    agents: dict,
    config: AgentConfig,
    ranker_config: RankerConfig,
    retriever_config: RetrieverConfig,
    session_turns: Optional[list[Turn]] = None,
) -> RecommendationState:
    """Run the orchestrator agent with a bounded tool loop."""
    activity: dict[str, int] = {"tool_calls": 0}
    tools, status = build_tools(
        state, agents, ranker_config, retriever_config, activity=activity
    )
    client = get_chat_client()
    prev_iters = _set_max_iterations(client, config.orchestrator_max_steps)

    prompt = _build_prompt(state, session_turns)
    text = ""
    outcome = "unknown"

    log_activity(
        f"loop start | max_steps={config.orchestrator_max_steps} | "
        f"query={state.user_query[:100]!r}"
    )

    try:
        agent = Agent(
            client=client,
            name="Orchestrator",
            instructions=_ORCHESTRATOR_INSTRUCTIONS,
            tools=tools,
        )
        kwargs = run_kwargs(state)
        result = await agent.run(prompt, **kwargs)
        save_session(state, kwargs.get("session"))
        text = agent_run_text(result).strip()

        if status["finalized"]:
            outcome = "finalized"
            return state

        if text:
            outcome = "clarification"
            return _clarification_from_text(text, state)

        force_top_k = max(1, min(5, int(state.recommendation_top_k or 3)))
        if await _force_finalize(state, agents, ranker_config, status, top_k=force_top_k):
            activity["tool_calls"] += 1
            log_activity(
                f"tool #{activity['tool_calls']}: finalize_recommendations (auto)"
            )
            outcome = "auto_finalized"
            return state

        empty_msg = _empty_pool_message(state)
        if empty_msg:
            outcome = "no_candidates"
            state.stopped_for_query_refinement = True
            state.refinement_assistant_text = empty_msg
            state.follow_up_questions = []
            state.agent_logs.append("Orchestrator: no candidates after loop")
            return state

        outcome = "incomplete"
        state.stopped_for_query_refinement = True
        state.refinement_assistant_text = (
            "I couldn't complete a recommendation for this turn. "
            "Please add more detail about your task, hardware, or constraints."
        )
        state.agent_logs.append("Orchestrator: loop ended without finalize or clarification")
        return state

    finally:
        _restore_max_iterations(client, prev_iters)
        log_activity(
            f"loop end | tool_calls={activity['tool_calls']} | outcome={outcome}"
        )
