"""
Agent tools — the actions the recommendation agent can autonomously call.

Each tool is a thin async wrapper over ``src/catalog.py`` that returns JSON the
LLM can reason over. The framework's function-calling loop decides which tools
to invoke, in what order, and how to combine their results (see
docs/agent_patterns.py, Pattern 2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Optional

from agent_framework import tool
from pydantic import Field

from src import catalog
from src.core.agent_activity_log import current_context, log_tool_call

logger = logging.getLogger(__name__)


def _json(payload) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _label(name: str) -> str:
    """Get a per-request tool label like 'tool#2 search_models', or just the name."""
    ctx = current_context()
    return ctx.next_tool(name) if ctx else name


@tool(approval_mode="never_require")
async def search_models(
    query: Annotated[
        str,
        Field(description="Natural-language description of the task or capability, "
                          "e.g. 'summarize legal documents' or 'multilingual chat assistant'."),
    ],
    limit: Annotated[int, Field(description="Max models to return (1-15).", ge=1, le=15)] = 8,
) -> str:
    """Semantic search over Hugging Face model cards. Best for fuzzy or task-based intent."""
    label = _label("search_models")
    t0 = time.monotonic()
    try:
        models = await asyncio.to_thread(catalog.semantic_search, query, limit)
        elapsed = (time.monotonic() - t0) * 1000
        log_tool_call(label, {"query": query, "limit": limit}, len(models), elapsed)
        return _json({"models": models, "count": len(models)})
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        log_tool_call(label, {"query": query, "limit": limit}, None, elapsed, error=str(e))
        logger.error("search_models failed: %s", e)
        return _json({"error": str(e), "models": [], "count": 0})


@tool(approval_mode="never_require")
async def filter_models(
    task_type: Annotated[
        Optional[str],
        Field(description="Exact Hugging Face pipeline_tag, e.g. 'text-generation', "
                          "'summarization', 'translation'."),
    ] = None,
    tag: Annotated[
        Optional[str],
        Field(description="A single tag that must be present (exact match), e.g. 'code' or "
                          "'conversational'. Tags are sparse and inconsistent: capabilities like "
                          "instruction tuning are usually NOT tagged — filter by name_contains "
                          "('instruct', 'chat', '-it') instead."),
    ] = None,
    name_contains: Annotated[
        Optional[str],
        Field(description="Substring the model id must contain, e.g. 'instruct', 'coder', 'distill'."),
    ] = None,
    min_params_b: Annotated[
        Optional[float], Field(description="Minimum parameter count in billions, e.g. 7. "
                                           "Excludes models whose size is unknown (~1/3 of catalog).")
    ] = None,
    max_params_b: Annotated[
        Optional[float], Field(description="Maximum parameter count in billions, e.g. 15. "
                                           "Excludes models whose size is unknown (~1/3 of catalog).")
    ] = None,
    sort_by: Annotated[
        str,
        Field(description="Ordering: 'downloads', 'likes', 'smallest', 'largest', or 'newest'."),
    ] = "downloads",
    limit: Annotated[int, Field(description="Max models to return (1-25).", ge=1, le=25)] = 8,
) -> str:
    """Structured catalog query for precise constraints, popularity, and size rankings.

    Use for concrete requirements ("under 4B parameters", "coding model", "most
    downloaded", "smallest instruction-tuned model"). The result includes
    'total_matches' (how many catalog models satisfy the filters overall) and
    'warnings' — heed both: a tiny total usually means the filter is too
    narrow, not that the catalog lacks such models. An empty result with no
    warnings means no catalog model satisfies the constraints.
    """
    label = _label("filter_models")
    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(
            catalog.filter_models,
            task_type, tag, name_contains, min_params_b, max_params_b, sort_by, limit,
        )
        models = result["models"]
        elapsed = (time.monotonic() - t0) * 1000
        log_tool_call(
            label,
            {
                "task_type": task_type, "tag": tag, "name": name_contains,
                "min_b": min_params_b, "max_b": max_params_b, "sort_by": sort_by,
            },
            len(models),
            elapsed,
        )
        return _json({
            "models": models,
            "count": len(models),
            "total_matches": result["total_matches"],
            "warnings": result["warnings"],
        })
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        log_tool_call(label, {"task_type": task_type}, None, elapsed, error=str(e))
        logger.error("filter_models failed: %s", e)
        return _json({"error": str(e), "models": [], "count": 0})


@tool(approval_mode="never_require")
async def get_model_details(
    model_id: Annotated[
        str, Field(description="Exact model id, e.g. 'Qwen/Qwen2.5-Coder-14B-Instruct'.")
    ],
) -> str:
    """Full metadata and model card (README) for one model. Use to compare or verify a candidate."""
    label = _label("get_model_details")
    t0 = time.monotonic()
    try:
        detail = await asyncio.to_thread(catalog.get_model_details, model_id)
        elapsed = (time.monotonic() - t0) * 1000
        if detail is None:
            log_tool_call(label, {"model_id": model_id}, 0, elapsed)
            return _json({"error": f"'{model_id}' not found in catalog"})
        log_tool_call(label, {"model_id": model_id}, 1, elapsed)
        return _json(detail)
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        log_tool_call(label, {"model_id": model_id}, None, elapsed, error=str(e))
        logger.error("get_model_details failed: %s", e)
        return _json({"error": str(e)})


TOOLS = [search_models, filter_models, get_model_details]
