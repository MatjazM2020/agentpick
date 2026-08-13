"""
Agent tools — the actions the recommendation agent can autonomously call.

Each tool is a thin async wrapper over ``src/catalog.py`` that returns JSON the
LLM can reason over. The framework's function-calling loop decides which tools
to invoke, in what order, and how to combine their results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Callable, Optional

from agent_framework import tool
from pydantic import Field

from src import catalog
from src.core.agent_activity_log import current_context, log_tool_call

logger = logging.getLogger(__name__)


def _json(payload) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


async def _run_tool(name: str, log_args: dict, body: Callable[[], tuple[dict, int]]) -> str:
    """Run one tool body in a worker thread, log the call + timing, return JSON.

    ``body`` is a no-arg blocking callable (catalog I/O) returning
    ``(payload, result_count)``. Exceptions become an ``{"error": ...}``
    payload so the LLM sees the failure and can adjust its approach.
    """
    ctx = current_context()
    label = ctx.next_tool(name) if ctx else name
    t0 = time.monotonic()
    try:
        payload, count = await asyncio.to_thread(body)
    except Exception as e:
        log_tool_call(label, log_args, None, (time.monotonic() - t0) * 1000, error=str(e))
        logger.error("%s failed: %s", name, e)
        return _json({"error": str(e)})
    log_tool_call(label, log_args, count, (time.monotonic() - t0) * 1000)
    return _json(payload)


@tool(approval_mode="never_require")
async def search_models(
    query: Annotated[
        str,
        Field(description="Natural-language description of the task or capability, "
                          "e.g. 'summarize legal documents' or 'multilingual chat assistant'."),
    ],
    limit: Annotated[int, Field(description="Max models to return (1-15).", ge=1, le=15)] = 8,
) -> str:
    """Semantic search over Hugging Face model cards. Best for fuzzy or task-based intent.

    Re-uploads of the same checkpoint are de-duplicated.
    """
    def body():
        models = catalog.semantic_search(query, limit)
        return {"models": models, "count": len(models)}, len(models)

    return await _run_tool("search_models", {"query": query, "limit": limit}, body)


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
        Field(description="Substring the model id must contain, e.g. a capability "
                          "marker such as 'instruct'."),
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
        Field(description="Ordering: 'downloads', 'likes', 'smallest', 'largest', or "
                          "'newest'. Use 'largest'/'smallest' for size superlatives; "
                          "'downloads' only when the user asks for popularity."),
    ] = "downloads",
    limit: Annotated[int, Field(description="Max models to return (1-25).", ge=1, le=25)] = 8,
) -> str:
    """Structured catalog query for precise constraints, popularity, and size rankings.

    Use for concrete requirements (an explicit size bound, a task, "most
    downloaded") and for size superlatives — the biggest or smallest model of a
    given kind — via sort_by=largest/smallest combined with name_contains
    filters, not via sort_by=downloads. The result includes
    'total_matches' (how many catalog models satisfy the filters overall) and
    'warnings' — heed both: a tiny total usually means the filter is too
    narrow, not that the catalog lacks such models. An empty result with no
    warnings means no catalog model satisfies the constraints.
    """
    def body():
        result = catalog.filter_models(
            task_type, tag, name_contains, min_params_b, max_params_b, sort_by, limit
        )
        payload = {
            "models": result["models"],
            "count": len(result["models"]),
            "total_matches": result["total_matches"],
            "warnings": result["warnings"],
        }
        return payload, len(result["models"])

    return await _run_tool(
        "filter_models",
        {
            "task_type": task_type, "tag": tag, "name": name_contains,
            "min_b": min_params_b, "max_b": max_params_b, "sort_by": sort_by,
        },
        body,
    )


@tool(approval_mode="never_require")
async def get_model_details(
    model_id: Annotated[
        str, Field(description="Exact model id as it appears in a search or filter "
                               "result, e.g. 'org/model-name'.")
    ],
) -> str:
    """Full metadata and model card (README) for one model. Use to compare or verify a candidate."""
    def body():
        detail = catalog.get_model_details(model_id)
        if detail is None:
            return {"error": f"'{model_id}' not found in catalog"}, 0
        return detail, 1

    return await _run_tool("get_model_details", {"model_id": model_id}, body)


TOOLS = [search_models, filter_models, get_model_details]
