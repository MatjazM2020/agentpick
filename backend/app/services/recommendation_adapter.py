"""
Recommendation adapter service.

Converts between internal RecommendationState and OpenAI-compatible API formats.
This adapter allows the recommendation engine to be used as a standard OpenAI provider.
"""

import json
from typing import Any, Dict, Iterator, List, Optional
from src.core.state import RecommendationState, ScoredModel


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return " ".join(text_parts).strip()
    return str(content).strip() if content is not None else ""


def extract_user_conversation_text(messages: List[Dict[str, Any]]) -> str:
    """
    All user turns in order, separated by blank lines — used as retrieval / requirements context.
    """
    parts: List[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        t = _normalize_message_content(m.get("content"))
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _format_model_block(index: int, model_id: str, reasons: List[str]) -> List[str]:
    """Numbered model entry with 2-3 pick reasons."""
    lines = [f"{index}. {model_id}:"]
    for reason in reasons[:3]:
        lines.append(f"- {reason}")
    return lines


def format_recommendations_as_text(
    recommendations: List[ScoredModel],
    model_summaries: Dict[str, dict],
    explanations: Dict[str, str],
    response_intro: Optional[str] = None,
) -> str:
    """
    Short numbered list: intro line, then each model with 2-3 bullets.

    Follow-up questions are omitted from text (Open WebUI renders clickable follow-ups).
    """
    if not recommendations:
        return (
            "No ranked models are available for this turn. "
            "Add more detail about your task, hardware, or constraints and try again."
        )

    default_intro = (
        "For your task, my top recommendation is:"
        if len(recommendations) == 1
        else "For your task, the best options are:"
    )
    intro = (response_intro or default_intro).strip()
    if not intro.endswith(":"):
        intro = intro.rstrip(".") + ":"

    lines: List[str] = [intro, ""]
    for index, rec in enumerate(recommendations, start=1):
        summary = model_summaries.get(rec.model_id, {})
        reasons = summary.get("reasons") or summary.get("pros") or []
        if not reasons and rec.model_id in explanations:
            reasons = [
                line.lstrip("- ").strip()
                for line in explanations[rec.model_id].splitlines()
                if line.strip()
            ]
        lines.extend(_format_model_block(index, rec.model_id, reasons))
        if index < len(recommendations):
            lines.append("")

    return "\n".join(lines).strip()


def assistant_content_from_state(state: RecommendationState) -> str:
    """OpenAI assistant `content` from pipeline state (clarification-only or top-3 path)."""
    if state.stopped_for_query_refinement and state.refinement_assistant_text:
        return state.refinement_assistant_text.strip()
    return format_recommendations_as_text(
        state.final_recommendations,
        state.model_summaries,
        state.explanations,
        state.response_intro,
    )


def state_to_openai_response(
    state: RecommendationState,
    model_id: str,
    completion_id: str,
    created_timestamp: int,
) -> Dict[str, Any]:
    """
    Convert RecommendationState to OpenAI-compatible chat completion response.

    Args:
        state: RecommendationState from recommendation pipeline
        model_id: Model identifier (e.g., "agentpick-recommender")
        completion_id: Full OpenAI-style id (e.g. chatcmpl-…)
        created_timestamp: Unix timestamp of request

    Returns:
        OpenAI-compatible response dict
    """
    assistant_message = assistant_content_from_state(state)

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_timestamp,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_message
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Not tracked in current pipeline
            "completion_tokens": 0,  # Not tracked in current pipeline
            "total_tokens": 0
        }
    }
    
    return response


def iter_chat_completion_sse(
    assistant_message: str,
    completion_id: str,
    model_id: str,
    created_timestamp: int,
) -> Iterator[bytes]:
    """
    Yield OpenAI-compatible chat.completion.chunk SSE frames, then [DONE].

    Used when clients (e.g. Open WebUI) call /v1/chat/completions with stream=true.
    Emits a minimal multi-chunk sequence: role, full content, finish.
    """
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": model_id,
    }

    def _frame(obj: dict) -> bytes:
        line = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
        return line.encode("utf-8")

    yield _frame(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield _frame(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": assistant_message},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield _frame(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


def extract_user_query(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the user's query from OpenAI message list.
    
    Takes the content of the last user message.
    
    Args:
        messages: List of OpenAI-format messages
        
    Returns:
        User query string
        
    Raises:
        ValueError: If no user message found
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if not user_messages:
        raise ValueError("No user message found in request")
    
    last_user_msg = user_messages[-1]
    content = last_user_msg.get("content")
    
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Handle multi-modal content (images, etc.)
        # For now, concatenate text parts
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        
        if text_parts:
            return " ".join(text_parts).strip()
        else:
            return ""
    else:
        return str(content).strip()
