"""
Recommendation adapter service.

Converts between internal RecommendationState and OpenAI-compatible API formats.
This adapter allows the recommendation engine to be used as a standard OpenAI provider.
"""

import json
from typing import Any, Dict, Iterator, List
from src.core.state import RecommendationState, ScoredModel


def format_recommendations_as_text(
    recommendations: List[ScoredModel],
    explanations: Dict[str, str],
) -> str:
    """
    Convert recommendation objects into readable assistant output (plain text).

    Args:
        recommendations: List of ScoredModel objects from pipeline
        explanations: model_id -> synthesizer explanation

    Returns:
        Formatted text response suitable for LLM chat
    """
    if not recommendations:
        return "No recommendations found for your query."

    lines = ["Recommended models:", ""]

    for idx, rec in enumerate(recommendations, start=1):
        lines.append(f"{idx}. {rec.model_id}")
        lines.append(f"   Score: {rec.score:.2f}")
        reason = explanations.get(rec.model_id)
        if reason:
            lines.append(f"   Reason: {reason.strip()}")
        lines.append("")

    return "\n".join(lines).rstrip()


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
    assistant_message = format_recommendations_as_text(
        state.final_recommendations,
        state.explanations,
    )

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
