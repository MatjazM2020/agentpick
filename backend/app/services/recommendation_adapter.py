"""
Recommendation adapter service.

Converts between internal RecommendationState and OpenAI-compatible API formats.
This adapter allows the recommendation engine to be used as a standard OpenAI provider.
"""

import json
from typing import Any, Dict, Iterator, List
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


def format_recommendations_as_text(
    recommendations: List[ScoredModel],
    explanations: Dict[str, str],
    follow_up_questions: List[str],
    needs_score_refinement: bool,
) -> str:
    """
    Convert recommendation objects into readable assistant output (plain text).

    Always formats up to three ranked models. Appends clarifying follow-ups and
    a closing question to help the user pick one of the three.
    """
    if not recommendations:
        return (
            "No ranked models are available for this turn. "
            "Add more detail about your task, hardware, or constraints and try again."
        )

    lines: List[str] = []
    if needs_score_refinement:
        lines.append(
            "Note: The best catalog matches for your wording are only moderate strength. "
            "The top three below are still the current best fits; answering the follow-up "
            "questions will sharpen the next ranking."
        )
        lines.append("")

    lines.append("Top 3 models:")
    lines.append("")

    for idx, rec in enumerate(recommendations, start=1):
        lines.append(f"{idx}. {rec.model_id}")
        if rec.score >= 0.62:
            lines.append("   Overall ranked fit: strong for your stated constraints.")
        elif rec.score >= 0.45:
            lines.append("   Overall ranked fit: moderate — compare deployment notes below.")
        else:
            lines.append("   Overall ranked fit: weaker match — treat as a candidate to validate.")
        reason = explanations.get(rec.model_id)
        if reason:
            lines.append(f"   Summary: {reason.strip()}")
        facts = rec.inference_facts or {}
        if facts:
            lines.append("   Deployment / inference notes (from catalog metadata only):")
            for key in (
                "parameter_count",
                "quantized_ram",
                "quantization",
                "recommended_quantization",
                "cpu_performance",
                "runtimes",
                "license",
            ):
                line = facts.get(key)
                if line:
                    lines.append(f"   - {line}")
        lines.append("")

    if follow_up_questions:
        lines.append("Follow-up questions (to refine your next request):")
        for q in follow_up_questions:
            lines.append(f"- {q.strip()}")
        lines.append("")

    ids = [r.model_id for r in recommendations]
    if len(ids) == 3:
        pick = (
            f"To pick one model among these three ({ids[0]}, {ids[1]}, {ids[2]}), "
            "what matters most for you: inference speed/latency, license/compliance, "
            "or fitting a specific memory / hardware budget?"
        )
    elif len(ids) == 2:
        pick = (
            f"To pick one model between {ids[0]} and {ids[1]}, "
            "which is more important: raw quality on your task type, or smallest footprint on your hardware?"
        )
    else:
        pick = (
            f"If you refine your constraints, we can suggest alternatives to {ids[0]}. "
            "What single constraint (latency, license, or model size) is tightest for you?"
        )
    lines.append(pick)

    return "\n".join(lines).rstrip()


def assistant_content_from_state(state: RecommendationState) -> str:
    """OpenAI assistant `content` from pipeline state (clarification-only or top-3 path)."""
    if state.stopped_for_query_refinement and state.refinement_assistant_text:
        return state.refinement_assistant_text.strip()
    return format_recommendations_as_text(
        state.final_recommendations,
        state.explanations,
        state.follow_up_questions,
        state.needs_score_refinement,
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
