"""
Recommendation adapter service.

Converts between internal RecommendationState and OpenAI-compatible API formats.
This adapter allows the recommendation engine to be used as a standard OpenAI provider.
"""

import logging
from typing import List, Dict, Any
from src.core.state import RecommendationState, ScoredModel

logger = logging.getLogger(__name__)


def format_recommendations_as_text(recommendations: List[ScoredModel]) -> str:
    """
    Convert recommendation objects into readable assistant output.
    
    Args:
        recommendations: List of ScoredModel objects from pipeline
        
    Returns:
        Formatted text response suitable for LLM chat
    """
    if not recommendations:
        return "No recommendations found for your query."
    
    lines = ["Recommended models:\n"]
    
    for idx, rec in enumerate(recommendations, start=1):
        # Format score as percentage
        score_percent = rec.score * 100
        
        # Main recommendation line
        lines.append(f"{idx}. **{rec.model_id}**")
        lines.append(f"   Score: {score_percent:.1f}%")
        
        # Add score breakdown if available
        if rec.score_breakdown:
            breakdown_items = []
            for component, value in rec.score_breakdown.items():
                component_display = component.replace("_", " ").title()
                breakdown_items.append(f"{component_display}: {value:.2f}")
            
            if breakdown_items:
                lines.append(f"   Breakdown: {', '.join(breakdown_items)}")
        
        # Add metadata if available (description, parameters, etc.)
        if rec.metadata:
            # Try to extract useful metadata fields
            if "description" in rec.metadata:
                lines.append(f"   Description: {rec.metadata['description']}")
            
            if "parameters" in rec.metadata:
                params = rec.metadata["parameters"]
                if isinstance(params, dict):
                    # Format key parameters
                    if "size" in params:
                        lines.append(f"   Size: {params['size']}")
                    if "quantized" in params:
                        lines.append(f"   Quantized: {params['quantized']}")
        
        lines.append("")  # Add spacing between recommendations
    
    return "\n".join(lines)


def state_to_openai_response(
    state: RecommendationState,
    model_id: str,
    request_id: str,
    created_timestamp: int,
) -> Dict[str, Any]:
    """
    Convert RecommendationState to OpenAI-compatible chat completion response.
    
    Args:
        state: RecommendationState from recommendation pipeline
        model_id: Model identifier (e.g., "agentpick-recommender")
        request_id: Unique request ID
        created_timestamp: Unix timestamp of request
        
    Returns:
        OpenAI-compatible response dict
    """
    # Format recommendations as assistant message
    assistant_message = format_recommendations_as_text(state.final_recommendations)
    
    response = {
        "id": f"chatcmpl-{request_id}",
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
