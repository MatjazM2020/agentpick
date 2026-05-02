"""
Chat completions endpoint (/v1/chat/completions).

Main endpoint for recommendation requests. Accepts OpenAI-format requests
and returns recommendations as assistant messages.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from src.services.recommendation_pipeline import run_recommendation
from app.schemas.openai_chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.recommendation_adapter import (
    extract_user_query,
    state_to_openai_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Create a chat completion using the recommendation engine.
    
    This endpoint accepts OpenAI-format chat requests and returns
    recommendations formatted as LLM responses.
    
    Args:
        request: OpenAI ChatCompletionRequest
        
    Returns:
        OpenAI ChatCompletionResponse
        
    Raises:
        HTTPException: On validation or processing errors
    """
    
    try:
        # Validate request
        if not request.messages:
            raise ValueError("No messages provided")
        
        # Extract user query from OpenAI message format
        try:
            user_query = extract_user_query(
                [{"role": m.role, "content": m.content} for m in request.messages]
            )
        except ValueError as e:
            logger.warning(f"Query extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid message format: {str(e)}")
        
        if not user_query or not user_query.strip():
            raise HTTPException(
                status_code=400,
                detail="User message content is empty"
            )
        
        logger.info(
            f"[ChatCompletions] Request: model={request.model}, "
            f"query={user_query[:80]}..."
        )
        
        # Run recommendation pipeline
        # The pipeline is async and handles event loop properly
        try:
            state = await run_recommendation(user_query)
        except Exception as e:
            logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Recommendation processing failed: {str(e)}"
            )
        
        # Generate response ID and timestamp
        response_id = str(uuid.uuid4()).replace("-", "")[:12]
        created_timestamp = int(time.time())
        
        # Convert to OpenAI format
        response_dict = state_to_openai_response(
            state=state,
            model_id=request.model,
            request_id=response_id,
            created_timestamp=created_timestamp,
        )
        
        logger.info(
            f"[ChatCompletions] Response: "
            f"recommendations={len(state.final_recommendations)}"
        )
        
        # Return as Pydantic model for validation
        return ChatCompletionResponse(**response_dict)
    
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in chat/completions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
