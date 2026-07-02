"""
Chat completions endpoint (/v1/chat/completions).

Main endpoint for recommendation requests. Accepts OpenAI-format requests
and returns recommendations as assistant messages.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src.services.recommendation_pipeline import run_recommendation
from src.conversation import conversation_store, session_id_from_messages
from src.conversation.openwebui_tasks import (
    fallback_follow_ups_from_messages,
    follow_ups_response_content,
    is_follow_up_generation_task,
)
from app.schemas.openai_chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.recommendation_adapter import (
    extract_user_conversation_text,
    extract_user_query,
    iter_chat_completion_sse,
    state_to_openai_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest):
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
        openai_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        try:
            user_query = extract_user_query(openai_messages)
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
        conversation_text = extract_user_conversation_text(openai_messages)
        session_id = session_id_from_messages(openai_messages)

        if is_follow_up_generation_task(user_query):
            follow_ups = conversation_store.get_follow_ups(session_id)
            if not follow_ups:
                follow_ups = fallback_follow_ups_from_messages(openai_messages)
            created_timestamp = int(time.time())
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            assistant_text = follow_ups_response_content(follow_ups)
            logger.info(
                f"[ChatCompletions] Follow-up task: returning {len(follow_ups)} suggestions"
            )
            if request.stream:
                return StreamingResponse(
                    iter_chat_completion_sse(
                        assistant_text,
                        completion_id,
                        request.model,
                        created_timestamp,
                    ),
                    media_type="text/event-stream",
                )
            return ChatCompletionResponse(
                id=completion_id,
                created=created_timestamp,
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_text},
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        try:
            state = await run_recommendation(
                user_query,
                conversation_text=conversation_text or None,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Recommendation processing failed: {str(e)}"
            )
        
        created_timestamp = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4()}"

        # Convert to OpenAI format
        response_dict = state_to_openai_response(
            state=state,
            model_id=request.model,
            completion_id=completion_id,
            created_timestamp=created_timestamp,
        )
        
        logger.info(
            f"[ChatCompletions] Response: "
            f"recommendations={len(state.final_recommendations)}"
        )

        if request.stream:
            assistant_text = response_dict["choices"][0]["message"]["content"]
            return StreamingResponse(
                iter_chat_completion_sse(
                    assistant_text,
                    completion_id,
                    request.model,
                    created_timestamp,
                ),
                media_type="text/event-stream",
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
