"""
Chat completions endpoint (/v1/chat/completions).

OpenAI-compatible entry point. Runs the AgentPick recommendation agent and
returns its answer, streaming when requested. Open WebUI background tasks
(title/tag/follow-up generation) are answered with a plain completion.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.agent import complete_reply, complete_task, stream_reply
from src.core.agent_activity_log import (
    RequestContext,
    log_request_end,
    log_request_start,
)
from app.schemas.openai_chat import ChatCompletionRequest, ChatCompletionResponse
from app.services import openai_io

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest):
    """Answer a chat request with the recommendation agent (OpenAI format)."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    openai_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    user_text = openai_io.last_user_text(openai_messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message content provided")

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    request_id = completion_id[-8:]  # short suffix for log readability

    ctx = RequestContext(request_id=request_id, query_snippet=user_text[:80])
    agent_messages = openai_io.to_agent_messages(openai_messages)
    is_task = openai_io.is_openwebui_task(user_text)
    log_request_start(request_id, user_text, request.stream)

    # Streaming agent answer: yield chunks as the agent generates them.
    if request.stream and not is_task:

        async def _streamed_sse():
            # Attach inside the stream body — StreamingResponse may run in a
            # different asyncio context than the route handler.
            ctx.attach()
            try:
                async for frame in openai_io.sse_from_stream(
                    stream_reply(agent_messages), completion_id, request.model, created
                ):
                    yield frame
            finally:
                log_request_end(ctx.elapsed_ms, ctx._tool_count)
                ctx.detach()

        return StreamingResponse(_streamed_sse(), media_type="text/event-stream")

    # Full answer: agent reply, or plain completion for Open WebUI background tasks.
    ctx.attach()
    try:
        text = await (complete_task if is_task else complete_reply)(agent_messages)
        log_request_end(ctx.elapsed_ms, ctx._tool_count, status="task" if is_task else "ok")
    except Exception as e:
        log_request_end(ctx.elapsed_ms, ctx._tool_count, status="error")
        logger.error("chat/completions failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    finally:
        ctx.detach()

    if request.stream:
        return StreamingResponse(
            openai_io.sse_from_text(text, completion_id, request.model, created),
            media_type="text/event-stream",
        )
    return ChatCompletionResponse(
        **openai_io.completion_response(text, completion_id, request.model, created)
    )
