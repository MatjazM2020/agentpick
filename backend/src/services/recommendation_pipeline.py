"""
Recommendation pipeline service layer.

Entry point for the recommendation system. Handles new and refinement requests,
in-memory session turns for multi-turn context, and orchestrator execution.
"""

import logging
import time
from typing import Optional

from src.conversation.intent import infer_recommendation_top_k
from src.core.state import RecommendationState, Message
from src.core.config import AgentConfig, RankerConfig, RetrieverConfig
from src.core.agent_factory import create_agents
from src.core.agent_session import ensure_session_id
from src.conversation import conversation_store
from src.conversation.openwebui_tasks import (
    fallback_follow_ups_from_messages,
    follow_ups_response_content,
    is_follow_up_generation_task,
)
from src.agents import orchestrator


logger = logging.getLogger(__name__)


async def run_recommendation(
    query: str,
    state: Optional[RecommendationState] = None,
    conversation_text: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    ranker_config: Optional[RankerConfig] = None,
    retriever_config: Optional[RetrieverConfig] = None,
) -> RecommendationState:
    """
    Run the complete agentic recommendation pipeline.

    Multi-turn context is passed to the orchestrator via ``session_turns`` (recent
    in-memory history) and ``conversation_text`` (all user turns from the client).
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    start_time = time.time()

    try:
        agents = create_agents()
        logger.info("[Pipeline] Agents initialized successfully")
    except Exception as e:
        logger.error(f"[Pipeline] Failed to initialize agents: {e}")
        raise RuntimeError(f"Agent initialization failed: {e}") from e

    session_turns = conversation_store.recent(session_id) if session_id else []

    if state is None:
        ct = (conversation_text or "").strip() or query.strip()
        state = RecommendationState(user_query=query.strip(), conversation_text=ct)
        request_type = "new"
        logger.info(f"[Pipeline] New request: {query[:100]}...")
    else:
        request_type = "refinement"
        state.messages.append(Message(role="user", content=query))
        state.user_query = query.strip()
        if conversation_text is not None and conversation_text.strip():
            state.conversation_text = conversation_text.strip()
        state.stopped_for_query_refinement = False
        state.refinement_assistant_text = None
        state.iteration += 1
        logger.info(
            f"[Pipeline] Refinement request (iteration={state.iteration}): {query[:100]}..."
        )

    ensure_session_id(state)
    if state.iteration < 1:
        state.iteration = 1

    if session_id:
        conversation_store.add(session_id, "user", query)

    agent_config = config or AgentConfig()
    is_follow_up = bool(session_turns) or request_type == "refinement"
    top_k = infer_recommendation_top_k(
        query,
        conversation_text=state.conversation_text,
        is_follow_up=is_follow_up,
    )
    state.recommendation_top_k = top_k

    ranker_cfg = ranker_config or RankerConfig(top_k=top_k)
    retriever_cfg = retriever_config or RetrieverConfig()
    logger.info(f"[Pipeline] recommendation_top_k={top_k}")

    try:
        logger.info(f"[Pipeline] Starting orchestrator: {state.user_query[:100]}...")
        state = await orchestrator.run_agentic(
            state,
            agents,
            agent_config,
            ranker_cfg,
            retriever_cfg,
            session_turns=session_turns,
        )
        logger.info(
            f"[Pipeline] {request_type} request completed. "
            f"Recommendations: {len(state.final_recommendations)}"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Pipeline execution failed: {e}")
        state.agent_logs.append(f"Pipeline error: {e}")
        raise

    execution_time = time.time() - start_time
    state.agent_logs.append(f"[Pipeline] Execution completed in {execution_time:.2f}s")

    if state.stopped_for_query_refinement and state.refinement_assistant_text:
        summary = state.refinement_assistant_text
    elif state.final_recommendations:
        summary = (
            f"Found {len(state.final_recommendations)} recommendations. "
            f"Top model: {state.final_recommendations[0].model_id} "
            f"(score: {state.final_recommendations[0].score:.3f})"
        )
    else:
        summary = "No recommendations found. Please refine your query."

    state.messages.append(Message(role="assistant", content=summary))

    if session_id:
        conversation_store.add(session_id, "assistant", summary)
        if state.follow_up_questions:
            conversation_store.set_follow_ups(session_id, state.follow_up_questions)

    logger.info(f"[Pipeline] Total execution time: {execution_time:.2f}s")
    return state
