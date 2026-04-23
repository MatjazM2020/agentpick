"""
Recommendation pipeline service layer.

Entry point for the recommendation system. Handles:
- New recommendation requests
- Refinement requests (continuing from previous state)
- State management across multiple turns
- Logging and observability

Usage:
    # New request
    state = await run_recommendation("I need a small model for CPU classification")
    
    # Refinement request
    refined_state = await run_recommendation("make it faster", state=state)
"""

import logging
import time
from typing import Optional
from datetime import datetime

from backend.src.core.state import RecommendationState, Message
from backend.src.core.config import AgentConfig, ScoringConfig, RetrieverConfig
from backend.src.core.agent_factory import create_agents
from backend.src.agents import supervisor


logger = logging.getLogger(__name__)


async def run_recommendation(
    query: str,
    state: Optional[RecommendationState] = None,
    config: Optional[AgentConfig] = None,
    scoring_config: Optional[ScoringConfig] = None,
    retriever_config: Optional[RetrieverConfig] = None,
) -> RecommendationState:
    """
    Run the complete recommendation pipeline.
    
    Handles both new requests and refinement requests.
    
    Args:
        query: User query (new or refinement)
        state: Existing RecommendationState (for refinement). If None, creates new state.
        config: AgentConfig (uses defaults if None)
        scoring_config: ScoringConfig (uses defaults if None)
        retriever_config: RetrieverConfig (uses defaults if None)
        
    Returns:
        Updated RecommendationState with recommendations and metadata
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If agents cannot be initialized
    """
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    start_time = time.time()
    
    # === Determine Request Type ===
    if state is None:
        # New request: create fresh state
        state = RecommendationState(user_query=query)
        request_type = "new"
        logger.info(f"[Pipeline] New request: {query[:100]}...")
    else:
        # Refinement: reuse existing state, append to conversation
        request_type = "refinement"
        state.messages.append(Message(role="user", content=query))
        state.user_query = query
        state.iteration += 1
        logger.info(f"[Pipeline] Refinement request (iteration={state.iteration}): {query[:100]}...")
    
    # === Initialize Agents ===
    try:
        agents = create_agents()
        logger.info(f"[Pipeline] Agents initialized successfully")
    except Exception as e:
        logger.error(f"[Pipeline] Failed to initialize agents: {e}")
        raise RuntimeError(f"Agent initialization failed: {e}")
    
    # === Run Pipeline ===
    try:
        state = await supervisor.run_pipeline(
            state=state,
            agents=agents,
            config=config or AgentConfig(),
            scorer_config=scoring_config or ScoringConfig(),
            retriever_config=retriever_config or RetrieverConfig(),
        )
        logger.info(
            f"[Pipeline] {request_type} request completed. "
            f"Recommendations: {len(state.final_recommendations)}"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Pipeline execution failed: {e}")
        state.agent_logs.append(f"Pipeline error: {e}")
        raise
    
    # === Add Execution Metadata ===
    execution_time = time.time() - start_time
    state.agent_logs.append(
        f"[Pipeline] Execution completed in {execution_time:.2f}s"
    )
    
    # Add assistant message with summary
    if state.final_recommendations:
        summary = f"Found {len(state.final_recommendations)} recommendations. " \
                  f"Top model: {state.final_recommendations[0].model_id} " \
                  f"(score: {state.final_recommendations[0].score:.3f})"
    else:
        summary = "No recommendations found. Please refine your query."
    
    state.messages.append(Message(
        role="assistant",
        content=summary
    ))
    
    logger.info(f"[Pipeline] Total execution time: {execution_time:.2f}s")
    
    return state
