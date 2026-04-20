"""
Retriever agent.

Queries Qdrant for candidate models using semantic search and metadata filtering.
Deterministic behavior - no LLM involved.

Responsibilities:
- Embed the user query
- Build metadata filters from task_type and constraints
- Query Qdrant collection
- Deduplicate results by model_id
- Sort by relevance score
- Return structured candidate models
- Log all operations for observability
"""

import logging
from typing import Optional
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.core.state import RecommendationState
from src.core.config import RetrieverConfig
from src.core.llm import embed


logger = logging.getLogger(__name__)


def run(
    state: RecommendationState,
    config: Optional[RetrieverConfig] = None,
    refine: bool = False
) -> RecommendationState:
    """
    Retrieve candidate models from Qdrant.
    
    Args:
        state: RecommendationState with user_query and extracted constraints
        config: RetrieverConfig (uses defaults if None)
        refine: If True, increase search limit and relax filters
        
    Returns:
        Updated state with retrieved_models populated
    """
    
    if config is None:
        config = RetrieverConfig()
    
    query = state.user_query
    logger.info(
        f"[Retriever] Starting retrieval. refine={refine}, "
        f"task_type={state.task_type}, constraints={state.constraints}"
    )
    
    # Step 1: Embed query
    try:
        embedding = embed(query)
        logger.info(f"[Retriever] Query embedded successfully (dimension={len(embedding)})")
    except Exception as e:
        logger.error(f"[Retriever] Embedding failed: {e}")
        state.agent_logs.append(f"Retriever: Embedding failed: {e}")
        return state
    
    # Step 2: Build metadata filter
    filter_obj = _build_filter(state, refine)
    log_filter = str(filter_obj) if filter_obj else "None"
    logger.info(f"[Retriever] Filter: {log_filter}")
    
    # Step 3: Search Qdrant
    top_k = config.top_k_chunks * 2 if refine else config.top_k_chunks
    
    try:
        client = QdrantClient(url=config.qdrant_url)
        search_results = client.search(
            collection_name=config.qdrant_collection_name,
            query_vector=embedding,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
        )
        logger.info(
            f"[Retriever] Qdrant search returned {len(search_results)} chunks "
            f"(top_k={top_k})"
        )
    except Exception as e:
        logger.error(f"[Retriever] Qdrant search failed: {e}")
        state.agent_logs.append(f"Retriever: Qdrant search failed: {e}")
        return state
    
    # Step 4: Deduplicate by model_id and aggregate scores
    # Structure: model_id -> {score_sum, count, metadata, point_id}
    model_aggregates = defaultdict(lambda: {
        "score_sum": 0.0,
        "count": 0,
        "metadata": {},
        "point_ids": []
    })
    
    for point in search_results:
        payload = point.payload
        model_id = payload.get("model_id")
        score = point.score
        
        if not model_id:
            logger.warning(f"[Retriever] Point {point.id} missing model_id, skipping")
            continue
        
        model_aggregates[model_id]["score_sum"] += score
        model_aggregates[model_id]["count"] += 1
        model_aggregates[model_id]["point_ids"].append(point.id)
        
        # Keep first (highest-scoring) metadata for this model
        if not model_aggregates[model_id]["metadata"]:
            model_aggregates[model_id]["metadata"] = payload
    
    logger.info(f"[Retriever] After deduplication: {len(model_aggregates)} unique models")
    
    # Step 5: Average scores and build output
    candidates = []
    for model_id, agg in model_aggregates.items():
        avg_score = agg["score_sum"] / agg["count"] if agg["count"] > 0 else 0.0
        
        # Filter by minimum similarity threshold
        if avg_score < config.min_similarity_threshold:
            logger.debug(
                f"[Retriever] Skipping {model_id}: score {avg_score} < "
                f"threshold {config.min_similarity_threshold}"
            )
            continue
        
        candidate = {
            "id": model_id,
            "score": avg_score,
            "point_ids": agg["point_ids"],
            "num_chunks": agg["count"],
            "metadata": agg["metadata"]
        }
        candidates.append(candidate)
    
    # Step 6: Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Step 7: Limit to top_k_models
    top_candidates = candidates[:config.top_k_models]
    logger.info(
        f"[Retriever] Final results: {len(top_candidates)} models. "
        f"Top 3: {[c['id'] for c in top_candidates[:3]]}"
    )
    
    # Log top scores
    for i, candidate in enumerate(top_candidates[:5], 1):
        logger.debug(
            f"[Retriever] #{i}: {candidate['id']}, "
            f"score={candidate['score']:.4f}, "
            f"chunks={candidate['num_chunks']}"
        )
    
    # Step 8: Update state
    state.retrieved_models = top_candidates
    state.retrieval_complete = True
    
    state.agent_logs.append(
        f"Retriever: Retrieved {len(top_candidates)} unique models "
        f"from {len(search_results)} chunks. "
        f"Top model: {top_candidates[0]['id'] if top_candidates else 'N/A'}"
    )
    
    return state


def _build_filter(
    state: RecommendationState,
    refine: bool = False
) -> Optional[Filter]:
    """
    Build Qdrant filter from task_type and constraints.
    
    Args:
        state: RecommendationState with task_type, constraints, preferences
        refine: If True, use relaxed filters
        
    Returns:
        Filter object for Qdrant or None if no filters needed
    """
    
    conditions = []
    
    # Filter by pipeline_tag (task_type)
    if state.task_type and state.task_type != "general" and not refine:
        logger.debug(f"[Retriever._build_filter] Adding task_type filter: {state.task_type}")
        conditions.append(
            FieldCondition(
                key="pipeline_tag",
                match=MatchValue(value=state.task_type)
            )
        )
    
    # Filter by license
    license_constraint = state.constraints.get("license")
    if license_constraint and not refine:
        logger.debug(f"[Retriever._build_filter] Adding license filter: {license_constraint}")
        conditions.append(
            FieldCondition(
                key="license",
                match=MatchValue(value=license_constraint)
            )
        )
    
    # Filter by tags if specified in preferences
    tags_constraint = state.constraints.get("tags")
    if tags_constraint and not refine:
        # tags_constraint can be a list or a single string
        if isinstance(tags_constraint, str):
            tags_constraint = [tags_constraint]
        
        logger.debug(f"[Retriever._build_filter] Adding tags filter: {tags_constraint}")
        for tag in tags_constraint:
            conditions.append(
                FieldCondition(
                    key="tags",
                    match=MatchValue(value=tag)
                )
            )
    
    if not conditions:
        logger.debug("[Retriever._build_filter] No filters applied")
        return None
    
    # Combine all conditions with AND logic
    # If only one condition, return it directly
    if len(conditions) == 1:
        return conditions[0]
    
    # Multiple conditions - combine with AND
    # Using nested Filter structure
    from qdrant_client.models import Filter as QdrantFilter
    return QdrantFilter(must=conditions)
