"""
Retriever agent.

Queries Qdrant for candidate models using semantic search and metadata filtering.
Deterministic behavior - no LLM involved.

Responsibilities:
- Embed the user query using SentenceTransformer
- Build metadata filters from task_type and constraints
- Query Qdrant "hf_models" collection via ``query_points`` (vector query API)
- Deduplicate results by model_id (averaging scores across chunks)
- Sort by relevance score
- Return structured candidate models with payloads
- Log all operations for observability

This module provides both:
1. QdrantRetriever: Standalone class for reusable retrieval logic
2. run(): Integration function for the recommendation pipeline
"""

import logging
from typing import Optional, List, Dict, Any
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.core.state import RecommendationState
from src.core.config import RetrieverConfig
from src.core.llm import embed


logger = logging.getLogger(__name__)


class QdrantRetriever:
    """
    Standalone retriever for semantic search against Qdrant.
    
    Encapsulates all retrieval logic: embedding, filtering, search, deduplication.
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize retriever with configuration.
        
        Args:
            config: RetrieverConfig instance (uses defaults if None)
        """
        self.config = config or RetrieverConfig()
        self.client = QdrantClient(url=self.config.qdrant_url)
        logger.info(
            f"[QdrantRetriever] Initialized with collection: "
            f"{self.config.qdrant_collection_name}"
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        relax_filters: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant models using semantic similarity.
        
        Args:
            query: User query string
            top_k: Number of models to return (uses config.top_k_models if None)
            metadata_filter: Optional filter dict with keys like:
                - task_type: filter by pipeline_tag
                - license: filter by license
                - tags: filter by tags (list or single string)
            relax_filters: If True, return more candidates and relax filtering
            
        Returns:
            List of dicts with keys:
            - id: model_id
            - score: float (0.0-1.0, averaged across chunks)
            - metadata: full payload from Qdrant
            - num_chunks: number of chunks aggregated
            - point_ids: Qdrant point IDs
        """
        logger.info(f"[QdrantRetriever.search] Query: '{query[:100]}...'")
        
        # Step 1: Embed query
        try:
            embedding = embed(query)
            logger.info(f"[QdrantRetriever.search] Embedded query (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"[QdrantRetriever.search] Embedding failed: {e}")
            raise
        
        # Step 2: Build filter from metadata
        filter_obj = self._build_filter(metadata_filter, relax_filters)
        log_filter = str(filter_obj) if filter_obj else "None"
        logger.info(f"[QdrantRetriever.search] Filter: {log_filter}")
        
        # Step 3: Query Qdrant
        search_limit = (self.config.top_k_chunks * 2) if relax_filters else self.config.top_k_chunks
        
        try:
            qp_kwargs: Dict[str, Any] = {
                "collection_name": self.config.qdrant_collection_name,
                "query": embedding,
                "query_filter": filter_obj,
                "limit": search_limit,
                "with_payload": True,
            }
            if self.config.qdrant_query_using:
                qp_kwargs["using"] = self.config.qdrant_query_using
            query_response = self.client.query_points(**qp_kwargs)
            search_results: List[Any] = list(query_response.points)
            logger.info(
                f"[QdrantRetriever.search] Found {len(search_results)} chunks "
                f"(limit={search_limit})"
            )
        except Exception as e:
            logger.error(f"[QdrantRetriever.search] Qdrant query failed: {e}")
            raise
        
        # Step 4: Deduplicate by model_id, averaging scores
        model_aggregates: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "score_sum": 0.0,
                "count": 0,
                "metadata": {},
                "point_ids": []
            }
        )
        
        for point in search_results:
            payload = point.payload
            model_id = payload.get("model_id")
            score = point.score
            
            if not model_id:
                logger.warning(f"[QdrantRetriever.search] Point {point.id} missing model_id, skipping")
                continue
            
            model_aggregates[model_id]["score_sum"] += score
            model_aggregates[model_id]["count"] += 1
            model_aggregates[model_id]["point_ids"].append(point.id)
            
            # Store first (highest-scoring) payload for this model
            if not model_aggregates[model_id]["metadata"]:
                model_aggregates[model_id]["metadata"] = payload
        
        logger.info(f"[QdrantRetriever.search] Deduplicated to {len(model_aggregates)} unique models")
        
        # Step 5: Build output candidates with averaged scores
        min_sim = self.config.min_similarity_threshold
        if relax_filters:
            # Relaxed catalog pass: keep weak vector hits so Python scoring can still rank top-3.
            min_sim = 0.0
        candidates: List[Dict[str, Any]] = []
        for model_id, agg in model_aggregates.items():
            avg_score = agg["score_sum"] / agg["count"] if agg["count"] > 0 else 0.0
            
            # Filter by similarity threshold
            if avg_score < min_sim:
                logger.debug(
                    f"[QdrantRetriever.search] Skipping {model_id}: "
                    f"score {avg_score:.4f} < threshold {min_sim}"
                )
                continue
            
            candidate = {
                "id": model_id,
                "score": avg_score,
                "metadata": agg["metadata"],
                "num_chunks": agg["count"],
                "point_ids": agg["point_ids"],
            }
            candidates.append(candidate)
        
        # Step 6: Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 7: Limit to top_k
        result_limit = top_k or self.config.top_k_models
        results = candidates[:result_limit]
        
        logger.info(
            f"[QdrantRetriever.search] Final results: {len(results)} models. "
            f"Top 3: {[c['id'] for c in results[:3]]}"
        )
        
        # Log top scores for debugging
        for i, candidate in enumerate(results[:5], 1):
            logger.debug(
                f"[QdrantRetriever.search] #{i}: {candidate['id']}, "
                f"score={candidate['score']:.4f}, chunks={candidate['num_chunks']}"
            )
        
        return results
    
    def _build_filter(
        self,
        metadata_filter: Optional[Dict[str, Any]],
        relax: bool = False
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from metadata constraints.
        
        Args:
            metadata_filter: Dict with keys like task_type, license, tags
            relax: If True, skip most filtering
            
        Returns:
            Qdrant Filter object or None
        """
        if not metadata_filter or relax:
            return None
        
        conditions = []
        
        # Filter by task_type (pipeline_tag)
        task_type = metadata_filter.get("task_type")
        if task_type and task_type != "general":
            logger.debug(f"[QdrantRetriever._build_filter] task_type filter: {task_type}")
            conditions.append(
                FieldCondition(
                    key="pipeline_tag",
                    match=MatchValue(value=task_type)
                )
            )
        
        # Filter by license
        license_val = metadata_filter.get("license")
        if license_val:
            logger.debug(f"[QdrantRetriever._build_filter] license filter: {license_val}")
            conditions.append(
                FieldCondition(
                    key="license",
                    match=MatchValue(value=license_val)
                )
            )
        
        # Filter by tags
        tags = metadata_filter.get("tags")
        if tags:
            if isinstance(tags, str):
                tags = [tags]
            logger.debug(f"[QdrantRetriever._build_filter] tags filter: {tags}")
            for tag in tags:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )
        
        if not conditions:
            return None
        
        # Single condition
        if len(conditions) == 1:
            return conditions[0]
        
        # Multiple conditions combined with AND
        return Filter(must=conditions)



def run(
    state: RecommendationState,
    config: Optional[RetrieverConfig] = None,
    refine: bool = False
) -> RecommendationState:
    """
    Retrieve candidate models from Qdrant (pipeline integration).
    
    This function uses QdrantRetriever internally and updates the RecommendationState
    with retrieved models.
    
    Args:
        state: RecommendationState with user_query and extracted constraints
        config: RetrieverConfig (uses defaults if None)
        refine: If True, increase search limit and relax filters
        
    Returns:
        Updated state with retrieved_models populated
    """
    
    if config is None:
        config = RetrieverConfig()
    
    logger.info(
        f"[retriever.run] Starting retrieval. refine={refine}, "
        f"task_type={state.task_type}, constraints={state.constraints}"
    )
    
    retriever = QdrantRetriever(config)
    
    metadata_filter: Optional[Dict[str, Any]] = None
    if config.apply_qdrant_structured_filter:
        mf: Dict[str, Any] = {}
        if state.task_type:
            mf["task_type"] = state.task_type
        if "license" in state.constraints:
            mf["license"] = state.constraints["license"]
        if "tags" in state.constraints:
            mf["tags"] = state.constraints["tags"]
        metadata_filter = mf if mf else None
    
    try:
        retrieved_models = retriever.search(
            query=state.effective_search_query(),
            metadata_filter=metadata_filter,
            relax_filters=refine
        )
    except Exception as e:
        logger.error(f"[retriever.run] Retrieval failed: {e}")
        state.agent_logs.append(f"Retriever: Search failed: {e}")
        return state
    
    # Update state
    state.retrieved_models = retrieved_models
    state.retrieval_complete = True
    
    top_id = retrieved_models[0]["id"] if retrieved_models else "N/A"
    state.agent_logs.append(
        f"Retriever: Retrieved {len(retrieved_models)} models. Top: {top_id}"
    )
    
    logger.info(f"[retriever.run] Retrieval complete: {len(retrieved_models)} models")
    return state
