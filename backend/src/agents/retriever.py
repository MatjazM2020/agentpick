"""
Retriever agent.

Deterministic semantic search against Qdrant (no LLM). Embeds the query, builds
metadata filters, queries the collection, deduplicates by model_id (averaging
chunk scores), sorts by relevance, and returns structured candidates.

Exposes ``QdrantRetriever`` (reusable logic) and ``run`` (pipeline integration).
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


# Cache QdrantRetriever instances (and their underlying QdrantClient) per
# connection target so we don't open a new client on every retrieval pass.
_retriever_cache: Dict[str, "QdrantRetriever"] = {}


def _get_retriever(config: RetrieverConfig) -> "QdrantRetriever":
    """Return a cached QdrantRetriever for this connection target (creates one if needed)."""
    key = f"{config.qdrant_url}|{config.qdrant_collection_name}|{config.qdrant_query_using}"
    cached = _retriever_cache.get(key)
    if cached is None:
        cached = QdrantRetriever(config)
        _retriever_cache[key] = cached
    return cached


class QdrantRetriever:
    """Standalone retriever for semantic search against Qdrant."""

    def __init__(self, config: Optional[RetrieverConfig] = None):
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

        Returns dicts with keys ``id``, ``score`` (averaged across chunks),
        ``metadata``, ``num_chunks`` and ``point_ids``. ``relax_filters`` widens
        the search and keeps weak hits so Python scoring can still rank top-3.
        """
        logger.info(f"[QdrantRetriever.search] Query: '{query[:100]}...'")

        try:
            embedding = embed(query)
            logger.info(f"[QdrantRetriever.search] Embedded query (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"[QdrantRetriever.search] Embedding failed: {e}")
            raise

        filter_obj = self._build_filter(metadata_filter, relax_filters)
        logger.info(f"[QdrantRetriever.search] Filter: {filter_obj if filter_obj else 'None'}")

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
            search_results: List[Any] = list(self.client.query_points(**qp_kwargs).points)
            logger.info(
                f"[QdrantRetriever.search] Found {len(search_results)} chunks "
                f"(limit={search_limit})"
            )
        except Exception as e:
            logger.error(f"[QdrantRetriever.search] Qdrant query failed: {e}")
            raise

        # Deduplicate by model_id, averaging scores across chunks
        model_aggregates: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "score_sum": 0.0,
                "count": 0,
                "metadata": {},
                "point_ids": [],
                "best_score": float("-inf"),
                "best_text_excerpt": "",
                "section_headers": set(),
            }
        )
        for point in search_results:
            payload = point.payload
            model_id = payload.get("model_id")
            if not model_id:
                logger.warning(f"[QdrantRetriever.search] Point {point.id} missing model_id, skipping")
                continue
            agg = model_aggregates[model_id]
            agg["score_sum"] += point.score
            agg["count"] += 1
            agg["point_ids"].append(point.id)
            if not agg["metadata"]:
                agg["metadata"] = payload
            section = (payload.get("section_header") or "").strip()
            if section:
                agg["section_headers"].add(section)
            if point.score >= agg["best_score"]:
                text = (payload.get("text") or "").strip()
                if text:
                    agg["best_text_excerpt"] = text[:500]
                agg["best_score"] = point.score

        logger.info(f"[QdrantRetriever.search] Deduplicated to {len(model_aggregates)} unique models")

        # Relaxed pass keeps weak vector hits so Python scoring can still rank top-3.
        min_sim = 0.0 if relax_filters else self.config.min_similarity_threshold
        candidates: List[Dict[str, Any]] = []
        for model_id, agg in model_aggregates.items():
            avg_score = agg["score_sum"] / agg["count"] if agg["count"] > 0 else 0.0
            if avg_score < min_sim:
                logger.debug(
                    f"[QdrantRetriever.search] Skipping {model_id}: "
                    f"score {avg_score:.4f} < threshold {min_sim}"
                )
                continue
            candidates.append({
                "id": model_id,
                "score": avg_score,
                "metadata": agg["metadata"],
                "num_chunks": agg["count"],
                "point_ids": agg["point_ids"],
                "matched_sections": sorted(agg["section_headers"])[:8],
                "card_excerpt": agg["best_text_excerpt"],
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:top_k or self.config.top_k_models]

        logger.info(
            f"[QdrantRetriever.search] Final results: {len(results)} models. "
            f"Top 3: {[c['id'] for c in results[:3]]}"
        )
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
        """Build a Qdrant filter from task_type/license/tags (None if empty or relaxed)."""
        if not metadata_filter or relax:
            return None

        conditions = []

        task_type = metadata_filter.get("task_type")
        if task_type and task_type != "general":
            logger.debug(f"[QdrantRetriever._build_filter] task_type filter: {task_type}")
            conditions.append(FieldCondition(key="pipeline_tag", match=MatchValue(value=task_type)))

        license_val = metadata_filter.get("license")
        if license_val:
            logger.debug(f"[QdrantRetriever._build_filter] license filter: {license_val}")
            conditions.append(FieldCondition(key="license", match=MatchValue(value=license_val)))

        tags = metadata_filter.get("tags")
        if tags:
            if isinstance(tags, str):
                tags = [tags]
            logger.debug(f"[QdrantRetriever._build_filter] tags filter: {tags}")
            conditions.extend(
                FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter(must=conditions)



def run(
    state: RecommendationState,
    config: Optional[RetrieverConfig] = None,
    refine: bool = False
) -> RecommendationState:
    """
    Retrieve candidate models from Qdrant (pipeline integration).

    Uses ``QdrantRetriever`` internally and populates ``state.retrieved_models``.
    ``refine`` relaxes filters and widens the search.
    """
    if config is None:
        config = RetrieverConfig()

    logger.info(
        f"[retriever.run] Starting retrieval. refine={refine}, "
        f"task_type={state.task_type}, constraints={state.constraints}"
    )

    retriever = _get_retriever(config)

    metadata_filter: Optional[Dict[str, Any]] = None
    if config.apply_qdrant_structured_filter:
        mf: Dict[str, Any] = {}
        if state.task_type:
            mf["task_type"] = state.task_type
        if "license" in state.constraints:
            mf["license"] = state.constraints["license"]
        if "tags" in state.constraints:
            mf["tags"] = state.constraints["tags"]
        metadata_filter = mf or None

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

    state.retrieved_models = retrieved_models
    state.retrieval_complete = True

    top_id = retrieved_models[0]["id"] if retrieved_models else "N/A"
    state.agent_logs.append(
        f"Retriever: Retrieved {len(retrieved_models)} models. Top: {top_id}"
    )

    logger.info(f"[retriever.run] Retrieval complete: {len(retrieved_models)} models")
    return state
