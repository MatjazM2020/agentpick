"""
Evaluator agent.

Deterministic, reproducible weighted scoring (no LLM): each feature is
normalized to [0, 1] and combined as ``sum(weight_i * feature_i)`` over the
seven features. The full numeric breakdown is logged and returned.
"""

import logging
import math
from datetime import datetime
from typing import Optional
from src.core.state import RecommendationState, ScoredModel
from src.core.config import ScoringConfig
from src.agents.evaluator_scoring import (
    build_inference_facts,
    compute_inference_profile,
    license_match_and_compliance,
    qualitative_score_phrases,
    should_apply_permissive_license_filter,
)


logger = logging.getLogger(__name__)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize ``value`` to [0, 1]; returns 0.5 (neutral) for an invalid range."""
    if max_val <= min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _compute_semantic_similarity(retrieval_score: float, config: ScoringConfig) -> float:
    """Semantic similarity feature [0, 1] from the Qdrant cosine score, thresholded."""
    normalized = (retrieval_score + 1.0) / 2.0  # cosine [-1, 1] -> [0, 1]
    if normalized < config.min_similarity_score:
        return 0.0
    return _normalize(normalized, config.min_similarity_score, 1.0)


def _compute_popularity(metadata: dict, config: ScoringConfig) -> float:
    """Popularity feature [0, 1] from log-scaled downloads and linear likes."""
    downloads = min(metadata.get("downloads", 0), config.max_downloads_cap)
    likes = min(metadata.get("likes", 0), config.max_likes_cap)

    if downloads > 0:
        download_score = math.log(downloads + 1) / math.log(config.max_downloads_cap + 1)
    else:
        download_score = 0.0

    like_score = _normalize(likes, 0, config.max_likes_cap)
    return min(1.0, 0.7 * download_score + 0.3 * like_score)


def _compute_recency(metadata: dict, config: ScoringConfig) -> float:
    """Recency feature [0, 1] from last_modified; newer is higher, 0.5 when unknown."""
    last_modified = metadata.get("last_modified")
    if not last_modified:
        return 0.5

    try:
        if isinstance(last_modified, str):
            if "T" in last_modified:
                modified_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            else:
                modified_date = datetime.strptime(last_modified[:10], "%Y-%m-%d")
        else:
            return 0.5

        now = datetime.now(modified_date.tzinfo) if modified_date.tzinfo else datetime.utcnow()
        age_days = min((now - modified_date).days, config.max_age_days)
        return max(0.0, 1.0 - (age_days / config.max_age_days))
    except Exception as e:
        logger.warning(f"Failed to parse last_modified: {last_modified}, error: {e}")
        return 0.5


def _compute_hardware_fit(metadata: dict, constraints: dict, config: ScoringConfig) -> float:
    """Hardware fit feature [0, 1] comparing model tags against the user's hardware constraint."""
    user_hardware = constraints.get("hardware", "any")
    model_tags = metadata.get("tags", [])

    hardware_penalty = 1.0
    if user_hardware == "cpu_only":
        if any(tag in model_tags for tag in ["gpu", "cuda", "torch-cuda"]):
            hardware_penalty = config.cpu_only_penalty
    elif user_hardware == "gpu":
        if any(tag in model_tags for tag in ["gpu", "cuda", "torch-cuda"]):
            hardware_penalty = 1.0

    if any(tag in model_tags for tag in ["tiny", "small", "distilled"]):
        hardware_penalty *= config.small_model_bonus
    elif any(tag in model_tags for tag in ["xl", "xxl", "very_large"]):
        hardware_penalty *= config.large_model_penalty

    return min(1.0, max(0.0, hardware_penalty))


def _compute_benchmark_score(metadata: dict) -> float:
    """Benchmark feature [0, 1] from metadata; neutral 0.5 when absent or non-numeric."""
    benchmark = metadata.get("benchmark_score", None)
    if benchmark is None:
        return 0.5
    if isinstance(benchmark, (int, float)):
        return min(1.0, max(0.0, float(benchmark)))
    return 0.5


def run(
    state: RecommendationState,
    config: Optional[ScoringConfig] = None
) -> RecommendationState:
    """
    Evaluate and rank retrieved models using deterministic weighted scoring.

    Populates ``state.scored_models`` sorted by descending composite score, each
    with a full numeric breakdown for explainability.
    """
    if config is None:
        config = ScoringConfig()

    logger.info(
        f"[Evaluator] Starting evaluation of {len(state.retrieved_models)} candidates"
    )
    logger.debug(f"[Evaluator] Scoring weights: "
        f"similarity={config.w_semantic_similarity}, "
        f"popularity={config.w_popularity}, "
        f"recency={config.w_recency}, "
        f"hardware={config.w_hardware_fit}, "
        f"license={config.w_license_match}, "
        f"benchmark={config.w_benchmark_score}"
    )

    scored = []
    nl_ctx = state.natural_language_context_for_requirements()

    for i, model in enumerate(state.retrieved_models):
        model_id = model.get("id")
        metadata = dict(model.get("metadata", {}))
        retrieval_score = model.get("score", 0.0)

        logger.info(f"[Evaluator] Processing {i+1}/{len(state.retrieved_models)}: {model_id}")

        similarity = _compute_semantic_similarity(retrieval_score, config)
        popularity = _compute_popularity(metadata, config)
        recency = _compute_recency(metadata, config)
        hardware_fit = _compute_hardware_fit(metadata, state.constraints, config)
        license_match, license_ok = license_match_and_compliance(
            metadata, state.constraints, state.preferences
        )
        metadata["_license_compliant"] = license_ok
        inference_profile = compute_inference_profile(
            model_id, metadata, state.constraints, state.preferences, state.task_type, nl_ctx,
        )
        benchmark = _compute_benchmark_score(metadata)

        final_score = (
            config.w_semantic_similarity * similarity +
            config.w_popularity * popularity +
            config.w_recency * recency +
            config.w_hardware_fit * hardware_fit +
            config.w_license_match * license_match +
            config.w_inference_profile * inference_profile +
            config.w_benchmark_score * benchmark
        )

        score_breakdown = {
            "semantic_similarity": round(similarity, 4),
            "popularity": round(popularity, 4),
            "recency": round(recency, 4),
            "hardware_fit": round(hardware_fit, 4),
            "license_match": round(license_match, 4),
            "inference_profile": round(inference_profile, 4),
            "benchmark_score": round(benchmark, 4),
        }

        logger.debug(
            f"[Evaluator] {model_id} scores: "
            f"similarity={similarity:.4f}, "
            f"popularity={popularity:.4f}, "
            f"recency={recency:.4f}, "
            f"hardware_fit={hardware_fit:.4f}, "
            f"license_match={license_match:.4f}, "
            f"inference={inference_profile:.4f}, "
            f"benchmark={benchmark:.4f} "
            f"-> final={final_score:.4f}"
        )

        score_explanations = qualitative_score_phrases(
            score_breakdown, state.constraints, license_ok
        )
        inference_facts = build_inference_facts(
            model_id, metadata, state.constraints, state.preferences, state.task_type, nl_ctx,
        )

        scored.append(ScoredModel(
            model_id=model_id,
            score=round(final_score, 4),
            score_breakdown=score_breakdown,
            metadata=metadata,
            score_explanations=score_explanations,
            inference_facts=inference_facts,
        ))

    if should_apply_permissive_license_filter(state.constraints, state.preferences):
        compliant_only = [m for m in scored if m.metadata.get("_license_compliant")]
        if compliant_only:
            dropped = len(scored) - len(compliant_only)
            if dropped:
                logger.info(
                    f"[Evaluator] Permissive license gate: dropped {dropped} non-compliant models"
                )
                state.agent_logs.append(
                    f"Evaluator: permissive license filter removed {dropped} candidates"
                )
            scored = compliant_only
        else:
            logger.warning(
                "[Evaluator] No license-compliant candidates; keeping full ranked list"
            )
            state.agent_logs.append(
                "Evaluator: no license-compliant candidates — ranking without license filter"
            )

    scored.sort(key=lambda x: x.score, reverse=True)

    logger.info(
        f"[Evaluator] Evaluation complete. Top 3: "
        f"{[(m.model_id, m.score) for m in scored[:3]]}"
    )

    state.scored_models = scored
    state.evaluation_complete = True
    return state
