"""
Evaluator agent.

Ranks retrieved models using deterministic Python scoring.
NO LLM usage, pure reproducible scoring function.

Scoring function:
    final_score = (
        w1 * semantic_similarity +
        w2 * popularity +
        w3 * recency +
        w4 * hardware_fit +
        w5 * license_match +
        w6 * benchmark_score
    )

All features normalized to [0, 1].
Full breakdown logged and returned for explainability.
"""

import logging
from datetime import datetime
from typing import Optional
from backend.src.core.state import RecommendationState, ScoredModel
from backend.src.core.config import ScoringConfig


logger = logging.getLogger(__name__)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1].
    
    Args:
        value: The value to normalize
        min_val: Minimum of the range
        max_val: Maximum of the range
        
    Returns:
        Normalized value in [0, 1]
    """
    if max_val <= min_val:
        return 0.5  # Default to neutral if range is invalid
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _compute_semantic_similarity(
    retrieval_score: float,
    config: ScoringConfig
) -> float:
    """
    Compute semantic similarity feature [0, 1].
    
    Input: Qdrant similarity score (already in [0, 1]).
    Apply minimum threshold filter.
    
    Args:
        retrieval_score: Score from Qdrant (0-1)
        config: Scoring configuration
        
    Returns:
        Normalized semantic similarity
    """
    # Qdrant returns scores in [-1, 1] for cosine similarity
    # We normalize cosine [-1, 1] -> [0, 1]
    normalized = (retrieval_score + 1.0) / 2.0
    
    # Apply minimum threshold
    if normalized < config.min_similarity_score:
        return 0.0
    
    # Scale above threshold to [0, 1]
    return _normalize(
        normalized,
        config.min_similarity_score,
        1.0
    )


def _compute_popularity(
    metadata: dict,
    config: ScoringConfig
) -> float:
    """
    Compute popularity feature [0, 1].
    
    Uses downloads and likes with log scaling.
    
    Args:
        metadata: Model metadata from Qdrant
        config: Scoring configuration
        
    Returns:
        Normalized popularity score
    """
    downloads = metadata.get("downloads", 0)
    likes = metadata.get("likes", 0)
    
    # Cap values to prevent outlier dominance
    downloads = min(downloads, config.max_downloads_cap)
    likes = min(likes, config.max_likes_cap)
    
    # Use log scale for downloads (0 -> 0, max -> 1)
    if downloads > 0:
        import math
        download_score = math.log(downloads + 1) / math.log(config.max_downloads_cap + 1)
    else:
        download_score = 0.0
    
    # Normalize likes linearly
    like_score = _normalize(likes, 0, config.max_likes_cap)
    
    # Weighted average (downloads matter more)
    popularity = 0.7 * download_score + 0.3 * like_score
    return min(1.0, popularity)


def _compute_recency(
    metadata: dict,
    config: ScoringConfig
) -> float:
    """
    Compute recency feature [0, 1].
    
    Based on last_modified timestamp. Newer is better.
    
    Args:
        metadata: Model metadata from Qdrant
        config: Scoring configuration
        
    Returns:
        Normalized recency score
    """
    last_modified = metadata.get("last_modified")
    
    if not last_modified:
        return 0.5  # Neutral default if no date available
    
    try:
        # Parse ISO format timestamp
        if isinstance(last_modified, str):
            # Handle ISO format: 2023-11-15T10:30:00Z or similar
            if "T" in last_modified:
                modified_date = datetime.fromisoformat(
                    last_modified.replace("Z", "+00:00")
                )
            else:
                # Try basic parsing
                modified_date = datetime.strptime(last_modified[:10], "%Y-%m-%d")
        else:
            return 0.5
        
        # Calculate age in days
        now = datetime.now(modified_date.tzinfo) if modified_date.tzinfo else datetime.utcnow()
        age_days = (now - modified_date).days
        
        # Cap age at max_age_days
        age_days = min(age_days, config.max_age_days)
        
        # Newer = higher score
        recency = 1.0 - (age_days / config.max_age_days)
        return max(0.0, recency)
        
    except Exception as e:
        logger.warning(f"Failed to parse last_modified: {last_modified}, error: {e}")
        return 0.5


def _compute_hardware_fit(
    metadata: dict,
    constraints: dict,
    config: ScoringConfig
) -> float:
    """
    Compute hardware fit feature [0, 1].
    
    Compare model size/requirements against user constraints.
    
    Args:
        metadata: Model metadata from Qdrant
        constraints: User constraints (hardware preference)
        config: Scoring configuration
        
    Returns:
        Normalized hardware fit score
    """
    # Extract hardware constraint from user preferences
    user_hardware = constraints.get("hardware", "any")
    model_tags = metadata.get("tags", [])
    
    # Check for CPU/GPU compatibility
    hardware_penalty = 1.0
    
    if user_hardware == "cpu_only":
        # Penalize GPU-only models
        if any(tag in model_tags for tag in ["gpu", "cuda", "torch-cuda"]):
            hardware_penalty = config.cpu_only_penalty
    elif user_hardware == "gpu":
        # Bonus for GPU-capable models
        if any(tag in model_tags for tag in ["gpu", "cuda", "torch-cuda"]):
            hardware_penalty = 1.0
    
    # Optional: check model size from tags
    if any(tag in model_tags for tag in ["tiny", "small", "distilled"]):
        hardware_penalty *= config.small_model_bonus
    elif any(tag in model_tags for tag in ["xl", "xxl", "very_large"]):
        hardware_penalty *= config.large_model_penalty
    
    return min(1.0, max(0.0, hardware_penalty))


def _compute_license_match(
    metadata: dict,
    constraints: dict
) -> float:
    """
    Compute license match feature [0, 1].
    
    Binary or scaled match with user license constraints.
    
    Args:
        metadata: Model metadata from Qdrant
        constraints: User constraints (license preference)
        
    Returns:
        Binary or scaled license match
    """
    model_license = metadata.get("license", "unknown")
    user_licenses = constraints.get("license")  # Can be None, str, or list
    
    # If no license constraint specified, no match requirement (full score)
    if not user_licenses:
        return 1.0
    
    # Convert to list if string
    if isinstance(user_licenses, str):
        user_licenses = [user_licenses]
    
    # Direct match
    if model_license.lower() in [lic.lower() for lic in user_licenses]:
        return 1.0
    
    # Permissive license match
    permissive_licenses = {"apache-2", "apache", "mit", "bsd", "cc0"}
    if model_license.lower() in permissive_licenses:
        return 0.8  # Good but not exact match
    
    # No match
    return 0.0


def _compute_benchmark_score(
    metadata: dict
) -> float:
    """
    Compute benchmark score feature [0, 1].
    
    From model metadata if available, else default.
    
    Args:
        metadata: Model metadata from Qdrant
        
    Returns:
        Normalized benchmark score
    """
    # Look for benchmark metrics in metadata
    benchmark = metadata.get("benchmark_score", None)
    
    if benchmark is None:
        return 0.5  # Neutral default
    
    # Normalize benchmark to [0, 1]
    if isinstance(benchmark, (int, float)):
        return min(1.0, max(0.0, float(benchmark)))
    
    return 0.5


def run(
    state: RecommendationState,
    config: Optional[ScoringConfig] = None
) -> RecommendationState:
    """
    Evaluate and rank retrieved models using deterministic scoring.
    
    Computes six features for each model, applies configurable weights,
    returns full score breakdown for explainability.
    
    Args:
        state: RecommendationState with retrieved_models populated
        config: ScoringConfig (uses defaults if None)
        
    Returns:
        Updated state with scored_models populated, sorted descending
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
    
    for i, model in enumerate(state.retrieved_models):
        model_id = model.get("id")
        metadata = model.get("metadata", {})
        retrieval_score = model.get("score", 0.0)
        
        logger.info(f"[Evaluator] Processing {i+1}/{len(state.retrieved_models)}: {model_id}")
        
        # Compute all feature scores
        similarity = _compute_semantic_similarity(retrieval_score, config)
        popularity = _compute_popularity(metadata, config)
        recency = _compute_recency(metadata, config)
        hardware_fit = _compute_hardware_fit(metadata, state.constraints, config)
        license_match = _compute_license_match(metadata, state.constraints)
        benchmark = _compute_benchmark_score(metadata)
        
        # Compute weighted final score
        final_score = (
            config.w_semantic_similarity * similarity +
            config.w_popularity * popularity +
            config.w_recency * recency +
            config.w_hardware_fit * hardware_fit +
            config.w_license_match * license_match +
            config.w_benchmark_score * benchmark
        )
        
        # Store breakdown
        score_breakdown = {
            "semantic_similarity": round(similarity, 4),
            "popularity": round(popularity, 4),
            "recency": round(recency, 4),
            "hardware_fit": round(hardware_fit, 4),
            "license_match": round(license_match, 4),
            "benchmark_score": round(benchmark, 4),
        }
        
        logger.debug(
            f"[Evaluator] {model_id} scores: "
            f"similarity={similarity:.4f}, "
            f"popularity={popularity:.4f}, "
            f"recency={recency:.4f}, "
            f"hardware_fit={hardware_fit:.4f}, "
            f"license_match={license_match:.4f}, "
            f"benchmark={benchmark:.4f} "
            f"-> final={final_score:.4f}"
        )
        
        scored_model = ScoredModel(
            model_id=model_id,
            score=round(final_score, 4),
            score_breakdown=score_breakdown,
            metadata=metadata
        )
        scored.append(scored_model)
    
    # Sort by final score descending
    scored.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(
        f"[Evaluator] Evaluation complete. Top 3: "
        f"{[(m.model_id, m.score) for m in scored[:3]]}"
    )
    
    # Update state
    state.scored_models = scored
    state.evaluation_complete = True
    
    return state
