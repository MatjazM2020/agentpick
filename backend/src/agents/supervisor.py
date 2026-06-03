"""
Supervisor agent.

Orchestrates the recommendation pipeline with bounded autonomy:
Requirements Analyst -> Retriever -> Evaluator -> [refinement loop, max 2
iterations] -> Synthesizer. Refinement triggers when the top score is below
threshold, variance is low, or there are too few candidates. All decisions,
iteration counts, and quality metrics are logged.
"""

import logging
from typing import Optional, Tuple



from src.agents import evaluator, requirements_analyst, retriever
from src.core import postgres
from src.core.state import RecommendationState
from src.core.config import AgentConfig, ScoringConfig, RetrieverConfig
from src.core.query_specificity import (
    should_stop_for_query_refinement,
    missing_requirement_slots,
    fallback_refinement_text,
    no_retrieval_hits_message,
)
from src.agents import synthesizer
from src.agents.refinement_advisor import run as refinement_advisor_run


logger = logging.getLogger(__name__)


def _compute_quality_metrics(
    state: RecommendationState,
    config: AgentConfig
) -> Tuple[float, dict]:
    """
    Compute a quality score and metrics to decide if refinement is needed.

    Checks top score vs threshold, top-5 variance, and candidate count.
    Returns ``(quality_score [0,1], metrics_dict)``.
    """
    metrics = {
        "num_candidates": len(state.scored_models),
        "top_score": 0.0,
        "top_5_avg": 0.0,
        "variance": 0.0,
        "reasons_to_refine": []
    }
    
    if not state.scored_models:
        metrics["reasons_to_refine"].append("No candidates retrieved")
        return 0.0, metrics
    
    scores = [m.score for m in state.scored_models]
    metrics["top_score"] = scores[0]

    if scores[0] < config.quality_threshold:
        metrics["reasons_to_refine"].append(
            f"Top score {scores[0]:.3f} < threshold {config.quality_threshold}"
        )

    if len(scores) >= 5:
        top_5_scores = scores[:5]
        metrics["top_5_avg"] = sum(top_5_scores) / len(top_5_scores)
        variance = sum((s - metrics["top_5_avg"]) ** 2 for s in top_5_scores) / len(top_5_scores)
        metrics["variance"] = variance
        if variance < 0.001:
            metrics["reasons_to_refine"].append(
                f"Low variance in top-5 ({variance:.6f}), hard to differentiate"
            )

    if len(state.scored_models) < 3:
        metrics["reasons_to_refine"].append(
            f"Too few candidates ({len(state.scored_models)} < 3)"
        )

    # Higher score => higher quality, no refinement needed
    quality_score = scores[0] * 0.7 + (1.0 if metrics["variance"] > 0.001 else 0.5) * 0.3

    return quality_score, metrics


def _should_refine(
    quality_score: float,
    metrics: dict,
    config: AgentConfig
) -> bool:
    """Refine if any refinement reason exists and auto-refinement is enabled."""
    should_refine = len(metrics["reasons_to_refine"]) > 0 and config.auto_refine_on_low_confidence
    
    logger.info(
        f"[Supervisor] Quality assessment: score={quality_score:.3f}, "
        f"should_refine={should_refine}, "
        f"reasons={metrics['reasons_to_refine']}"
    )
    
    return should_refine


def _log_iteration(
    state: RecommendationState,
    phase: str,
    details: dict
) -> None:
    """Log a supervisor decision to ``state.agent_logs`` and the logger."""
    log_msg = f"[Supervisor:{phase}] iteration={state.iteration}, " + \
              ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(log_msg)
    state.agent_logs.append(log_msg)


def _popularity_mode(state: RecommendationState) -> str:
    """Return the popularity routing mode: 'popularity_only', 'hybrid', or 'none'."""
    mode = (state.popularity or {}).get("mode", "none")
    return mode if mode in ("popularity_only", "hybrid") else "none"


def _retrieve_popularity_only(
    state: RecommendationState,
    retriever_config: RetrieverConfig,
) -> bool:
    """
    Populate ``retrieved_models`` from PostgreSQL ordered by downloads/likes.

    Returns True on success. Returns False (so the caller falls back to Qdrant)
    when PostgreSQL is unavailable or has no matching rows.
    """
    pop = state.popularity or {}
    try:
        candidates = postgres.query_top_models(
            task_type=state.task_type,
            tags=state.constraints.get("tags"),
            sort_by=pop.get("sort_by"),
            min_downloads=pop.get("min_downloads"),
            min_likes=pop.get("min_likes"),
            limit=retriever_config.top_k_models,
        )
    except postgres.PostgresUnavailable as e:
        logger.warning(
            f"[Supervisor] PostgreSQL unavailable for popularity query: {e}; "
            f"falling back to Qdrant"
        )
        state.agent_logs.append(
            f"Supervisor: PostgreSQL unavailable ({e}); using Qdrant fallback"
        )
        return False

    if not candidates:
        logger.info("[Supervisor] PostgreSQL returned no popularity rows; falling back to Qdrant")
        state.agent_logs.append("Supervisor: PostgreSQL returned no rows; using Qdrant fallback")
        return False

    state.retrieved_models = candidates
    state.retrieval_complete = True
    _log_iteration(
        state, "retrieval_postgres",
        {"candidates": len(candidates), "source": "postgres", "sort_by": pop.get("sort_by") or "downloads"},
    )
    return True


def _apply_postgres_filter_rerank(state: RecommendationState) -> None:
    """
    Hybrid mode: enrich Qdrant candidates with authoritative downloads/likes from
    PostgreSQL and drop any below the requested thresholds.

    Degrades gracefully: on PostgreSQL failure or if thresholds would empty the
    list, the Qdrant candidates are kept and the popularity-weighted evaluator
    still ranks them.
    """
    pop = state.popularity or {}
    min_downloads = pop.get("min_downloads")
    min_likes = pop.get("min_likes")
    model_ids = [m["id"] for m in state.retrieved_models if m.get("id")]

    try:
        pg_meta = postgres.fetch_metadata(model_ids)
    except postgres.PostgresUnavailable as e:
        logger.warning(
            f"[Supervisor] PostgreSQL unavailable for hybrid rerank: {e}; keeping Qdrant ranking"
        )
        state.agent_logs.append(
            f"Supervisor: PostgreSQL unavailable for hybrid ({e}); kept Qdrant candidates"
        )
        return

    if not pg_meta:
        state.agent_logs.append(
            "Supervisor: hybrid PostgreSQL lookup found no matching rows; kept Qdrant candidates"
        )
        return

    kept = []
    for m in state.retrieved_models:
        meta = pg_meta.get(m["id"])
        if meta:
            # Authoritative PostgreSQL counts override the Qdrant payload values
            m["metadata"]["downloads"] = meta.get("downloads", m["metadata"].get("downloads", 0))
            m["metadata"]["likes"] = meta.get("likes", m["metadata"].get("likes", 0))
        dl = m["metadata"].get("downloads", 0) or 0
        lk = m["metadata"].get("likes", 0) or 0
        if min_downloads is not None and dl < min_downloads:
            continue
        if min_likes is not None and lk < min_likes:
            continue
        kept.append(m)

    if kept:
        dropped = len(state.retrieved_models) - len(kept)
        state.retrieved_models = kept
        _log_iteration(state, "hybrid_postgres_filter", {"kept": len(kept), "dropped": dropped})
    else:
        _log_iteration(
            state, "hybrid_postgres_filter",
            {"kept": 0, "note": "thresholds emptied list; kept Qdrant candidates"},
        )


async def run_pipeline(
    state: RecommendationState,
    agents: dict,
    config: Optional[AgentConfig] = None,
    scorer_config: Optional[ScoringConfig] = None,
    retriever_config: Optional[RetrieverConfig] = None,
) -> RecommendationState:
    """
    Run the complete recommendation pipeline with autonomous refinement.

    Requirements Analyst -> Retriever -> Evaluator -> refinement loop (max 2
    iterations when quality is low) -> Synthesizer. ``agents`` supplies the LLM
    Agent objects (requirements_analyst, refinement_advisor, synthesizer).
    Returns the updated state with the complete pipeline output.
    """
    if config is None:
        config = AgentConfig()
    if scorer_config is None:
        scorer_config = ScoringConfig()
    if retriever_config is None:
        retriever_config = RetrieverConfig()
    
    logger.info(
        f"[Supervisor] Starting pipeline for query: {state.user_query[:100]}..."
    )
    
    # === PHASE 1: Requirements Analysis ===
    state.iteration = 1
    _log_iteration(state, "requirements", {"phase": "starting"})
    
    try:
        state = await requirements_analyst.run(
            state,
            agents["requirements_analyst"],
            max_retries=config.max_analyst_retries
        )
        state.requirements_extracted = True
        _log_iteration(
            state, "requirements",
            {
                "task_type": state.task_type,
                "constraints": len(state.constraints),
                "preferences": len(state.preferences)
            }
        )
    except Exception as e:
        logger.error(f"[Supervisor] Requirements analysis failed: {e}")
        state.agent_logs.append(f"Supervisor: Requirements analysis failed: {e}")
        return state
    
    # Popularity-driven requests want a ranked list, not clarification questions.
    pop_mode = _popularity_mode(state)

    # === Underspecified query: ask follow-ups before catalog search ===
    if pop_mode == "none" and should_stop_for_query_refinement(state, config):
        _log_iteration(state, "query_refinement", {"reason": "underspecified_or_low_confidence"})
        try:
            state = await refinement_advisor_run(
                state,
                agents["refinement_advisor"],
                max_retries=config.max_synthesizer_retries,
            )
        except Exception as e:
            logger.error(f"[Supervisor] Refinement advisor failed: {e}")
            state.agent_logs.append(f"Supervisor: Refinement advisor failed: {e}")
            slots = missing_requirement_slots(state)
            state.refinement_assistant_text = fallback_refinement_text(slots)
            state.stopped_for_query_refinement = True
            state.follow_up_questions = []
        state.final_recommendations = []
        state.scored_models = []
        state.retrieved_models = []
        return state
    
    # === PHASE 2: Retrieval (popularity-aware routing) ===
    _log_iteration(state, "retrieval", {"phase": "starting", "popularity_mode": pop_mode})

    # popularity_only: query PostgreSQL directly; fall back to Qdrant if unavailable/empty.
    used_postgres = (
        _retrieve_popularity_only(state, retriever_config)
        if pop_mode == "popularity_only"
        else False
    )

    if not used_postgres:
        try:
            state = retriever.run(state, retriever_config, refine=False)
            state.retrieval_complete = True
            _log_iteration(
                state, "retrieval",
                {"candidates": len(state.retrieved_models)}
            )
        except Exception as e:
            logger.error(f"[Supervisor] Initial retrieval failed: {e}")
            state.agent_logs.append(f"Supervisor: Initial retrieval failed: {e}")
            return state

        if not state.retrieved_models:
            logger.info("[Supervisor] Initial retrieval empty; retrying with relaxed filters")
            try:
                state = retriever.run(state, retriever_config, refine=True)
                _log_iteration(
                    state, "retrieval_relaxed",
                    {"candidates": len(state.retrieved_models)},
                )
            except Exception as e:
                logger.error(f"[Supervisor] Relaxed retrieval failed: {e}")
                state.agent_logs.append(f"Supervisor: Relaxed retrieval failed: {e}")

        if not state.retrieved_models:
            state.refinement_assistant_text = no_retrieval_hits_message(state, config)
            state.stopped_for_query_refinement = True
            state.follow_up_questions = []
            state.final_recommendations = []
            state.scored_models = []
            _log_iteration(state, "retrieval", {"candidates": 0, "branch": "no_hits_refinement"})
            return state

        # hybrid: semantic candidates from Qdrant, then PostgreSQL filter/rerank by popularity.
        if pop_mode == "hybrid":
            _apply_postgres_filter_rerank(state)
    
    # === PHASE 3: Initial Evaluation ===
    _log_iteration(state, "evaluation", {"phase": "starting"})
    
    try:
        state = evaluator.run(state, scorer_config)
        state.evaluation_complete = True
        _log_iteration(
            state, "evaluation",
            {
                "scored_candidates": len(state.scored_models),
                "top_score": state.scored_models[0].score if state.scored_models else 0
            }
        )
    except Exception as e:
        logger.error(f"[Supervisor] Initial evaluation failed: {e}")
        state.agent_logs.append(f"Supervisor: Initial evaluation failed: {e}")
        return state

    # popularity_only: honor "ranked/sorted by downloads/likes" over the composite score.
    if pop_mode == "popularity_only" and state.scored_models:
        metric = "likes" if (state.popularity.get("sort_by") or "").lower() == "likes" else "downloads"
        state.scored_models.sort(key=lambda m: (m.metadata.get(metric) or 0), reverse=True)
        _log_iteration(state, "popularity_sort", {"metric": metric})

    if state.scored_models:
        top = state.scored_models[0]
        sem = float(top.score_breakdown.get("semantic_similarity", 0.0))
        comp = float(top.score)
        if sem < config.min_top_semantic_similarity or comp < config.min_top_composite_score:
            state.needs_score_refinement = True
            state.agent_logs.append(
                f"[Supervisor] Below similarity/score thresholds: "
                f"semantic_similarity={sem:.3f} (min {config.min_top_semantic_similarity}), "
                f"top_composite={comp:.3f} (min {config.min_top_composite_score})"
            )
    
    # === PHASE 4: Refinement Loop (max 2 iterations) ===
    # Skip for popularity-routed queries: relaxing vector filters does not help a
    # popularity-driven request and only adds latency.
    max_refinements = 2 if pop_mode == "none" else 0
    refinement_count = 0
    
    while refinement_count < max_refinements:
        quality_score, metrics = _compute_quality_metrics(state, config)
        
        if not _should_refine(quality_score, metrics, config):
            logger.info(
                f"[Supervisor] Quality sufficient, stopping refinement loop "
                f"(iteration {state.iteration})"
            )
            _log_iteration(state, "quality_check", {"result": "pass", "score": quality_score})
            break

        state.iteration = 2 + refinement_count
        
        _log_iteration(
            state, "quality_check",
            {"result": "fail", "score": quality_score, "reasons": metrics["reasons_to_refine"]}
        )
        
        logger.info(
            f"[Supervisor] Refinement triggered (iteration {state.iteration}): "
            f"{metrics['reasons_to_refine']}"
        )
        
        try:
            logger.info("[Supervisor] Re-running retriever with refine=True")
            state = retriever.run(state, retriever_config, refine=True)
            _log_iteration(
                state, "retrieval_refined",
                {"candidates": len(state.retrieved_models)}
            )
        except Exception as e:
            logger.error(f"[Supervisor] Refined retrieval failed: {e}")
            state.agent_logs.append(f"Supervisor: Refined retrieval failed: {e}")
            break
        
        try:
            logger.info("[Supervisor] Re-running evaluator")
            state = evaluator.run(state, scorer_config)
            _log_iteration(
                state, "evaluation_refined",
                {
                    "scored_candidates": len(state.scored_models),
                    "top_score": state.scored_models[0].score if state.scored_models else 0
                }
            )
        except Exception as e:
            logger.error(f"[Supervisor] Refined evaluation failed: {e}")
            state.agent_logs.append(f"Supervisor: Refined evaluation failed: {e}")
            break
        
        refinement_count += 1
    
    if max_refinements and refinement_count >= max_refinements:
        logger.warning(
            f"[Supervisor] Max refinement iterations ({max_refinements}) reached"
        )
        state.agent_logs.append(
            f"Supervisor: Max refinement iterations ({max_refinements}) reached"
        )
    
    # === PHASE 5: Synthesis ===
    _log_iteration(state, "synthesis", {"phase": "starting"})
    
    try:
        state = await synthesizer.run(
            state, agents["synthesizer"], top_k=config.recommendation_top_k
        )
        _log_iteration(
            state, "synthesis",
            {"explanations": len(state.final_recommendations)}
        )
    except Exception as e:
        logger.error(f"[Supervisor] Synthesis failed: {e}")
        state.agent_logs.append(f"Supervisor: Synthesis failed: {e}")
        # Don't return yet - we have scored models even without explanations
    
    if state.needs_score_refinement and state.final_recommendations:
        extras = [
            "In one sentence, what does a correct model output look like for your use case?",
            "Any hard limits on latency, VRAM/RAM, or license (e.g. Apache-2.0 only)?",
        ]
        merged = list(state.follow_up_questions or [])
        for q in extras:
            if q not in merged:
                merged.append(q)
        state.follow_up_questions = merged[:6]
    
    # === Pipeline Complete ===
    final_iteration = state.iteration
    logger.info(
        f"[Supervisor] Pipeline complete. "
        f"Iterations: {final_iteration}, "
        f"Final recommendations: {len(state.final_recommendations)}"
    )
    _log_iteration(
        state, "pipeline_complete",
        {
            "total_iterations": final_iteration,
            "refinements": refinement_count,
            "recommendations": len(state.final_recommendations),
            "scored_models": len(state.scored_models)
        }
    )
    
    return state
