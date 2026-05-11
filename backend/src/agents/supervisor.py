"""
Supervisor agent.

Orchestrates the entire recommendation pipeline with autonomous decision-making
and bounded reasoning.

Pipeline flow:
1. Requirements Analyst (extract structured requirements from query)
2. Retriever (semantic search for candidates)
3. Evaluator (deterministic scoring)
4. [Optional] Refinement loop (up to 2 iterations if quality is low)
   - Re-run Retriever with relaxed filters
   - Re-run Evaluator
5. Synthesizer (generate explanations)

CRITICAL: Bounded autonomy - max 2 refinement iterations, no infinite loops.

Quality Heuristic:
- If top score < threshold OR variance is low OR too few candidates
  -> trigger refinement
- Else -> proceed to synthesis

Decision logging:
- Log all refinement decisions
- Log iteration count
- Log quality metrics that triggered refinement
"""

import logging
from typing import Optional, Tuple



from src.agents import evaluator, requirements_analyst, retriever
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
    Compute quality metrics to decide if refinement is needed.
    
    Checks:
    - Top score (is it above threshold?)
    - Variance of top-K scores (is there good differentiation?)
    - Number of candidates (do we have enough?)
    
    Args:
        state: RecommendationState with scored_models
        config: AgentConfig with thresholds
        
    Returns:
        Tuple of (quality_score [0,1], metrics_dict for logging)
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
    
    # Check: top score above threshold
    if scores[0] < config.quality_threshold:
        metrics["reasons_to_refine"].append(
            f"Top score {scores[0]:.3f} < threshold {config.quality_threshold}"
        )
    
    # Check: variance in top 5
    if len(scores) >= 5:
        top_5_scores = scores[:5]
        metrics["top_5_avg"] = sum(top_5_scores) / len(top_5_scores)
        variance = sum((s - metrics["top_5_avg"]) ** 2 for s in top_5_scores) / len(top_5_scores)
        metrics["variance"] = variance
        
        # Low variance = hard to differentiate
        if variance < 0.001:
            metrics["reasons_to_refine"].append(
                f"Low variance in top-5 ({variance:.6f}), hard to differentiate"
            )
    
    # Check: minimum candidates
    if len(state.scored_models) < 3:
        metrics["reasons_to_refine"].append(
            f"Too few candidates ({len(state.scored_models)} < 3)"
        )
    
    # Compute overall quality score
    # Higher score = higher quality, no refinement needed
    quality_score = scores[0] * 0.7 + (1.0 if metrics["variance"] > 0.001 else 0.5) * 0.3
    
    return quality_score, metrics


def _should_refine(
    quality_score: float,
    metrics: dict,
    config: AgentConfig
) -> bool:
    """
    Decide whether to trigger refinement based on quality metrics.
    
    Args:
        quality_score: Computed quality score [0,1]
        metrics: Quality metrics dict
        config: AgentConfig with thresholds
        
    Returns:
        True if refinement should be triggered
    """
    # Simple heuristic: if any refinement reason exists, refine
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
    """
    Log a supervisor decision to state.agent_logs.
    
    Args:
        state: RecommendationState
        phase: Phase name (e.g., "requirements", "retrieval", "evaluation")
        details: Details dict
    """
    log_msg = f"[Supervisor:{phase}] iteration={state.iteration}, " + \
              ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(log_msg)
    state.agent_logs.append(log_msg)


async def run_pipeline(
    state: RecommendationState,
    agents: dict,
    config: Optional[AgentConfig] = None,
    scorer_config: Optional[ScoringConfig] = None,
    retriever_config: Optional[RetrieverConfig] = None,
) -> RecommendationState:
    """
    Run the complete recommendation pipeline with autonomous refinement.
    
    Pipeline:
    1. Requirements Analyst -> extract constraints/preferences
    2. Retriever -> search for candidates
    3. Evaluator -> score candidates
    4. [Loop] Refinement (max 2 iterations):
       - If quality is low:
         - Retriever (refine=True, relaxed filters)
         - Evaluator (score again)
    5. Synthesizer -> generate explanations
    
    Args:
        state: RecommendationState with user_query populated
        agents: Dict with Agent objects:
                {"requirements_analyst": Agent, "synthesizer": Agent}
        config: AgentConfig (uses defaults if None)
        scorer_config: ScoringConfig for evaluator
        retriever_config: RetrieverConfig for retriever
        
    Returns:
        Updated state with complete pipeline output
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
    
    # === Underspecified query: ask follow-ups before catalog search ===
    if should_stop_for_query_refinement(state, config):
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
    
    # === PHASE 2: Initial Retrieval ===
    _log_iteration(state, "retrieval", {"phase": "starting"})
    
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
    max_refinements = 2
    refinement_count = 0
    
    while refinement_count < max_refinements:
        # Check quality at current iteration before deciding to refine
        quality_score, metrics = _compute_quality_metrics(state, config)
        
        if not _should_refine(quality_score, metrics, config):
            logger.info(
                f"[Supervisor] Quality sufficient, stopping refinement loop "
                f"(iteration {state.iteration})"
            )
            _log_iteration(state, "quality_check", {"result": "pass", "score": quality_score})
            break
        
        # Quality is low, move to refinement iteration
        state.iteration = 2 + refinement_count
        
        _log_iteration(
            state, "quality_check",
            {"result": "fail", "score": quality_score, "reasons": metrics["reasons_to_refine"]}
        )
        
        logger.info(
            f"[Supervisor] Refinement triggered (iteration {state.iteration}): "
            f"{metrics['reasons_to_refine']}"
        )
        
        # Refine: Re-run retriever with relaxed filters
        try:
            logger.info(f"[Supervisor] Re-running retriever with refine=True")
            state = retriever.run(state, retriever_config, refine=True)
            _log_iteration(
                state, "retrieval_refined",
                {"candidates": len(state.retrieved_models)}
            )
        except Exception as e:
            logger.error(f"[Supervisor] Refined retrieval failed: {e}")
            state.agent_logs.append(f"Supervisor: Refined retrieval failed: {e}")
            break
        
        # Refine: Re-run evaluator
        try:
            logger.info(f"[Supervisor] Re-running evaluator")
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
    
    if refinement_count >= max_refinements:
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
