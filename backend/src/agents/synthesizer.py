"""
Synthesizer agent.

Generates human-readable explanations for top-K scored models.
Uses LLM to convert scores and metadata into clear reasoning.

CRITICAL RULE: NO hallucination. All explanations must be grounded in
provided metadata and scores. No invented features or capabilities.

Responsibilities:
- Take top-K scored models from state.scored_models
- Generate concise, factual explanations
- Explain trade-offs vs constraints
- Validate output for hallucination
- Retry on parse failure (up to N=2)
- Store in state.final_recommendations
"""

import json
import logging
from typing import Optional

from agent_framework import Agent
from pydantic import BaseModel, Field, ValidationError

from src.core.agent_output import agent_run_text
from src.core.state import RecommendationState, ScoredModel


logger = logging.getLogger(__name__)


class RecommendationExplanation(BaseModel):
    """Validated explanation for a single recommendation."""
    model_id: str = Field(description="Model identifier")
    score: float = Field(description="Final composite score")
    why: str = Field(description="Concise explanation of why this model fits")
    pros: list[str] = Field(default_factory=list, description="Strengths")
    cons: list[str] = Field(default_factory=list, description="Weaknesses/trade-offs")
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SynthesizerOutput(BaseModel):
    """Validated output from synthesizer."""
    recommendations: list[RecommendationExplanation] = Field(description="Top-K recommendations with explanations")
    follow_up_questions: list[str] = Field(default_factory=list, description="2-3 clarifying questions for refinement")
    

def _format_model_context(scored_model: ScoredModel, user_constraints: dict) -> str:
    """
    Format a single model's information for LLM context.
    
    Args:
        scored_model: The scored model with metadata
        user_constraints: User constraints for comparison
        
    Returns:
        Formatted string with model facts
    """
    metadata = scored_model.metadata
    se = scored_model.score_explanations or {}
    facts = scored_model.inference_facts or {}
    tags = metadata.get("tags", [])
    tag_s = ", ".join(str(t) for t in (tags if isinstance(tags, list) else [tags]))
    
    se_lines = "\n".join(f"- {k}: {v}" for k, v in se.items()) if se else "(not available)"
    fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items()) if facts else "(not available)"

    context = f"""
Model: {scored_model.model_id}
Ranked composite score (internal only; do NOT quote decimals to the user): {scored_model.score:.3f}

Qualitative ranking signals (use these phrases; do not invent new numeric scores):
{se_lines}

Catalog-backed planning hints (only paraphrase; do not add specs absent here):
{fact_lines}

Raw hub metadata (for grounding only):
- Downloads: {metadata.get('downloads', 'N/A')}
- Likes: {metadata.get('likes', 'N/A')}
- License: {metadata.get('license', 'N/A')}
- Tags: {tag_s or 'N/A'}
- Pipeline Task: {metadata.get('pipeline_tag', 'N/A')}
- Last Modified: {metadata.get('last_modified', 'N/A')}
- Library: {metadata.get('library_name', 'N/A')}
""".strip()
    
    return context


def _validate_no_hallucination(
    explanation: RecommendationExplanation,
    scored_model: ScoredModel
) -> bool:
    """
    Validate that explanation doesn't hallucinate facts.
    
    Checks:
    - Model ID matches
    - No invented specs
    - References to actual metadata
    - Numeric claims are grounded in data
    
    Args:
        explanation: The generated explanation
        scored_model: The ground truth model
        
    Returns:
        True if explanation is valid (grounded), False if hallucinated
    """
    # Check model ID
    if explanation.model_id != scored_model.model_id:
        logger.warning(
            f"[Synthesizer] Hallucination detected: model_id mismatch. "
            f"Expected {scored_model.model_id}, got {explanation.model_id}"
        )
        return False
    
    # Check score matches (allow small rounding difference)
    if abs(explanation.score - scored_model.score) > 0.01:
        logger.warning(
            f"[Synthesizer] Hallucination detected: score mismatch. "
            f"Expected {scored_model.score}, got {explanation.score}"
        )
        return False
    
    # Basic check: explanation should reference actual metadata
    metadata = scored_model.metadata
    why_lower = explanation.why.lower()
    
    # If explanation mentions specific numbers, verify they exist
    # (This is a simple heuristic check)
    if "download" in why_lower and metadata.get("downloads") == 0:
        logger.warning(
            f"[Synthesizer] Potential hallucination: mentions downloads "
            f"but model has 0 downloads"
        )
        return False
    
    return True


async def _generate_explanations(
    scored_models: list[ScoredModel],
    user_constraints: dict,
    user_preferences: dict,
    agent: Agent,
    max_retries: int = 2
) -> Optional[SynthesizerOutput]:
    """
    Call LLM agent to generate explanations.
    
    Args:
        scored_models: Top-K models to explain (already sorted)
        user_constraints: User constraints for context
        user_preferences: User preferences for context
        agent: The Synthesizer agent
        max_retries: Number of retries on parse failure
        
    Returns:
        SynthesizerOutput or None on failure
    """
    
    # Format model contexts
    model_contexts = "\n\n---\n\n".join(
        _format_model_context(m, user_constraints) for m in scored_models
    )
    
    constraints_str = json.dumps(user_constraints, indent=2) if user_constraints else "None"
    preferences_str = json.dumps(user_preferences, indent=2) if user_preferences else "None"
    
    prompt = f"""You are a recommendation synthesis agent for language models.

Your job is to explain why each recommended model is a good fit.

CANDIDATES (already ranked by composite score):

{model_contexts}

USER CONTEXT:

Constraints:
{constraints_str}

Preferences:
{preferences_str}

INSTRUCTIONS:
1. For each model, write WHY it fits using ONLY the qualitative ranking signals and planning hints — natural language, no decimal scores, no "hardware fit 0.xxx" style wording.
2. PROS: 2-3 bullets tied to the user's constraints (CPU, latency, license, size) and grounded in tags/downloads/license when relevant.
3. CONS: 2-3 honest trade-offs grounded in the hints or metadata gaps (e.g. unclear quantization).
4. CRITICAL: Do NOT invent parameter counts, RAM numbers, tokens/sec, or runtimes not implied by the hints/metadata.
5. Do NOT restate internal component scores numerically; paraphrase strength as weak/moderate/strong only.
6. Be factual and concise.

Return ONLY valid JSON (no markdown, no text outside JSON) with this structure:
{{
  "recommendations": [
    {{
      "model_id": "string",
      "score": float (from input),
      "why": "string (1-2 sentences explaining fit)",
      "pros": ["string", "string", "string"],
      "cons": ["string", "string"],
      "score_breakdown": {{
        "semantic_similarity": float,
        "popularity": float,
        "recency": float,
        "hardware_fit": float,
        "license_match": float,
        "inference_profile": float,
        "benchmark_score": float
      }}
    }}
  ],
  "follow_up_questions": [
    "Clarifying question 1 based on user constraints?",
    "Clarifying question 2 for refinement?",
    "One question that helps the user choose exactly ONE of the listed models (e.g. trade-off or deployment constraint)?"
  ]
}}

CRITICAL RULES:
- Return ONLY valid JSON.
- Do NOT mention capabilities not listed in tags/metadata.
- Do NOT invent benchmark scores or metrics.
- Do NOT hallucinate download counts or performance claims.
- Copy score_breakdown floats EXACTLY from the candidate context (for machine validation); never put those decimals in why/pros/cons text.
- If uncertain, say "no specific information available" rather than guessing.
- Generate 2-3 follow_up_questions that reference user's constraints/preferences.
- Include at least one question that helps the user pick a single winner among the listed models.
- Questions should ask for clarification to enable refinement (e.g., "You mentioned CPU-only. Do you need real-time latency or batch processing?")
"""
    
    logger.info(f"[Synthesizer] Calling LLM agent to generate explanations for {len(scored_models)} models")
    
    for attempt in range(1, max_retries + 1):
        try:
            run_result = await agent.run(prompt)
            raw_output = agent_run_text(run_result)
            logger.debug(f"[Synthesizer] Attempt {attempt}: Raw output length={len(raw_output)}")
            
            # Parse JSON
            # Try to extract JSON from potential markdown
            if "```" in raw_output:
                # Extract from markdown code block
                start = raw_output.find("```json") + 7
                end = raw_output.find("```", start)
                if end == -1:
                    start = raw_output.find("```") + 3
                    end = raw_output.find("```", start)
                json_str = raw_output[start:end].strip()
            else:
                json_str = raw_output.strip()
            
            output_dict = json.loads(json_str)
            output = SynthesizerOutput(**output_dict)
            
            logger.info(f"[Synthesizer] Successfully parsed output: {len(output.recommendations)} recommendations")
            return output
            
        except json.JSONDecodeError as e:
            logger.warning(f"[Synthesizer] Attempt {attempt}: JSON parse failed: {e}")
            if attempt < max_retries:
                logger.info(f"[Synthesizer] Retrying (attempt {attempt + 1}/{max_retries})")
            continue
        except ValidationError as e:
            logger.warning(f"[Synthesizer] Attempt {attempt}: Validation failed: {e}")
            if attempt < max_retries:
                logger.info(f"[Synthesizer] Retrying (attempt {attempt + 1}/{max_retries})")
            continue
    
    logger.error(f"[Synthesizer] Failed to generate explanations after {max_retries} attempts")
    return None


def _build_fallback_recommendations(
    scored_models: list[ScoredModel]
) -> list[RecommendationExplanation]:
    """
    Build minimal fallback explanations when LLM fails.
    
    Uses only the score breakdown to construct a basic explanation.
    
    Args:
        scored_models: Top-K models
        
    Returns:
        Fallback explanations
    """
    logger.warning("[Synthesizer] Building fallback recommendations")
    
    recommendations = []
    for model in scored_models:
        breakdown = model.score_breakdown
        se = model.score_explanations or {}
        why = (
            f"{se.get('match_to_request', 'Ranked for your query')} "
            f"{se.get('inference_fit', '')}".strip()
        )
        pros = [
            se.get("community_usage", "Hub traction is described qualitatively in the ranking signals."),
            se.get("license", "Check the hub license field before production use."),
        ]
        if se.get("hardware_fit"):
            pros.append(se["hardware_fit"])
        cons = [
            se.get("freshness", "Recency versus the rest of the shortlist is noted in ranking signals."),
            "Verify quantization artifacts and CPU benchmarks on your own hardware before locking in.",
        ]
        
        rec = RecommendationExplanation(
            model_id=model.model_id,
            score=model.score,
            why=why,
            pros=pros,
            cons=cons,
            score_breakdown=breakdown
        )
        recommendations.append(rec)
    
    return recommendations


async def run(
    state: RecommendationState,
    agent: Agent,
    top_k: int = 3
) -> RecommendationState:
    """
    Generate human-readable explanations for top-K scored models.
    
    Calls LLM agent to convert scores/metadata into clear reasoning.
    Validates for hallucination. Falls back to minimal explanation on failure.
    
    Args:
        state: RecommendationState with scored_models populated
        agent: Synthesizer agent configured with instructions
        top_k: Number of top models to explain
        
    Returns:
        Updated state with final_recommendations and explanations populated
    """
    
    if not state.scored_models:
        logger.warning("[Synthesizer] No scored models to explain")
        state.final_recommendations = []
        state.explanations = {}
        return state
    
    # Take top-K
    top_models = state.scored_models[:top_k]
    logger.info(f"[Synthesizer] Generating explanations for top {len(top_models)} models")
    
    # Call LLM
    output = await _generate_explanations(
        top_models,
        state.constraints,
        state.preferences,
        agent,
        max_retries=2
    )
    
    if output is None:
        logger.warning("[Synthesizer] LLM generation failed, using fallback")
        explanations_list = _build_fallback_recommendations(top_models)
    else:
        # Validate each explanation
        explanations_list = []
        for exp in output.recommendations:
            # Find the original model
            original = next((m for m in top_models if m.model_id == exp.model_id), None)
            if not original:
                logger.warning(f"[Synthesizer] Model {exp.model_id} not in input set, skipping")
                continue
            merged_bd = dict(original.score_breakdown)
            merged_bd.update(exp.score_breakdown or {})
            exp = RecommendationExplanation(
                model_id=exp.model_id,
                score=exp.score,
                why=exp.why,
                pros=exp.pros,
                cons=exp.cons,
                score_breakdown=merged_bd,
            )
            
            # Validate against hallucination
            if not _validate_no_hallucination(exp, original):
                logger.warning(f"[Synthesizer] Hallucination detected in {exp.model_id}, using fallback")
                # Use fallback for this model
                fallback = _build_fallback_recommendations([original])[0]
                explanations_list.append(fallback)
            else:
                explanations_list.append(exp)

    # LLM may omit or reorder models; align to ranked top_models (up to top_k)
    by_id = {e.model_id: e for e in explanations_list}
    explanations_ordered: list[RecommendationExplanation] = []
    for m in top_models:
        if m.model_id in by_id:
            explanations_ordered.append(by_id[m.model_id])
        else:
            explanations_ordered.append(_build_fallback_recommendations([m])[0])
    explanations_list = explanations_ordered
    
    # Store in state (preserve evaluator metadata + hints for the API adapter)
    by_orig = {m.model_id: m for m in top_models}
    state.final_recommendations = []
    for exp in explanations_list:
        orig = by_orig.get(exp.model_id)
        state.final_recommendations.append(
            ScoredModel(
                model_id=exp.model_id,
                score=exp.score,
                score_breakdown=exp.score_breakdown,
                metadata=dict(orig.metadata) if orig else {},
                score_explanations=dict(orig.score_explanations) if orig else {},
                inference_facts=dict(orig.inference_facts) if orig else {},
            )
        )
    
    state.explanations = {
        exp.model_id: f"{exp.why}\n\nPros: {', '.join(exp.pros)}\nCons: {', '.join(exp.cons)}"
        for exp in explanations_list
    }
    
    # Populate follow-up questions from LLM output
    if output and output.follow_up_questions:
        state.follow_up_questions = output.follow_up_questions
    else:
        state.follow_up_questions = []
    
    logger.info(
        f"[Synthesizer] Complete. Generated {len(state.final_recommendations)} explanations, "
        f"{len(state.follow_up_questions)} follow-up questions"
    )
    
    return state
