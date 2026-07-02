"""
Ranker agent (Microsoft Agent Framework) — staged, evidence-weighted ranking.

Replaces marketing-heavy semantic ordering with a deterministic composite score:

  final = 0.40 * task_match
        + 0.50 * objective_evidence
        + 0.10 * community_signal

Pipeline (3 stages, stages 2 and 3 run dimension-scoring calls in parallel):
  Stage 1 — Hard filter (LLM): drop constraint violations.
  Stage 2 — Score (2 parallel LLM calls + Python): task/domain match and objective
            evidence concurrently, then deterministic community signal and weighted sum.
  Stage 3 — Explain (2 parallel LLM calls): response intro/follow-ups and
            model-card-grounded justifications concurrently.
"""

import asyncio
import json
import logging
import math
import re
from typing import Optional

from agent_framework import Agent
from pydantic import BaseModel, Field, ValidationError

from src.core.agent_output import agent_run_text
from src.core.agent_activity_log import log_activity
from src.core.config import RankerConfig
from src.core import postgres
from src.core.state import RecommendationState, ScoredModel

logger = logging.getLogger(__name__)

_MODEL_CARD_MAX_CHARS = 6000


class RankedModel(BaseModel):
    model_id: str
    score: float = Field(description="Weighted composite 0.0–1.0")
    reasons: list[str] = Field(default_factory=list)
    evidence: dict = Field(
        default_factory=dict,
        description="task_match, objective_evidence, community_signal, notes",
    )


class JustifiedModel(BaseModel):
    model_id: str
    reasons: list[str] = Field(default_factory=list)


class JustificationOutput(BaseModel):
    models: list[JustifiedModel] = Field(default_factory=list)


class HardFilterOutput(BaseModel):
    survivors: list[str] = Field(default_factory=list)
    filtered_out: list[dict] = Field(default_factory=list)


class DimensionScore(BaseModel):
    model_id: str
    score: float = Field(ge=0.0, le=1.0)
    notes: str = Field(default="", description="Internal grounding — not shown to user")


class DimensionScoresOutput(BaseModel):
    scores: list[DimensionScore] = Field(default_factory=list)


class ResponseMetaOutput(BaseModel):
    intro: str = Field(default="")
    follow_up_questions: list[str] = Field(default_factory=list)


def _normalize_semantic(retrieval_score: float) -> float:
    """Map a Qdrant cosine score from [-1, 1] to [0, 1] for display/grounding."""
    return max(0.0, min(1.0, (retrieval_score + 1.0) / 2.0))


def _format_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M downloads"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K downloads"
    return f"{n} downloads"


def _format_parameter_count(n: Optional[int]) -> Optional[str]:
    if not n or n <= 0:
        return None
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B parameters"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M parameters"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K parameters"
    return f"{n} parameters"


def _candidate_view(model: dict) -> dict:
    """Rich, catalog-grounded view of one candidate for the LLM."""
    meta = model.get("metadata", {}) or {}
    tags = meta.get("tags", []) or []
    if not isinstance(tags, list):
        tags = [tags]

    downloads = int(meta.get("downloads", 0) or 0)
    likes = int(meta.get("likes", 0) or 0)
    param_count = meta.get("parameter_count")

    view = {
        "model_id": model.get("id"),
        "semantic_score": round(_normalize_semantic(model.get("score", 0.0)), 4),
        "pipeline_tag": meta.get("pipeline_tag"),
        "library_name": meta.get("library_name"),
        "license": meta.get("license"),
        "tags": [str(t) for t in tags[:12]],
        "downloads": downloads,
        "downloads_label": _format_downloads(downloads) if downloads else None,
        "likes": likes,
        "parameter_count": param_count,
        "parameter_count_label": _format_parameter_count(param_count),
        "last_modified": meta.get("last_modified"),
        "matched_sections": model.get("matched_sections") or [],
        "card_excerpt": (model.get("card_excerpt") or "").strip()[:500] or None,
        "num_matched_chunks": int(model.get("num_chunks", 0) or 0),
    }
    return {k: v for k, v in view.items() if v not in (None, "", [], 0)}


def _truncate_model_card(text: str, max_chars: int = _MODEL_CARD_MAX_CHARS) -> str:
    card = (text or "").strip()
    if len(card) <= max_chars:
        return card
    return card[: max_chars - 3].rstrip() + "..."


def _candidate_scoring_block(model: dict, model_cards: dict[str, str]) -> dict:
    """Candidate payload for dimension scoring — includes authoritative model_card."""
    mid = model.get("id") or ""
    view = _candidate_view(model)
    card = _truncate_model_card(model_cards.get(mid, ""))
    meta = model.get("metadata", {}) or {}
    if not card:
        card = _truncate_model_card(meta.get("model_card") or "")
    return {
        "model_id": mid,
        "catalog_metadata": view,
        "model_card": card or None,
    }


def _compute_community_signal(
    downloads: int,
    likes: int,
    config: RankerConfig,
) -> float:
    """Deterministic [0, 1] score from PostgreSQL downloads/likes (log-scaled, capped)."""
    dl = min(int(downloads or 0), config.max_downloads_cap)
    lk = min(int(likes or 0), config.max_likes_cap)

    if dl > 0:
        dl_score = math.log(dl + 1) / math.log(config.max_downloads_cap + 1)
    else:
        dl_score = 0.0

    lk_score = lk / config.max_likes_cap if config.max_likes_cap > 0 else 0.0
    return round(min(1.0, 0.7 * dl_score + 0.3 * lk_score), 4)


def _combine_scores(
    task_match: float,
    objective_evidence: float,
    community_signal: float,
    config: RankerConfig,
) -> float:
    """Deterministic weighted composite in [0, 1]."""
    raw = (
        config.w_task_match * task_match
        + config.w_objective_evidence * objective_evidence
        + config.w_community_signal * community_signal
    )
    return round(max(0.0, min(1.0, raw)), 4)


def _build_hard_filter_prompt(
    candidates: list[dict],
    model_cards: dict[str, str],
    constraints: dict,
    preferences: dict,
    user_context: str,
) -> str:
    blocks = [
        _candidate_scoring_block(c, model_cards)
        for c in candidates
        if c.get("id")
    ]
    return f"""Hard-filter Hugging Face model candidates that violate EXPLICIT user requirements.

USER REQUEST:
{user_context or "(not available)"}

EXPLICIT CONSTRAINTS (hard requirements): {json.dumps(constraints) if constraints else "None"}
PREFERENCES (soft — do NOT hard-filter on these alone): {json.dumps(preferences) if preferences else "None"}

CANDIDATES:
{json.dumps(blocks, indent=2, default=str)}

Return ONLY valid JSON (no markdown):
{{
  "survivors": ["model_id strings that pass all explicit constraints"],
  "filtered_out": [{{"model_id": "string", "reason": "string"}}]
}}

Rules:
- Drop only on clear EXPLICIT constraint violations: wrong pipeline/task type, wrong modality,
  unsupported language, incompatible hardware, disallowed license.
- When unsure, keep the candidate (score it later).
- Every input model_id must appear in either survivors or filtered_out.
"""


def _build_task_match_prompt(
    survivors: list[dict],
    model_cards: dict[str, str],
    user_context: str,
    intent_summary: str,
    task_type: Optional[str],
) -> str:
    blocks = [
        _candidate_scoring_block(c, model_cards)
        for c in survivors
        if c.get("id")
    ]
    return f"""Score each candidate's TASK/DOMAIN MATCH for the user's request on [0, 1].

USER REQUEST:
{user_context or "(not available)"}

PARSED INTENT:
{intent_summary or user_context or "(not available)"}
TASK TYPE HINT: {task_type or "unknown"}

CANDIDATES (model_card is the primary source; catalog_metadata is supplementary):
{json.dumps(blocks, indent=2, default=str)}

Return ONLY valid JSON (no markdown):
{{
  "scores": [
    {{
      "model_id": "string",
      "score": 0.0,
      "notes": "internal — cite concrete training/specialization evidence or lack thereof"
    }}
  ]
}}

Scoring rules:
- Reward explicit evidence the model was trained or fine-tuned for the requested task/domain
  (e.g. Coder series for programming, math specialists, vision models, summarization fine-tunes).
- Use pipeline_tag, tags, training objective statements, and dataset mentions when present.
- IGNORE marketing language ("state-of-the-art", "best", "leading") unless backed by specifics.
- Penalize general-purpose models with no task-specific evidence when a specialist is requested.
- Score 0.0–1.0 for every survivor; one entry per model_id.
"""


def _build_objective_evidence_prompt(
    survivors: list[dict],
    model_cards: dict[str, str],
    user_context: str,
    intent_summary: str,
) -> str:
    blocks = [
        _candidate_scoring_block(c, model_cards)
        for c in survivors
        if c.get("id")
    ]
    return f"""Score each candidate's OBJECTIVE EVIDENCE on [0, 1] — facts, not marketing.

USER REQUEST:
{user_context or "(not available)"}

PARSED INTENT:
{intent_summary or user_context or "(not available)"}

CANDIDATES (model_card is the primary source):
{json.dumps(blocks, indent=2, default=str)}

Return ONLY valid JSON (no markdown):
{{
  "scores": [
    {{
      "model_id": "string",
      "score": 0.0,
      "notes": "internal — list benchmarks, numbers, datasets, architecture facts cited"
    }}
  ]
}}

Score based ONLY on verifiable content in model_card / metadata:
- Benchmark or evaluation results with numbers (MMLU, HumanEval, GSM8K, etc.)
- Named training datasets and data mixtures
- Architecture details (MoE, context length, parameter counts when stated)
- Documented capabilities: tool use, function calling, vision, code execution support
- Reproducible training details (steps, tokens, methods)

Penalize:
- Unsubstantiated superlatives and hype with no numbers or citations
- Vague claims ("high quality", "powerful") without supporting evidence

Do NOT invent benchmarks or features. Score 0.0–1.0 for every survivor; one entry per model_id.
"""


def _build_response_meta_prompt(
    top: list[RankedModel],
    user_context: str,
    top_k: int,
) -> str:
    picks = [
        {
            "model_id": r.model_id,
            "score": r.score,
            "evidence": r.evidence,
        }
        for r in top
    ]
    if top_k == 1:
        intro_example = (
            'One short line: For your task (<brief restatement>), my top recommendation is:'
        )
    else:
        intro_example = (
            'One short line: For your task (<brief restatement>), the best options are:'
        )
    return f"""Write a one-line intro and 2-3 follow-up questions for ranked model picks.

USER REQUEST:
{user_context or "(not available)"}

TOP PICKS (already ranked by evidence-weighted score):
{json.dumps(picks, indent=2, default=str)}

Return ONLY valid JSON (no markdown):
{{
  "intro": "{intro_example}",
  "follow_up_questions": [
    "2-3 short questions the user might click next — from the user's point of view"
  ]
}}
"""


def _build_justification_prompt(
    picked: list[RankedModel],
    model_cards: dict[str, str],
    candidates_by_id: dict[str, dict],
    user_context: str,
    other_model_ids: list[str],
) -> str:
    blocks: list[dict] = []
    for r in picked:
        card = _truncate_model_card(model_cards.get(r.model_id, ""))
        view = _candidate_view(candidates_by_id.get(r.model_id, {}))
        blocks.append(
            {
                "model_id": r.model_id,
                "catalog_metadata": view,
                "model_card": card or None,
                "ranking_evidence": r.evidence,
            }
        )

    blocks_json = json.dumps(blocks, indent=2, default=str)
    others = ", ".join(other_model_ids) if other_model_ids else "(none)"

    return f"""Write user-facing pick reasons for models already selected for the user.

USER REQUEST:
{user_context or "(not available)"}

OTHER PICKS IN THIS LIST (mention advantages over these when relevant):
{others}

PICKED MODELS — use the PostgreSQL ``model_card`` (Hugging Face README markdown) as the
primary source for every reason. Prefer objective evidence (benchmarks, training details,
capabilities) over marketing language. Catalog metadata is supplementary.

{blocks_json}

Return ONLY valid JSON (no markdown):
{{
  "models": [
    {{
      "model_id": "string (must match one of the picked models)",
      "reasons": [
        "Complete sentence grounded in model_card content — task fit, capabilities, or trade-offs.",
        "Another sentence citing a concrete fact from the model_card or metadata.",
        "Optional third sentence on advantages vs the other picks when useful."
      ]
    }}
  ]
}}

Rules:
- Exactly 2-3 "reasons" per model — plain English bullets the user can read.
- Ground every reason in model_card text when available; never invent benchmarks or features.
- Cite objective evidence (numbers, datasets, benchmarks) when present; avoid hype.
- Focus on why this model fits the user's request and what makes it stand out vs the other picks.
- No JSON field names, no abbreviations-only lines, max ~140 characters per reason.
- Return one entry per picked model, in the same order as above.
"""


def _sanitize_reason(text: str) -> str:
    line = re.sub(r"\s+", " ", (text or "").strip())
    if not line:
        return ""
    if re.search(r"\b(semantic_score|constraint_match|domain_evidence|evidence)\b", line, re.I):
        return ""
    if line.startswith("{") or line.startswith("["):
        return ""
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line


def _sanitize_reasons(reasons: list[str]) -> list[str]:
    cleaned: list[str] = []
    for reason in reasons:
        line = _sanitize_reason(reason)
        if line and line not in cleaned:
            cleaned.append(line)
        if len(cleaned) >= 3:
            break
    return cleaned


def _reasons_from_candidate(model_id: str, candidate: dict, user_context: str) -> list[str]:
    """Readable fallback reasons from catalog metadata when LLM output is weak."""
    view = _candidate_view(candidate)
    reasons: list[str] = []

    meta = candidate.get("metadata", {}) or {}
    model_card = (meta.get("model_card") or "").strip()
    if model_card:
        for paragraph in model_card.split("\n\n"):
            line = re.sub(r"^#+\s*", "", paragraph).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line) > 40 and not line.startswith("!["):
                reasons.append(line[:140].rstrip() + ("..." if len(line) > 140 else "."))
                break

    pipeline = view.get("pipeline_tag")
    if pipeline and len(reasons) < 3:
        reasons.append(f"Listed as a {pipeline} model on Hugging Face, matching your request.")

    param = view.get("parameter_count_label")
    if param:
        reasons.append(f"Model size is {param}, which helps estimate hardware needs.")

    downloads = view.get("downloads_label")
    if downloads:
        reasons.append(f"Widely used in the community with {downloads}.")

    sections = view.get("matched_sections") or []
    if sections:
        section_hint = ", ".join(str(s) for s in sections[:2])
        reasons.append(f"Model card sections matched your query, including {section_hint}.")

    excerpt = view.get("card_excerpt")
    if excerpt and len(reasons) < 3:
        snippet = excerpt.split(".")[0].strip()
        if len(snippet) > 20:
            reasons.append(f"Card excerpt: {snippet[:120]}.")

    if not reasons:
        score = view.get("semantic_score", 0.0)
        reasons.append(
            f"Strong semantic match to your request ({model_id}, relevance {score:.2f})."
        )

    return _sanitize_reasons(reasons)[:3]


def _default_follow_up_questions(state: RecommendationState) -> list[str]:
    models = [m.model_id for m in state.final_recommendations]
    task_hint = (state.task_type or state.user_query or "this task").strip()[:60]
    if len(models) >= 2:
        return [
            f"Which of these is the single best choice for {task_hint}?",
            "Which option works best on a GPU with 8GB VRAM?",
            f"How do {models[0]} and {models[1]} compare on speed vs quality?",
        ]
    if len(models) == 1:
        return [
            f"Are there good alternatives to {models[0]}?",
            "What hardware do I need to run this model?",
            f"Why is {models[0]} better than other options for {task_hint}?",
        ]
    return [
        "Can you recommend one model for my task?",
        "What hardware constraints should I consider?",
        "Show me popular open-source options for this use case.",
    ]


async def _run_llm_json(
    agent: Agent,
    prompt: str,
    max_retries: int,
    output_model: type[BaseModel],
) -> Optional[BaseModel]:
    """Execute a single JSON-output LLM call with retries.

    No session is used — all ranker scoring prompts are self-contained, so session
    history would only add token overhead and create race conditions under parallelism.
    """
    for attempt in range(1, max_retries + 1):
        try:
            run_result = await agent.run(prompt)
            raw = agent_run_text(run_result)
            json_str = raw.strip()
            if "```" in raw:
                start = raw.find("```json")
                start = start + 7 if start != -1 else raw.find("```") + 3
                end = raw.find("```", start)
                json_str = raw[start:end].strip() if end != -1 else json_str
            return output_model(**json.loads(json_str))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"[Ranker] Attempt {attempt}/{max_retries} parse failed ({output_model.__name__}): {e}"
            )
    return None


def _dimension_score_map(
    output: Optional[DimensionScoresOutput],
    expected_ids: set[str],
    fallback: float,
) -> dict[str, tuple[float, str]]:
    """Parse dimension scores; missing ids get ``fallback``."""
    out: dict[str, tuple[float, str]] = {}
    if output and isinstance(output, DimensionScoresOutput):
        for item in output.scores:
            if item.model_id in expected_ids:
                out[item.model_id] = (round(float(item.score), 4), item.notes or "")
    for mid in expected_ids:
        if mid not in out:
            out[mid] = (fallback, "")
    return out


async def _run_hard_filter(
    agent: Agent,
    state: RecommendationState,
    pool: list[dict],
    model_cards: dict[str, str],
    config: RankerConfig,
) -> tuple[list[dict], list[dict]]:
    """Return (survivors, filtered_out). On LLM failure, keep full pool."""
    prompt = _build_hard_filter_prompt(
        candidates=pool,
        model_cards=model_cards,
        constraints=state.constraints,
        preferences=state.preferences,
        user_context=state.natural_language_context_for_requirements(),
    )
    result = await _run_llm_json(agent, prompt, config.max_retries, HardFilterOutput)
    by_id = {m["id"]: m for m in pool if m.get("id")}
    all_ids = set(by_id.keys())

    if not result or not isinstance(result, HardFilterOutput):
        logger.warning("[Ranker] Hard filter LLM failed; keeping full pool")
        return pool, []

    survivor_ids = {mid for mid in result.survivors if mid in all_ids}
    if not survivor_ids:
        logger.warning("[Ranker] Hard filter returned no survivors; keeping full pool")
        return pool, []

    filtered = [
        {"model_id": fo.get("model_id"), "reason": fo.get("reason", "")}
        for fo in result.filtered_out
        if fo.get("model_id")
    ]
    survivors = [by_id[mid] for mid in result.survivors if mid in survivor_ids]
    # Preserve retrieval order for survivors not explicitly listed
    if len(survivors) < len(survivor_ids):
        seen = {m["id"] for m in survivors}
        for m in pool:
            if m.get("id") in survivor_ids and m["id"] not in seen:
                survivors.append(m)
    return survivors, filtered


async def _score_survivors(
    agent: Agent,
    state: RecommendationState,
    survivors: list[dict],
    model_cards: dict[str, str],
    config: RankerConfig,
) -> list[RankedModel]:
    """Run task_match and objective_evidence scoring in parallel, then compute composite scores."""
    if not survivors:
        return []

    user_context = state.natural_language_context_for_requirements()
    intent_summary = (state.intent_summary or "").strip()
    expected_ids = {m["id"] for m in survivors if m.get("id")}

    task_prompt = _build_task_match_prompt(
        survivors, model_cards, user_context, intent_summary, state.task_type
    )
    obj_prompt = _build_objective_evidence_prompt(
        survivors, model_cards, user_context, intent_summary
    )

    # Run both dimension scoring calls concurrently — they are fully independent
    task_result, obj_result = await asyncio.gather(
        _run_llm_json(agent, task_prompt, config.max_retries, DimensionScoresOutput),
        _run_llm_json(agent, obj_prompt, config.max_retries, DimensionScoresOutput),
    )
    task_scores = _dimension_score_map(task_result, expected_ids, fallback=0.0)
    obj_scores = _dimension_score_map(obj_result, expected_ids, fallback=0.0)

    ranked: list[RankedModel] = []
    for model in survivors:
        mid = model.get("id")
        if not mid:
            continue
        meta = model.get("metadata", {}) or {}
        task_match, task_notes = task_scores.get(mid, (0.0, ""))
        objective, obj_notes = obj_scores.get(mid, (0.0, ""))
        community = _compute_community_signal(
            meta.get("downloads", 0), meta.get("likes", 0), config
        )
        final = _combine_scores(task_match, objective, community, config)
        ranked.append(
            RankedModel(
                model_id=mid,
                score=final,
                evidence={
                    "task_match": task_match,
                    "objective_evidence": objective,
                    "community_signal": community,
                    "task_match_notes": task_notes,
                    "objective_evidence_notes": obj_notes,
                    "semantic_score": round(_normalize_semantic(model.get("score", 0.0)), 4),
                },
            )
        )

    ranked.sort(
        key=lambda r: (
            r.score,
            r.evidence.get("task_match", 0.0),
            r.evidence.get("objective_evidence", 0.0),
            r.evidence.get("semantic_score", 0.0),
        ),
        reverse=True,
    )
    log_activity(
        f"ranker | scored {len(ranked)} survivors "
        f"(weights: task={config.w_task_match}, obj={config.w_objective_evidence}, "
        f"community={config.w_community_signal})"
    )
    return ranked


async def _run_justification(
    agent: Agent,
    state: RecommendationState,
    picked: list[RankedModel],
    model_cards: dict[str, str],
    candidates_by_id: dict[str, dict],
    max_retries: int,
) -> dict[str, list[str]]:
    """Ground pick reasons in model_card for the final response."""
    if not picked or not model_cards:
        return {}

    user_context = state.natural_language_context_for_requirements()
    all_ids = [r.model_id for r in picked]
    prompt = _build_justification_prompt(
        picked=picked,
        model_cards=model_cards,
        candidates_by_id=candidates_by_id,
        user_context=user_context,
        other_model_ids=all_ids,
    )
    result = await _run_llm_json(agent, prompt, max_retries, JustificationOutput)
    if not result or not isinstance(result, JustificationOutput):
        return {}

    out: dict[str, list[str]] = {}
    for item in result.models:
        if item.model_id not in all_ids:
            continue
        reasons = _sanitize_reasons(item.reasons)
        if len(reasons) >= 2:
            out[item.model_id] = reasons

    if out:
        log_activity(f"ranker | justified {len(out)}/{len(picked)} picks from model_card")
        logger.info(f"[Ranker] Justified {len(out)}/{len(picked)} picks from PostgreSQL model_card")
    return out


async def _compute_justification(
    agent: Agent,
    state: RecommendationState,
    top: list[RankedModel],
    model_cards: dict[str, str],
    by_id: dict[str, dict],
    max_retries: int,
) -> dict[str, dict]:
    """Compute per-model pick reasons using pre-fetched model_cards (no extra PG call)."""
    justified = await _run_justification(agent, state, top, model_cards, by_id, max_retries)
    summaries: dict[str, dict] = {}
    for r in top:
        reasons = justified.get(r.model_id) or []
        if len(reasons) < 2 and r.model_id in by_id:
            reasons = _reasons_from_candidate(r.model_id, by_id[r.model_id], state.user_query)
        summaries[r.model_id] = {"reasons": reasons}
    return summaries


def _scored_from_candidate(model_id: str, score: float, candidate: dict) -> ScoredModel:
    return ScoredModel(
        model_id=model_id,
        score=round(max(0.0, min(1.0, score)), 4),
        semantic_score=round(_normalize_semantic(candidate.get("score", 0.0)), 4),
        metadata=dict(candidate.get("metadata", {}) or {}),
    )


def _fallback(candidates: list[dict], top_k: int, config: RankerConfig) -> list[ScoredModel]:
    """Embedding + community fallback when LLM dimension scoring is unusable."""
    scored: list[ScoredModel] = []
    for c in candidates:
        mid = c.get("id")
        if not mid:
            continue
        meta = c.get("metadata", {}) or {}
        semantic = _normalize_semantic(c.get("score", 0.0))
        community = _compute_community_signal(
            meta.get("downloads", 0), meta.get("likes", 0), config
        )
        score = _combine_scores(semantic, 0.5, community, config)
        scored.append(_scored_from_candidate(mid, score, c))
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:top_k]


async def run(
    state: RecommendationState,
    agent: Agent,
    config: Optional[RankerConfig] = None,
) -> RecommendationState:
    """Staged ranking: hard-filter → parallel score → parallel explain. Populates state."""
    config = config or RankerConfig()

    if not state.retrieved_models:
        state.scored_models = []
        state.final_recommendations = []
        state.explanations = {}
        state.response_intro = None
        state.model_summaries = {}
        return state

    # Enrich candidates with PostgreSQL metadata (blocking IO → offload to thread pool)
    pool = await asyncio.to_thread(
        postgres.enrich_candidates,
        state.retrieved_models[: config.candidate_pool_size],
    )
    by_id = {m.get("id"): m for m in pool}
    pool_ids = [m["id"] for m in pool if m.get("id")]

    # Extract model_cards from enriched metadata — enrich_candidates already fetches
    # model_card in the same query, so no second PG round-trip is needed.
    model_cards: dict[str, str] = {}
    for mid in pool_ids:
        card = ((by_id[mid].get("metadata") or {}).get("model_card") or "").strip()
        if card:
            model_cards[mid] = card

    log_activity(f"ranker | candidates={len(pool)} top_k={config.top_k}")
    logger.info(f"[Ranker] Evidence-weighted ranking over {len(pool)} candidates")

    # Stage 1: Hard filter
    survivors, filtered_out = await _run_hard_filter(agent, state, pool, model_cards, config)
    # Stage 2: Score — task_match and objective_evidence run in parallel
    ranked = await _score_survivors(agent, state, survivors, model_cards, config)

    if not ranked:
        logger.warning("[Ranker] Falling back to embedding + community ranking")
        scored = _fallback(pool, config.top_k, config)
        state.scored_models = scored
        state.final_recommendations = scored[: config.top_k]
        top_ranked = [
            RankedModel(model_id=m.model_id, score=m.score)
            for m in state.final_recommendations
        ]
        state.model_summaries = await _compute_justification(
            agent, state, top_ranked, model_cards, by_id, config.max_retries
        )
        state.explanations = {
            mid: _format_explanation_from_reasons(summary["reasons"])
            for mid, summary in state.model_summaries.items()
        }
        state.response_intro = (
            "For your task, my top recommendation is:"
            if config.top_k == 1
            else "For your task, the best options are:"
        )
        state.follow_up_questions = _default_follow_up_questions(state)
        state.evaluation_complete = True
        state.agent_logs.append("Ranker: LLM scoring unavailable; used embedding fallback")
        return state

    top = ranked[: config.top_k]
    scored = [
        _scored_from_candidate(r.model_id, r.score, by_id[r.model_id])
        for r in ranked
        if r.model_id in by_id
    ]
    state.scored_models = scored
    state.final_recommendations = scored[: config.top_k]

    # Stage 3: Explain — response_meta and justification run in parallel
    user_context = state.natural_language_context_for_requirements()
    meta_prompt = _build_response_meta_prompt(top, user_context, config.top_k)
    meta, summaries = await asyncio.gather(
        _run_llm_json(agent, meta_prompt, config.max_retries, ResponseMetaOutput),
        _compute_justification(agent, state, top, model_cards, by_id, config.max_retries),
    )

    default_intro = (
        "For your task, my top recommendation is:"
        if config.top_k == 1
        else "For your task, the best options are:"
    )
    if meta and isinstance(meta, ResponseMetaOutput):
        state.response_intro = (meta.intro or "").strip() or default_intro
        follow_ups = [q.strip() for q in meta.follow_up_questions if q.strip()]
        state.follow_up_questions = follow_ups or _default_follow_up_questions(state)
    else:
        state.response_intro = default_intro
        state.follow_up_questions = _default_follow_up_questions(state)

    state.model_summaries = summaries

    state.explanations = {
        r.model_id: _format_explanation_from_reasons(state.model_summaries[r.model_id]["reasons"])
        for r in top
        if r.model_id in state.model_summaries
    }
    state.evaluation_complete = True

    state.agent_logs.append(
        f"Ranker: filtered out {len(filtered_out)}, ranked {len(ranked)}, "
        f"top: {[(m.model_id, m.score) for m in state.final_recommendations]}"
    )
    if filtered_out:
        logger.info(f"[Ranker] Hard-filtered: {filtered_out}")
    logger.info(
        f"[Ranker] Top {config.top_k}: "
        f"{[(m.model_id, m.score) for m in state.final_recommendations]}"
    )
    return state


def _format_explanation_from_reasons(reasons: list[str]) -> str:
    return "\n".join(f"- {reason}" for reason in reasons if reason)
