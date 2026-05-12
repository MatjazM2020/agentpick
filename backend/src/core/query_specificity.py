"""
Query specificity and missing-requirement detection.

Used when the requirements phase indicates an underspecified (broad) query:
we only ask the user for information that is not already present in state.

Slots align with structured fields in project.md (task, constraints, preferences).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import AgentConfig
    from src.core.state import RecommendationState

# Stable ids passed to the Refinement Advisor (machine + prompt)
SLOT_TASK = "concrete_task"
SLOT_HARDWARE = "hardware_deployment"
SLOT_LATENCY = "latency_throughput"
SLOT_MEMORY = "memory_budget"
SLOT_LICENSE = "license_constraints"
SLOT_PREFERENCES = "speed_size_preferences"


SLOT_DESCRIPTIONS: dict[str, str] = {
    SLOT_TASK: "The ML or NLP task (e.g. summarization, code completion, classification) and domain if relevant.",
    SLOT_HARDWARE: "Deployment target (CPU-only, GPU, edge device, memory limits).",
    SLOT_LATENCY: "Latency or throughput needs (real-time, batch, approximate max latency).",
    SLOT_MEMORY: "RAM / VRAM budget or model size limits.",
    SLOT_LICENSE: "License needs (e.g. Apache-2.0, MIT, commercial-friendly).",
    SLOT_PREFERENCES: "Trade-offs such as speed vs accuracy and preferred model size.",
}


def _constraints_dict(state: "RecommendationState") -> dict:
    c = state.constraints or {}
    return c if isinstance(c, dict) else {}


def _preferences_dict(state: "RecommendationState") -> dict:
    p = state.preferences or {}
    return p if isinstance(p, dict) else {}


def _has_latency_budget(c: dict) -> bool:
    for k in (
        "max_latency_ms",
        "max_latency_seconds",
        "max_latency_s",
        "per_doc_latency_ms",
        "latency_budget_ms",
        "max_latency_per_doc_ms",
    ):
        if c.get(k) is not None:
            return True
    return False


def _has_memory_budget(c: dict) -> bool:
    for k in ("max_memory_gb", "max_vram_gb", "vram_gb", "memory_budget_gb", "max_model_size_gb"):
        if c.get(k) is not None:
            return True
    return False


def _has_hardware(c: dict) -> bool:
    for k in ("hardware", "deployment", "device", "compute"):
        v = c.get(k)
        if v is not None and str(v).strip():
            return True
    return False


def _has_license(c: dict, p: dict) -> bool:
    for k in ("license", "licenses", "license_family"):
        v = c.get(k)
        if v is not None and str(v).strip():
            return True
        v2 = p.get(k)
        if v2 is not None and str(v2).strip():
            return True
    return False


def _conversation_lower(state: "RecommendationState") -> str:
    return (state.natural_language_context_for_requirements() or state.user_query or "").lower()


def _nl_mentions_task(nl: str) -> bool:
    """True when prose already states a concrete ML / NLP use (analyst may still use task_type=general)."""
    return any(
        x in nl
        for x in (
            "chatbot",
            "chat bot",
            "dialog",
            "conversation",
            "inference",
            "classification",
            "summar",
            "translation",
            "code completion",
            "embedding",
            "retrieval",
            "question answering",
            "qa ",
            "fine-tun",
            "language model",
            " text generation",
            "token classification",
            "ner",
            "named entity",
        )
    )


def _nl_mentions_latency(nl: str) -> bool:
    return any(
        x in nl
        for x in (
            "real-time",
            "realtime",
            "real time",
            "low latency",
            "latency",
            "millisecond",
            "throughput",
            "sub-second",
            "fast response",
            "response time",
            "inference speed",
        )
    )


def _nl_mentions_memory(nl: str) -> bool:
    return any(
        x in nl
        for x in (
            "memory",
            "vram",
            "lightweight",
            "light weight",
            "small model",
            "limited ram",
            "low memory",
            "footprint",
            "gigabyte",
        )
    )


def _nl_mentions_license(nl: str) -> bool:
    return any(
        x in nl
        for x in (
            "mit",
            "apache",
            "apache-2",
            "permissive",
            "commercial",
            "license",
            "open source",
            "gpl",
            "bsd",
        )
    )


def _nl_mentions_hardware(nl: str) -> bool:
    return any(
        x in nl
        for x in (
            "cpu",
            "gpu",
            "vram",
            "edge",
            "machine",
            "device",
            "deployment",
            "on-device",
            "on device",
        )
    )


def _nl_covers_speed_and_size_preferences(nl: str, p: dict) -> bool:
    """Speed/accuracy trade-off and model size hints often appear only in prose."""
    has_speed = bool(p.get("speed_vs_accuracy")) or (
        ("latency" in nl or "speed" in nl or "fast" in nl or "prioritize" in nl or "prioritise" in nl)
        and ("accuracy" in nl or "quality" in nl)
    )
    has_size = bool(p.get("model_size")) or any(
        x in nl for x in ("lightweight", "light weight", "small", "tiny", "large model", "7b", "13b", "parameter")
    )
    return has_speed and has_size


def missing_requirement_slots(state: "RecommendationState") -> list[str]:
    """
    Return slot ids for structured fields the user has not yet provided.

    Only encodes absence of data in state — no LLM calls.
    """
    missing: list[str] = []
    nl = _conversation_lower(state)

    tt = (state.task_type or "").strip().lower()
    if (not tt or tt == "general") and not _nl_mentions_task(nl):
        missing.append(SLOT_TASK)

    c = _constraints_dict(state)
    p = _preferences_dict(state)
    if not _has_hardware(c) and not _nl_mentions_hardware(nl):
        missing.append(SLOT_HARDWARE)
    if not _has_latency_budget(c) and not _nl_mentions_latency(nl):
        missing.append(SLOT_LATENCY)
    if not _has_memory_budget(c) and not _nl_mentions_memory(nl):
        missing.append(SLOT_MEMORY)
    if not _has_license(c, p) and not _nl_mentions_license(nl):
        missing.append(SLOT_LICENSE)

    if not _nl_covers_speed_and_size_preferences(nl, p):
        if not p.get("speed_vs_accuracy") or not p.get("model_size"):
            missing.append(SLOT_PREFERENCES)

    # De-duplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for s in missing:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def should_stop_for_query_refinement(state: "RecommendationState", config: "AgentConfig") -> bool:
    """
    True when we should not run retrieval and instead return an interactive refinement message.

    Primary signal: requirements analyst confidence (same band as vague-query guidance in
    agent_factory). Optional lexical hint for very generic prompts that might still get
    middling confidence.
    """
    nl = (state.natural_language_context_for_requirements() or state.user_query or "").strip()
    slots = missing_requirement_slots(state)
    # Long, mostly-structured prompts should reach retrieval even if the LLM under-scores confidence.
    if len(nl) >= 180 and len(slots) <= 2:
        return False

    conf = state.requirements_confidence
    if conf is not None and conf < config.stop_for_query_refinement_below:
        return True

    q = nl.lower()
    if len(q.split()) <= 6:
        generic = (
            "best model",
            "best llm",
            "recommend a model",
            "any model",
            "what model",
            "which model",
            "good llm",
            "suggest a model",
        )
        if any(g in q for g in generic):
            tt = (state.task_type or "").strip().lower()
            if not tt or tt == "general":
                return True

    return False


def fallback_refinement_text(missing_slots: list[str]) -> str:
    """Deterministic copy when the Refinement Advisor JSON fails."""
    if not missing_slots:
        return (
            "We still need a bit more structured detail (task, hardware, latency, memory, "
            "license, or speed vs accuracy) to search the catalog safely. "
            "Reply with those constraints in one message."
        )
    lines = [
        "Your request is quite broad for a safe recommendation. "
        "Please add a bit more detail so we can search and rank models for you.",
        "",
        "It would help if you could clarify:",
    ]
    for slot in missing_slots[:6]:
        desc = SLOT_DESCRIPTIONS.get(slot, slot)
        lines.append(f"- {desc}")
    lines.append("")
    lines.append(
        "Reply in one message with as many of these details as you can; "
        "we will re-run recommendations using only catalog data (Qdrant)."
    )
    return "\n".join(lines)


def no_retrieval_hits_message(state: "RecommendationState", config: "AgentConfig") -> str:
    """
    When Qdrant returns no usable candidates: do not blame an already-detailed user
    for 'vagueness' if structured slots are mostly filled or confidence is high.
    """
    slots = missing_requirement_slots(state)
    nl = (state.natural_language_context_for_requirements() or "").strip()
    conf = state.requirements_confidence
    detailed_nl = len(nl) >= 120 or len(nl.split()) >= 18
    confident = conf is not None and conf >= getattr(
        config, "stop_for_query_refinement_below", 0.55
    )
    structured_ok = len(slots) <= 2

    intro = (
        "Search did not return enough indexed model sections above our thresholds for this request. "
        "That usually means metadata filters were too tight, vector similarity was low, or the "
        "index/collection has no overlapping data — not that your message lacked detail.\n\n"
    )

    if (detailed_nl or confident) and structured_ok:
        body = (
            "Your description already looks detailed. Try one of the following next:\n"
            "- Send **one primary deployment** per message (for example only the GPU batch path, "
            "or only the edge CPU path) so the embedding matches a single scenario.\n"
            "- Shorten to one paragraph focused on task + single target hardware + license.\n"
            "- If you control the index, confirm the Qdrant collection is populated and pipeline_tag "
            "values align with extracted task_type.\n\n"
            "We will use only catalog data on the next run."
        )
        return intro + body

    return intro + fallback_refinement_text(slots)
