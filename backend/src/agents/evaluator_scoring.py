"""
Deterministic scoring helpers: license compliance, CPU/inference heuristics,
natural-language score bands, and inference fact lines (no LLM).

Grounded only on Qdrant payload fields (tags, license, downloads, model_id, text).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_PERMISSIVE_MODEL_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "apache 2.0",
        "apache-2",
        "apache2",
        "apache",
        "bsd",
        "bsd-3-clause",
        "bsd-2-clause",
        "isc",
        "cc0-1.0",
        "cc0",
        "unlicense",
        "openrail",
    }
)
_RESTRICTIVE_MARKERS = (
    "gpl",
    "agpl",
    "cc-by-nc",
    "proprietary",
    "lgpl",
)


def _tags_blob(metadata: dict) -> str:
    tags = metadata.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts = [str(t).lower() for t in tags]
    parts.append(str(metadata.get("library_name", "")).lower())
    parts.append(str(metadata.get("pipeline_tag", "")).lower())
    parts.append(str(metadata.get("text", ""))[:2000].lower())
    return " ".join(parts)


def user_requires_permissive_license(constraints: dict, preferences: dict) -> bool:
    """True when the user explicitly asked for MIT/Apache/permissive/commercial-safe licenses."""
    blobs: list[str] = []
    lic = constraints.get("license")
    if lic:
        if isinstance(lic, list):
            blobs.extend(str(x).lower() for x in lic)
        else:
            blobs.append(str(lic).lower())
    for k in ("license", "licenses", "license_family"):
        v = preferences.get(k)
        if v:
            blobs.append(str(v).lower())
    joined = " ".join(blobs)
    if not joined.strip():
        return False
    return bool(
        re.search(
            r"\b(mit|apache|apache-2|apache2|bsd|cc0|unlicense|permissive|commercial-friendly|open-source)\b",
            joined.replace("_", "-"),
        )
    )


def _normalize_hub_license(raw: str) -> str:
    return (raw or "unknown").strip().lower()


def _model_license_is_restrictive(lic: str) -> bool:
    return any(m in lic for m in _RESTRICTIVE_MARKERS)


def _model_license_is_permissive(lic: str) -> bool:
    if not lic or lic in ("unknown", "other", "none", ""):
        return False
    if _model_license_is_restrictive(lic):
        return False
    normalized = lic.replace("_", "-")
    return any(p in normalized for p in _PERMISSIVE_MODEL_LICENSES) or lic in _PERMISSIVE_MODEL_LICENSES


def _parse_license_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    s = str(raw).strip().lower()
    if not s:
        return []
    return [x.strip() for x in re.split(r"[,/]|\s+or\s+", s) if x.strip()]


def should_apply_permissive_license_filter(constraints: dict, preferences: dict) -> bool:
    """When True, drop non-compliant licenses from the ranked list if any compliant remain."""
    return _user_intent_permissive_or_mit_apache_list(constraints, preferences)


def _user_intent_permissive_or_mit_apache_list(constraints: dict, preferences: dict) -> bool:
    """True for explicit permissive wording OR typical MIT/Apache style lists."""
    if user_requires_permissive_license(constraints, preferences):
        return True
    listed = _parse_license_list(constraints.get("license"))
    tokens = " ".join(listed)
    return bool(re.search(r"\b(mit|apache|bsd|cc0|unlicense|permissive)\b", tokens))


def license_match_and_compliance(
    metadata: dict,
    constraints: dict,
    preferences: dict,
) -> Tuple[float, bool]:
    """
    Returns (license_match_score, compliant_with_user_license_intent).

    When the user asks for permissive / MIT / Apache-style licensing, models that
    are restrictive or have an unknown hub license are marked non-compliant so they
    can be excluded whenever at least one compliant candidate exists.
    """
    model_lic = _normalize_hub_license(str(metadata.get("license", "unknown")))
    listed = _parse_license_list(constraints.get("license"))
    wants_permissive_family = _user_intent_permissive_or_mit_apache_list(
        constraints, preferences
    )

    if not wants_permissive_family and not listed:
        return 1.0, True

    if wants_permissive_family:
        if _model_license_is_permissive(model_lic):
            return 1.0, True
        if _model_license_is_restrictive(model_lic):
            return 0.0, False
        return 0.0, False

    # Specific non-permissive list (unusual): require exact-ish match
    if listed:
        if model_lic in listed or any(model_lic.startswith(x) or x in model_lic for x in listed):
            return 1.0, True
        return 0.0, False

    return 1.0, True


def estimate_param_billions(model_id: str, metadata: dict) -> Optional[float]:
    """Best-effort parameter count in billions from id / card text (may be None)."""
    blob = f"{model_id} {_tags_blob(metadata)}".lower()
    for pattern, divisor in (
        (r"(\d+\.?\d*)\s*b\b", 1.0),
        (r"(?<![a-z])(\d+)m\b", 1000.0),
        (r"(\d+)\s*billion", 1.0),
    ):
        m = re.search(pattern, blob)
        if m:
            try:
                return float(m.group(1)) / divisor
            except ValueError:
                pass
    return None


def _chat_or_instruct_context(task_type: Optional[str], nl: str) -> bool:
    t = (task_type or "").lower()
    nl = nl.lower()
    hints = (
        "chat",
        "chatbot",
        "dialog",
        "conversation",
        "instruct",
        "assistant",
        "real-time",
        "realtime",
        "inference",
    )
    return any(h in t for h in ("chat", "text-generation", "conversational", "dialog")) or any(
        h in nl for h in hints
    )


def user_prioritizes_minimal_size(preferences: dict, constraints: dict) -> bool:
    p = preferences.get("model_size")
    if isinstance(p, str) and p.strip().lower() in ("tiny", "smallest", "minimal"):
        return True
    if preferences.get("prioritize_minimal_size") is True:
        return True
    if constraints.get("prioritize_minimal_size") is True:
        return True
    return False


def compute_inference_profile(
    model_id: str,
    metadata: dict,
    constraints: dict,
    preferences: dict,
    task_type: Optional[str],
    nl_context: str,
) -> float:
    """
    Heuristic [0,1] for CPU-friendly quantization, instruct/chat fit, and mature small models.
    Grounded on tags / id substrings only.
    """
    tags = _tags_blob(metadata)
    mid = model_id.lower()
    score = 0.42

    cpu = (constraints.get("hardware") or "").lower() in ("cpu_only", "cpu", "cpu-only", "edge")
    if not cpu:
        cpu = "cpu" in nl_context.lower() and "gpu" not in nl_context.lower()

    if any(x in tags for x in ("gguf", "q4", "q5", "q8", "quantized", "onnx")):
        score += 0.18
    if "llama.cpp" in tags or "llamacpp" in tags.replace(".", "") or "ctranslate" in tags:
        score += 0.12
    if any(x in tags or x in mid for x in ("instruct", "chat", "it-", "-it", "chatml")):
        score += 0.10

    family_hits = (
        "tinyllama",
        "phi-3",
        "phi3",
        "phi-2",
        "qwen2.5",
        "qwen2",
        "qwen3",
        "smolm",
        "granite",
    )
    if any(f in mid for f in family_hits):
        score += 0.12

    params_b = estimate_param_billions(model_id, metadata)
    chat = _chat_or_instruct_context(task_type, nl_context)
    tiny_penalty = 1.0
    if chat and params_b is not None and params_b < 0.5 and not user_prioritizes_minimal_size(
        preferences, constraints
    ):
        # <500M parameters: penalize for chat quality unless user wants smallest footprint
        tiny_penalty = 0.78
    elif chat and params_b is not None and 0.5 <= params_b <= 8:
        score += 0.06

    if cpu:
        score += 0.06

    score *= tiny_penalty
    return max(0.0, min(1.0, score))


def build_inference_facts(
    model_id: str,
    metadata: dict,
    constraints: dict,
    preferences: dict,
    task_type: Optional[str],
    nl_context: str,
) -> dict[str, str]:
    """Short factual lines for UI; conservative wording when data is missing."""
    tags = metadata.get("tags") or []
    tag_s = ", ".join(str(t) for t in (tags if isinstance(tags, list) else [tags]))
    lic = str(metadata.get("license", "unknown"))
    params_b = estimate_param_billions(model_id, metadata)
    blob = _tags_blob(metadata)

    lines: dict[str, str] = {}
    if params_b is not None:
        if params_b < 1:
            lines["parameter_count"] = f"Roughly {int(params_b * 1000)}M parameters (inferred from the model id / card text)."
        else:
            lines["parameter_count"] = f"Roughly {params_b:.1f}B parameters (inferred from the model id / card text)."
    else:
        lines["parameter_count"] = "Parameter count not clear from catalog metadata; check the model card for exact sizes."

    if any(x in blob for x in ("gguf", "q4_", "q5_", "q8_", "q4k", "q5k")):
        lines["quantization"] = "Card/tags mention GGUF or quantized weights — suitable stacks often use Q4/Q5 class files when published."
        lines["recommended_quantization"] = (
            "If you use GGUF with llama.cpp-style runners, Q4_K_M or Q5_K_M is a common CPU starting point; confirm available files on the hub."
        )
    elif "safetensors" in blob:
        lines["quantization"] = "Safetensors appears in metadata; GGUF may or may not be published separately."
        lines["recommended_quantization"] = "If no GGUF is listed, plan quantization in your own pipeline or use vendor CPU builds when available."
    else:
        lines["quantization"] = "Quantization format not obvious from indexed tags alone."
        lines["recommended_quantization"] = "Inspect the model repository for GGUF/ONNX artifacts before picking a runtime."

    if params_b is not None and params_b < 8:
        # Very rough order-of-magnitude for planning only
        qram = max(0.2, params_b * 0.65)
        lines["quantized_ram"] = (
            f"Very rough planning range ~{qram:.1f}–{qram + 0.4:.1f} GB system RAM for Q4-class weights at this scale (verify against real files)."
        )
    else:
        lines["quantized_ram"] = "RAM for quantized weights depends on chosen quantization; confirm from published artifacts."

    if "llama.cpp" in blob or "gguf" in blob:
        lines["cpu_performance"] = (
            "Tags suggest llama.cpp / GGUF-friendly workflows; real tokens/sec depend on your CPU, thread count, and exact quant."
        )
    else:
        lines["cpu_performance"] = (
            "CPU throughput is not in the index; expect to benchmark with your chosen runtime (e.g. llama.cpp, ONNX) on your hardware."
        )

    lines["runtimes"] = (
        f"Library / pipeline hints: {metadata.get('library_name', 'n/a')} / {metadata.get('pipeline_tag', 'n/a')}. "
        f"Tags: {tag_s or 'none listed'}."
    )
    lines["license"] = lic

    return lines


def qualitative_score_phrases(
    breakdown: dict[str, float],
    constraints: dict,
    license_compliant: bool,
) -> dict[str, str]:
    """User-facing short phrases — no decimal component scores."""

    def pick(v: float, hi: float, mid: float, strong: str, moderate: str, weak: str) -> str:
        if v >= hi:
            return strong
        if v >= mid:
            return moderate
        return weak

    sim = breakdown.get("semantic_similarity", 0.0)
    pop = breakdown.get("popularity", 0.0)
    rec = breakdown.get("recency", 0.0)
    hw = breakdown.get("hardware_fit", 0.0)
    lic = breakdown.get("license_match", 0.0)
    inf = breakdown.get("inference_profile", 0.0)

    phrases = {
        "match_to_request": pick(
            sim, 0.72, 0.45,
            "Strong match between your wording and this model's indexed chunks.",
            "Moderate textual match to your request.",
            "Looser match to your request; still worth comparing if constraints are tight.",
        ),
        "community_usage": pick(
            pop, 0.65, 0.35,
            "High community usage (downloads/likes) on the hub.",
            "Moderate hub traction.",
            "Smaller or newer footprint on the hub.",
        ),
        "freshness": pick(
            rec, 0.65, 0.4,
            "Recently touched on the hub.",
            "Average age on the hub.",
            "Older snapshot by hub dates — less weight in ranking now.",
        ),
        "hardware_fit": (
            "Tag/text signals align with a CPU‑friendly or small‑footprint deployment."
            if (constraints.get("hardware") or "").lower() in ("cpu_only", "cpu", "cpu-only", "edge")
            and hw >= 0.85
            else (
                "Looks compatible with your stated hardware preferences."
                if hw >= 0.75
                else "Hardware fit is mixed; double‑check runtime requirements on the card."
            )
        ),
        "license": (
            "License fits your stated permissive / commercial‑friendly requirement."
            if license_compliant and lic >= 0.9
            else (
                "License is permissive but not an exact string match to your list."
                if lic >= 0.75
                else "License does not meet the stated requirement — normally excluded when alternatives exist."
            )
        ),
        "inference_fit": (
            "Good fit for quantized / CPU inference heuristics (tags + id)."
            if inf >= 0.72
            else (
                "Reasonable fit for lightweight CPU inference heuristics."
                if inf >= 0.5
                else "Weaker fit on CPU / quantization heuristics; verify artifacts and benchmarks yourself."
            )
        ),
    }
    return phrases
