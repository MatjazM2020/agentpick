"""Ranking metrics and answer parsing.

Standard information-retrieval metrics (precision@k, recall@k, MRR, nDCG@k)
over ranked model-id lists, plus the model-id extraction that turns a
free-text agent answer into a ranked prediction (validated against a snapshot
of the catalog's ids).

All matching of model ids is case-insensitive; nDCG uses graded relevance
derived from the gold ranking (best gold model gets the highest grade).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

# "org/model" mentions: both sides must start alphanumeric.
_MODEL_ID_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*\b"
)
_URL_RE = re.compile(r"https?://\S+")

# Some catalog models are reachable under more than one id after a Hugging Face
# repo rename; canonicalize known aliases so an equivalent id isn't scored as a
# miss. The "Meta-Llama-3(.1)-*" repos were renamed to "Llama-3(.1)-*".
_META_LLAMA_ALIAS_RE = re.compile(r"^meta-llama/meta-llama-")


def _norm(model_id: str) -> str:
    key = model_id.strip().casefold()
    return _META_LLAMA_ALIAS_RE.sub("meta-llama/llama-", key)


@lru_cache(maxsize=1)
def _catalog_keys() -> frozenset[str]:
    """Normalized ids of every catalog model, from the committed snapshot
    (catalog_ids.txt) so scoring works offline and stays reproducible."""
    text = (Path(__file__).parent / "catalog_ids.txt").read_text(encoding="utf-8")
    return frozenset(
        _norm(line) for line in text.splitlines() if line and not line.startswith("#")
    )


# Candidates NOT in the catalog are either real out-of-catalog models (which
# must stay in the prediction list so hallucinated or unavailable picks cost
# precision) or slash-separated prose that happens to look like "org/repo"
# ("chat/instruction-tuned", "FP32/FP16", "A100/A6000"). Two shape checks
# separate them; neither matches a single one of the 1,709 catalog ids.
#
# A side that is entirely one quantity, precision, quant level, version, size
# bound, GPU code, family-version fragment, or serialization format is prose.
_JUNK_TOKEN_RE = re.compile(
    r"""^(?:
        \d+(?:\.\d+)?(?:-?(?:b|m|k|t|bit|bits|gb|mb))?                      # 3, 70B, 4-bit
        |(?:fp|bf|int)\d+                                                   # FP16, int8
        |q\d+(?:_[a-z0-9]+)?                                                # Q4, Q5_0
        |v\d+(?:\.\d+)*                                                     # v2
        |(?:under|over|near|about|sub|upto|up-to)-?\d+(?:\.\d+)?[a-z]{0,3}  # under-35B
        |[a-z]\d{3,4}                                                       # A100, H100
        |[a-z]+\d+\.\d+                                                     # Qwen2.5
        |gguf|awq|gptq|mlx|exl2|bnb|onnx|safetensors                        # GGUF, MLX
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
# Digitless prose compounds ("open-instruct-style", "PubMed-like", "seq-to-seq").
_PROSE_SUFFIX_RE = re.compile(
    r"(?:-style|-like|-compatible|-efficient|-ready|-friendly|-only|-based)$|-to-",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
_SEPARATOR_RE = re.compile(r"[-_.]")
# Uppercase after lowercase ("BioMedLM"): id-style casing that prose words
# ("French", "CPU", "Pearl", "PRs") never have.
_INNER_CAMEL_RE = re.compile(r"[a-z][A-Z]")


def _plausible_id(org: str, repo: str) -> bool:
    """Whether an out-of-catalog "org/repo" candidate plausibly names a real
    model repo rather than slash-separated prose."""
    if _JUNK_TOKEN_RE.match(org) or _JUNK_TOKEN_RE.match(repo):
        return False
    if _DIGIT_RE.search(repo):
        return True  # version or size in the name ("Phi-3.5-mini-instruct")
    if _PROSE_SUFFIX_RE.search(repo):
        return False
    # Digitless: needs id-style casing or several separators; single prose
    # words ("transformers") and word pairs ("instruction-tuned") have neither.
    return bool(_INNER_CAMEL_RE.search(repo)) or len(_SEPARATOR_RE.findall(repo)) >= 2


def extract_model_ids(text: str) -> list[str]:
    """Ranked model ids mentioned in an answer, in order of first appearance."""
    return extract_predictions(text, ())


def extract_predictions(text: str, gold: Sequence[str]) -> list[str]:
    """Ranked predictions for scoring against ``gold``.

    Every "org/model" id mentioned in the answer — kept when it is a catalog
    model, or when an out-of-catalog id still plausibly names a real repo —
    plus gold models mentioned by bare repo name (answers often drop the org
    prefix and write just "Qwen2.5-Coder-7B-Instruct"), merged in order of
    first mention. A bare mention inside a longer id — or right after a
    different org's "/" — is not credited.
    """
    cleaned = _URL_RE.sub(" ", text or "")
    # normalized id -> (first-mention offset, display form)
    entries: dict[str, tuple[int, str]] = {}
    for match in _MODEL_ID_RE.finditer(cleaned):
        candidate = match.group(0).rstrip(".-")
        key = _norm(candidate)
        if key not in _catalog_keys() and not _plausible_id(*candidate.split("/", 1)):
            continue
        if key not in entries:
            entries[key] = (match.start(), candidate)
    folded = cleaned.casefold()
    for gold_id in gold:
        repo = gold_id.split("/", 1)[1] if "/" in gold_id else gold_id
        bare = re.compile(
            r"(?<![A-Za-z0-9_./-])" + re.escape(repo.casefold()) + r"(?![A-Za-z0-9_.-])"
        )
        found = bare.search(folded)
        key = _norm(gold_id)
        if found and (key not in entries or found.start() < entries[key][0]):
            entries[key] = (found.start(), gold_id)
    # A family umbrella written in id form ("Qwen/Qwen2.5-Coder" introducing
    # the Qwen2.5-Coder-* picks) is not itself a recommendation: drop an id
    # that two or more other mentioned ids extend, unless it is gold in its
    # own right. Requiring several members keeps a genuine base-model pick
    # ("Kimi-K2-Instruct") alive when just one variant of it is mentioned.
    gold_keys = {_norm(g) for g in gold}
    for key in [k for k in entries if k not in gold_keys]:
        members = sum(1 for other in entries if other != key and other.startswith(key + "-"))
        if members >= 2:
            del entries[key]
    return [candidate for _, candidate in sorted(entries.values())]


def precision_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the top-k predictions that are relevant (in the gold list)."""
    if k <= 0:
        return 0.0
    gold_set = {_norm(g) for g in gold}
    hits = sum(1 for p in predicted[:k] if _norm(p) in gold_set)
    return hits / k


def recall_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the gold models found in the top-k predictions."""
    if not gold:
        return 0.0
    gold_set = {_norm(g) for g in gold}
    hits = sum(1 for p in predicted[:k] if _norm(p) in gold_set)
    return hits / len(gold_set)


def mrr(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant prediction (0 if none)."""
    gold_set = {_norm(g) for g in gold}
    for rank, p in enumerate(predicted, start=1):
        if _norm(p) in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """nDCG@k with graded relevance from the gold order.

    The i-th gold model (0-based) has relevance ``len(gold) - i``, so a
    prediction that reproduces the gold order scores 1.0 and swapped
    positions are penalized.
    """
    if not gold or k <= 0:
        return 0.0
    grades = {_norm(g): len(gold) - i for i, g in enumerate(gold)}
    dcg = sum(
        grades.get(_norm(p), 0) / math.log2(rank + 1)
        for rank, p in enumerate(predicted[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0

