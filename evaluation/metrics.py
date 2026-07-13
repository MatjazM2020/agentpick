"""Ranking metrics and answer parsing.

Standard information-retrieval metrics (precision@k, recall@k, MRR, nDCG@k)
over ranked model-id lists, plus the heuristics that turn a free-text agent
answer into a ranked prediction: model-id extraction, abstention detection,
and clarifying-question detection.

All matching of model ids is case-insensitive; nDCG uses graded relevance
derived from the gold ranking (best gold model gets the highest grade).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

# "org/model" mentions: both sides must start alphanumeric.
_MODEL_ID_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*\b"
)
_URL_RE = re.compile(r"https?://\S+")

# Slash-separated prose that looks like an id must be filtered out. Junk falls
# into a few shapes: size/precision tokens ("4-bit/8-bit", "bigger/70B",
# "FP32/FP16"), two-word compounds ("chat/instruction-tuned", "Llama/Qwen-style"),
# filenames ("GGUF/llama.cpp"), and plain words ("Spanish/French", "GPU/CPU").
_REPO_LETTER_RE = re.compile(r"[A-Za-z]")
_QUANTITY_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?-?(?:b|m|k|t|bit|bits|gb|mb)|(?:fp|bf|int)\d+|q\d+(?:_[a-z0-9]+)?)$",
    re.IGNORECASE,
)
# Bare version tags ("exllama/v2") are runtimes, not model repos.
_VERSION_ONLY_RE = re.compile(r"^v\d+(?:\.\d+)*$", re.IGNORECASE)
# Exactly two plain words joined by one hyphen ("instruction-tuned",
# "Qwen-style"); real hyphenated repos have digits, more segments
# ("opus-mt-mul-en"), or id-style casing ("BioGPT-Large").
_TWO_PLAIN_WORDS_RE = re.compile(r"^[A-Za-z][a-z]*-[A-Za-z][a-z]*$")
_FILENAME_RE = re.compile(r"^[a-z]+\.[a-z]{1,4}$")
_DIGIT_RE = re.compile(r"\d")
_SEPARATOR_RE = re.compile(r"[-_.]")
# Uppercase after lowercase ("BioMedLM"): id-style casing that prose words
# ("French", "CPU", "Pearl", "PRs") never have.
_INNER_CAMEL_RE = re.compile(r"[a-z][A-Z]")


# Some catalog models are reachable under more than one id after a Hugging Face
# repo rename; canonicalize known aliases so an equivalent id isn't scored as a
# miss. The "Meta-Llama-3(.1)-*" repos were renamed to "Llama-3(.1)-*".
_META_LLAMA_ALIAS_RE = re.compile(r"^meta-llama/meta-llama-")


def _norm(model_id: str) -> str:
    key = model_id.strip().casefold()
    return _META_LLAMA_ALIAS_RE.sub("meta-llama/llama-", key)


def _plausible_repo(repo: str) -> bool:
    """Whether the repo part of an "org/repo" candidate looks like a real
    model repo name rather than a prose word pair."""
    if not _REPO_LETTER_RE.search(repo):
        return False  # fractions like "3/4"
    if _VERSION_ONLY_RE.match(repo):
        return False
    if _DIGIT_RE.search(repo):
        return True  # version or size in the name ("Qwen2.5-3B-Instruct")
    if _SEPARATOR_RE.search(repo):
        return not (_TWO_PLAIN_WORDS_RE.match(repo) or _FILENAME_RE.match(repo))
    return bool(_INNER_CAMEL_RE.search(repo))


def extract_model_ids(text: str) -> list[str]:
    """Ranked model ids mentioned in an answer, in order of first appearance."""
    return extract_predictions(text, ())


def extract_predictions(text: str, gold: Sequence[str]) -> list[str]:
    """Ranked predictions for scoring against ``gold``.

    Every "org/model" id mentioned in the answer, plus gold models mentioned
    by bare repo name (answers often drop the org prefix and write just
    "Qwen2.5-Coder-7B-Instruct"), merged in order of first mention. A bare
    mention inside a longer id — or right after a different org's "/" — is
    not credited.
    """
    cleaned = _URL_RE.sub(" ", text or "")
    # normalized id -> (first-mention offset, display form)
    entries: dict[str, tuple[int, str]] = {}
    for match in _MODEL_ID_RE.finditer(cleaned):
        candidate = match.group(0).rstrip(".-")
        org, repo = candidate.split("/", 1)
        if _QUANTITY_RE.match(org) or _QUANTITY_RE.match(repo):
            continue
        if not _plausible_repo(repo):
            continue
        key = _norm(candidate)
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


_ABSTENTION_PHRASES = (
    "no model",
    "no catalog model",
    "no such model",
    "no available model",
    "not possible",
    "cannot be satisfied",
    "can't be satisfied",
    "cannot satisfy",
    "can't satisfy",
    "cannot simultaneously",
    "cannot be both",
    "can't be both",
    "cannot both",
    "can't both",
    "internally inconsistent",
    "mutually exclusive",
    "incompatible",
    "contradict",
    "conflict",
    "impossible",
    "does not exist",
    "doesn't exist",
    "not aware of any",
    "not found",
    "isn't found",
    "none match",
    # catalog-grounded absence (e.g. a nonexistent model id in the request)
    "not in the catalog",
    "not in our catalog",
    "not present in",
    "not listed in",
    "didn't show up",
    "did not show up",
    # empty catalog scans ("a scan ... returns nothing", "none meet the threshold")
    "returns nothing",
    "found nothing",
    "no results",
    "no matches",
    "nothing matches",
    "none meet",
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "unable to find",
    # soft grounded abstention ("I can't verify that model in the catalog")
    "cannot verify",
    "can't verify",
    "unable to verify",
    "cannot confirm",
    "can't confirm",
)


# Negative-existence statements with a qualifier between "no" and "model(s)"
# ("no instruction-tuned models in the catalog meet that threshold",
# "no models with >= 2T (2000B) parameters available").
_NO_MODEL_RE = re.compile(
    r"\bno\b[\w\s,'-]{0,60}?\bmodels?\b[\w\s,'()<>=≥≤~.%-]{0,40}?"
    r"\b(?:meets?|match\w*|satisf\w*|exists?|fits?|qualif\w*|available)\b"
)


def detects_impossible(text: str) -> bool:
    """Whether the answer states that no model satisfies the constraints
    or that the requested model is not in the catalog."""
    # Models emit curly apostrophes ("can’t"); normalize before matching.
    lowered = (
        (text or "")
        .replace("\u2019", "'")
        .replace("*", "")
        .replace("`", "")
        .casefold()
    )
    return any(p in lowered for p in _ABSTENTION_PHRASES) or bool(
        _NO_MODEL_RE.search(lowered)
    )


# Imperative clarification requests without a question mark
# ("If you tell me your target ..., I can narrow it down",
# "say what hardware you have and I'll narrow it down").
_CLARIFICATION_REQUEST_RE = re.compile(
    r"\b(?:if you (?:can )?tell me|(?:please )?tell me(?:\s+\w+){0,3}\s+(?:your|what|which|more|me|about)|let me know|reply with|say (?:what|which|whether|how much|how many))\b",
    re.IGNORECASE,
)


def asks_clarification(text: str) -> bool:
    """Whether the answer asks the user a (clarifying) question, directly
    ("What hardware do you have?") or as an imperative request."""
    text = text or ""
    return "?" in text or bool(_CLARIFICATION_REQUEST_RE.search(text))
