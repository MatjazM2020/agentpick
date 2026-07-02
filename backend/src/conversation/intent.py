"""Infer recommendation intent from user messages."""

from __future__ import annotations

import re
from typing import Optional

_SINGLE_PICK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwhich (one|model|of (those|them|these|the (models|options)))\b",
        r"\b(of those|of them|of these|from those|from the list)\b.*\b(best|recommend|pick|choose)\b",
        r"\b(best|top) (one|pick|choice)\b",
        r"\b(single|just one|one) (model|recommendation|pick|option)\b",
        r"\brecommend (me )?(just )?one\b",
        r"\bpick (one|the best|a single)\b",
        r"\bchoose (one|the best|a single)\b",
        r"\btop pick\b",
        r"\bnarrow (it )?down\b",
        r"\bwhich would you (recommend|choose|pick)\b",
        r"\bif you (had|have) to (pick|choose|recommend) (just )?one\b",
        r"\bwhat('s| is) your (top |#1 )?(pick|recommendation)\b",
        r"\b(most|single) recommended\b",
        r"\bonly (one|1) (model|recommendation)\b",
        r"\btop 1\b",
        r"\bwhich (single )?model (would you|should i|do you)\b",
    )
)

_MULTI_PICK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(show|give|list) (me )?(more )?(options|alternatives|models|choices)\b",
        r"\btop 3\b",
        r"\b(a few|several|multiple) (options|models|recommendations)\b",
    )
)

_FOLLOW_UP_SINGLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^which (is|one|model)\b",
        r"^what('s| is) (the )?best\b",
        r"^which\b.*\bbest\b",
        r"^pick\b",
        r"^recommend\b",
        r"^which (of|one)\b",
    )
)


def infer_recommendation_top_k(
    user_query: str,
    *,
    conversation_text: Optional[str] = None,
    is_follow_up: bool = False,
) -> int:
    """
    Return 1 when the user wants a single best pick; otherwise 3 (default).

    Detects explicit single-model requests and short follow-ups like
    "which of those is best?" after a prior recommendation turn.
    """
    query = (user_query or "").strip()
    if not query:
        return 3

    for pattern in _MULTI_PICK_PATTERNS:
        if pattern.search(query):
            return 3

    for pattern in _SINGLE_PICK_PATTERNS:
        if pattern.search(query):
            return 1

    has_prior_context = is_follow_up or bool(
        (conversation_text or "").strip()
        and len([p for p in (conversation_text or "").split("\n\n") if p.strip()]) > 1
    )
    if has_prior_context and len(query.split()) <= 20:
        for pattern in _FOLLOW_UP_SINGLE_PATTERNS:
            if pattern.search(query):
                return 1

    return 3
