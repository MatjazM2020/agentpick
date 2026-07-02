"""Detect and format Open WebUI background task requests."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def is_follow_up_generation_task(query: str) -> bool:
    """True when Open WebUI asks for clickable follow-up suggestions."""
    q = (query or "").strip().lower()
    if not q.startswith("### task:"):
        return False
    return "follow-up" in q or "follow_ups" in q


def follow_ups_response_content(questions: List[str]) -> str:
    """JSON body Open WebUI parses from the task completion content."""
    cleaned = [q.strip() for q in questions if q and q.strip()]
    return json.dumps({"follow_ups": cleaned[:5]}, ensure_ascii=False)


def fallback_follow_ups_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Generic follow-ups when none were stored from the last recommendation."""
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            last_assistant = content if isinstance(content, str) else str(content or "")
            break

    model_ids = re.findall(r"^\d+\.\s+([\w./-]+)\s*:", last_assistant, re.MULTILINE)
    if len(model_ids) >= 2:
        return [
            f"Which of {model_ids[0]} and {model_ids[1]} is better for my use case?",
            "Which option runs best on limited GPU memory?",
            "Can you recommend just one model from this list?",
        ]
    if len(model_ids) == 1:
        return [
            f"What are good alternatives to {model_ids[0]}?",
            "What hardware do I need to run this model?",
            "Are there smaller models that still work well for my task?",
        ]
    return [
        "Can you narrow this down to a single best pick?",
        "What hardware constraints should I consider?",
        "Are there popular open-source options for this task?",
    ]
