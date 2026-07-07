"""Gold-standard evaluation dataset (loaded from dataset.json).

The dataset holds six question categories:

- ``deterministic`` — one correct model (multiple ids = alternative golds).
- ``ranking``       — an ordered gold list, best model first.
- ``ambiguous``     — underspecified query; a good answer suggests plausible
                      options and asks a clarifying question.
- ``impossible``    — unsatisfiable request; a good answer abstains.
- ``multi_turn``    — two user turns; turn 1 should draw a clarifying
                      question, the final answer is scored against the gold.
- ``off_topic``     — out-of-scope query; a good answer politely redirects
                      and recommends no models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATASET_PATH = Path(__file__).parent / "dataset.json"

CATEGORIES = (
    "deterministic",
    "ranking",
    "ambiguous",
    "impossible",
    "multi_turn",
    "off_topic",
)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    turns: tuple[str, ...]  # user turns; single-turn questions have one
    expected_models: tuple[str, ...]  # ranked, best first; empty when abstaining
    justification: str = ""

    @property
    def question(self) -> str:
        """Full question text (turns joined for multi-turn questions)."""
        return "\n".join(self.turns)


def load_dataset(
    path: Path = DATASET_PATH,
    ids: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list[EvalQuestion]:
    """Load the evaluation questions, optionally filtered by id or category."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for q in raw["questions"]:
        if q["category"] not in CATEGORIES:
            raise ValueError(f"Question {q['id']}: unknown category '{q['category']}'")
        turns = tuple(q["turns"]) if "turns" in q else (q["question"],)
        if not turns or not all(turns):
            raise ValueError(f"Question {q['id']}: empty turn text")
        questions.append(
            EvalQuestion(
                id=q["id"],
                category=q["category"],
                turns=turns,
                expected_models=tuple(q["expected_models"]),
                justification=q.get("justification", ""),
            )
        )
    if ids:
        wanted = {i.lower() for i in ids}
        questions = [q for q in questions if q.id.lower() in wanted]
    if categories:
        questions = [q for q in questions if q.category in categories]
    return questions
