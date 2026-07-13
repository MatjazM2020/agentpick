"""Evaluation runner CLI.

Runs one or more systems over the gold-standard dataset, scores every answer,
and writes per-question results plus per-category aggregates to a JSON file.

Usage (from the repository root, with Qdrant/Postgres up and OPENAI_API_KEY set):

    python -m evaluation.run                          # all systems, all questions
    python -m evaluation.run --systems agent          # agent only
    python -m evaluation.run --ids D1 Q5 Q20          # subset
    python -m evaluation.run --categories ranking     # one category

Scoring per category:

- deterministic / ranking — precision@k, recall@k, MRR, nDCG@k against the
  gold list (k=3 by default).
- ambiguous  — does the answer ask a clarifying question, and does it mention
  at least one of the acceptable models.
- impossible — does the answer abstain (state that no model fits / is not in
  the catalog).
- multi_turn — does the first answer ask a clarifying question, plus the
  ranking and explanation metrics on the final answer.
- off_topic  — does the answer redirect without recommending any model.

Every raw answer is stored in the results file so answers can additionally be
graded by a human. Latency (seconds per answer) is recorded for all questions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluation import metrics
from evaluation.dataset import CATEGORIES, EvalQuestion, load_dataset
from src.core.agent_activity_log import dialogue_turn

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"
SYSTEM_NAMES = ("agent", "llm_only", "single_round", "qdrant_only")  # keys of evaluation.systems.SYSTEMS


def score_answer(question: EvalQuestion, answers: list[str], k: int) -> dict:
    """Category-appropriate metrics for one answer (or dialogue of answers)."""
    final = answers[-1] if answers else ""
    predicted = metrics.extract_predictions(final, question.expected_models)
    scores: dict = {"predicted_models": predicted}
    gold = list(question.expected_models)

    if question.category in ("deterministic", "ranking", "multi_turn"):
        scores[f"precision@{k}"] = metrics.precision_at_k(predicted, gold, k)
        scores[f"recall@{k}"] = metrics.recall_at_k(predicted, gold, k)
        scores["mrr"] = metrics.mrr(predicted, gold)
        scores[f"ndcg@{k}"] = metrics.ndcg_at_k(predicted, gold, k)
        if question.category == "multi_turn":
            scores["asks_clarification_turn1"] = float(
                metrics.asks_clarification(answers[0] if answers else "")
            )
    elif question.category == "ambiguous":
        scores["asks_clarification"] = float(metrics.asks_clarification(final))
        scores["mentions_expected"] = float(
            metrics.recall_at_k(predicted, gold, len(predicted) or 1) > 0
        )
    elif question.category == "impossible":
        scores["abstains"] = float(
            metrics.detects_impossible(final) or not predicted
        )
    elif question.category == "off_topic":
        scores["redirects"] = float(not predicted)
    return scores


def bootstrap_ci(
    values: list[float], n_boot: int = 10_000, seed: int = 0
) -> list[float]:
    """Percentile-bootstrap 95% confidence interval of the mean."""
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot)
    )
    return [
        round(means[round(0.025 * (n_boot - 1))], 4),
        round(means[round(0.975 * (n_boot - 1))], 4),
    ]


def aggregate(results: list[dict]) -> dict:
    """Mean of every numeric metric (with a bootstrap 95% CI when n >= 2),
    per category, plus overall latency."""
    summary: dict = {}
    for category in CATEGORIES:
        rows = [r for r in results if r["category"] == category]
        if not rows:
            continue
        numeric_keys = sorted(
            k
            for row in rows
            for k, v in row["scores"].items()
            if isinstance(v, (int, float))
        )
        means: dict = {}
        ci95: dict = {}
        for key in dict.fromkeys(numeric_keys):
            values = [r["scores"].get(key, 0.0) for r in rows]
            means[key] = round(statistics.mean(values), 4)
            if len(values) >= 2:
                ci95[key] = bootstrap_ci(values)
        summary[category] = {"n": len(rows), **means}
        if ci95:
            summary[category]["ci95"] = ci95
    if results:
        summary["latency_s"] = {
            "mean": round(statistics.mean(r["latency_s"] for r in results), 2),
            "max": round(max(r["latency_s"] for r in results), 2),
        }
    return summary


async def run_dialogue(
    run_fn, turns: list[str], question_id: str = ""
) -> list[str]:
    """Feed the user turns one by one, passing the growing dialogue each time."""
    messages: list[dict] = []
    answers: list[str] = []
    dialogue_id = uuid.uuid4().hex[:8]
    total = len(turns)
    for user_turn, turn in enumerate(turns, start=1):
        messages.append({"role": "user", "content": turn})
        with dialogue_turn(dialogue_id, user_turn, total, question_id):
            answer = await run_fn(messages)
        answers.append(answer)
        messages.append({"role": "assistant", "content": answer})
    return answers


async def evaluate_system(system: str, questions: list[EvalQuestion], k: int) -> dict:
    """Run one system over all questions sequentially and score the answers."""
    # Imported here so scoring-only commands (--rescore) don't need the
    # backend dependencies / data stores.
    from evaluation.systems import SYSTEMS
    from src.core import config  # importable via the bootstrap in evaluation/__init__.py

    run_fn = SYSTEMS[system]
    if system in ("agent", "single_round", "qdrant_only"):
        # Pre-load the query embedder (as the API server does at startup) so
        # the first semantic search isn't charged its ~20s load time.
        from src.core.llm import warmup

        await asyncio.to_thread(warmup)
    results = []
    for q in questions:
        t0 = time.monotonic()
        try:
            answers = await run_dialogue(run_fn, list(q.turns), q.id)
            error = None
        except Exception as e:  # score a failed call as an empty answer
            logger.error("[eval] %s / %s failed: %s", system, q.id, e)
            answers, error = [""], str(e)
        latency = round(time.monotonic() - t0, 2)
        result = {
            "id": q.id,
            "category": q.category,
            "question": q.question,
            "expected_models": list(q.expected_models),
            "answer": answers[-1],
            "latency_s": latency,
            "scores": score_answer(q, answers, k),
        }
        if len(q.turns) > 1:
            result["turn_answers"] = answers
        if error:
            result["error"] = error
        results.append(result)
        logger.info("[eval] %s / %s done in %.1fs", system, q.id, latency)
    return {
        "system": system,
        "chat_model": config.CHAT_MODEL,
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": aggregate(results),
        "results": results,
    }


def print_summary(report: dict) -> None:
    print(f"\n=== {report['system']} (model: {report['chat_model']}, k={report['k']}) ===")
    for category, values in report["summary"].items():
        ci95 = values.get("ci95", {})
        parts = []
        for k, v in values.items():
            if k == "ci95":
                continue
            if k in ci95:
                lo, hi = ci95[k]
                parts.append(f"{k}={v} [{lo}, {hi}]")
            else:
                parts.append(f"{k}={v}")
        print(f"  {category:>13}: {', '.join(parts)}")


def rescore_report(path: Path, k: int) -> dict:
    """Recompute scores and summary for an existing results file from its
    stored raw answers (no API calls). Gold models come from the stored
    results (self-consistent with the old run)."""
    report = json.loads(path.read_text(encoding="utf-8"))
    for result in report["results"]:
        answers = result.get("turn_answers") or [result.get("answer", "")]
        question = EvalQuestion(
            id=result["id"],
            category=result["category"],
            turns=(result["question"],),
            expected_models=tuple(result["expected_models"]),
        )
        result["scores"] = score_answer(question, answers, k)
    report["k"] = k
    report["rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["summary"] = aggregate(report["results"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AgentPick against the gold dataset.")
    parser.add_argument(
        "--systems", nargs="+", choices=sorted(SYSTEM_NAMES),
        default=list(SYSTEM_NAMES),
        help="Systems to evaluate (default: all).",
    )
    parser.add_argument(
        "--rescore", nargs="+", type=Path, metavar="RESULTS_JSON",
        help="Recompute scores/summary for existing result files from their "
        "stored answers (no API calls); writes <name>_rescored.json.",
    )
    parser.add_argument("--k", type=int, default=3, help="Cutoff for @k metrics (default: 3).")
    parser.add_argument("--ids", nargs="+", help="Only these question ids (e.g. D1 Q5).")
    parser.add_argument(
        "--categories", nargs="+", choices=CATEGORIES, help="Only these categories."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Directory for result JSON files (default: {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.rescore:
        for path in args.rescore:
            report = rescore_report(path, args.k)
            out_path = path.with_name(f"{path.stem}_rescored.json")
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print_summary(report)
            print(f"  results -> {out_path}")
        return

    questions = load_dataset(ids=args.ids, categories=args.categories)
    if not questions:
        raise SystemExit("No questions matched the given filters.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for system in args.systems:
        report = asyncio.run(evaluate_system(system, questions, args.k))
        out_path = args.out_dir / f"{stamp}_{system}.json"
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print_summary(report)
        print(f"  results -> {out_path}")


if __name__ == "__main__":
    main()
