"""Pooled multi-run comparison of the evaluation systems.

Single runs are noisy — the same question can flip 0↔1 between runs with
identical code — so headline claims rest on several repetitions. This module
groups a batch of result files by system (``<stamp>_r<i>_<system>.json``, as
written by ``evaluation.run --runs N``), averages every question's scores
across the runs, and compares one reference system against the others with an
*exact* paired sign-flip permutation test over all 2^n sign assignments.

Usage (from the repository root):

    python -m evaluation.pooled                       # latest batch
    python -m evaluation.pooled 20260812_210000       # a specific batch
    python -m evaluation.pooled --metrics all --per-question
    python -m evaluation.pooled --reference single_round

For every comparison it reports the mean paired difference with a
percentile-bootstrap 95% CI, per-question win/tie/loss counts, the p-value
range when any single run is left out (with >= 3 runs), and Holm-corrected
p-values: the reference system is tested against every baseline, so the raw
p-values of one metric form a family and need the correction.

It also derives a per-question ``composite`` task-success score so the scored
categories enter a single paired test:

    deterministic / ranking / multi_turn   nDCG@k
    ambiguous                              mentions_expected
    impossible / off_topic                 no automatic score

The last line carries no score on purpose. The only rule expressible without
phrase heuristics — "correct iff the answer names no model" — measures silence
rather than correctness: a retrieval baseline that happens to find nothing scores
1.0, while an answer that correctly denies the request and offers a verified
alternative scores 0.0. It also contradicts the dataset it scores, whose N5
justification accepts naming a real alternative alongside the denial. Those
answers are stored in the results file for qualitative grading instead, as
``evaluation.run`` documents.

Finally, each system's out-of-catalog recommendation rate (the share of
recommended ids absent from the catalog snapshot) is reported as grounding
evidence. That rate ignores model ids the user named in the question itself:
a correct answer to "should I use <nonexistent model>?" quotes that id back to
deny it, and echoing the user's own id is not a recommendation.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation import metrics
from evaluation.compare import DEFAULT_RESAMPLES, bootstrap_diff_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SYSTEMS = ("agent", "llm_only", "single_round", "qdrant_only")
RANKED_CATEGORIES = ("deterministic", "ranking", "multi_turn")

_FILE_RE = re.compile(
    rf"^(?P<stamp>\d{{8}}_\d{{6}})(?:_r(?P<run>\d+))?_(?P<system>{'|'.join(SYSTEMS)})"
    r"(?P<rescored>_rescored)?\.json$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_batches() -> dict[str, dict[str, dict[str, Path]]]:
    """stamp -> system -> run index -> path, preferring _rescored files."""
    batches: dict[str, dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(RESULTS_DIR.glob("*.json")):
        match = _FILE_RE.match(path.name)
        if not match:
            continue
        runs = batches[match["stamp"]][match["system"]]
        run = match["run"] or "1"
        if run not in runs or match["rescored"]:
            runs[run] = path
    return batches


def _system_picks(result: dict) -> list[str]:
    """The model ids the system itself put forward.

    Ids the user named in the question are dropped: N5 asks about a
    nonexistent model, and every correct answer quotes that id back in order
    to deny it — echoing the user's own id is not a recommendation.
    """
    asked = (result["question"] or "").casefold()
    return [
        model_id
        for model_id in result["scores"].get("predicted_models", [])
        if model_id.casefold() not in asked
    ]


def _composite(result: dict, k: int) -> float | None:
    """Task success of one question, or None when its category carries no
    automatic score (impossible / off_topic — see the module docstring)."""
    if result["category"] in RANKED_CATEGORIES:
        return float(result["scores"].get(f"ndcg@{k}", 0.0))
    if result["category"] == "ambiguous":
        return float(result["scores"].get("mentions_expected", 0.0))
    return None


def load_runs(paths: list[Path], k: int) -> tuple[list[dict[str, dict[str, float]]], list[str]]:
    """Per-run ``{question id: {metric: score}}`` for one system, plus every
    model id it recommended across the batch (for the out-of-catalog rate)."""
    runs: list[dict[str, dict[str, float]]] = []
    predictions: list[str] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        run: dict[str, dict[str, float]] = {}
        for result in report["results"]:
            scores = {
                metric: float(value)
                for metric, value in result["scores"].items()
                if isinstance(value, (int, float))
            }
            scores["latency_s"] = float(result["latency_s"])
            composite = _composite(result, k)
            if composite is not None:
                scores["composite"] = composite
            run[result["id"]] = scores
            predictions.extend(_system_picks(result))
        runs.append(run)
    return runs, predictions


def pooled_means(
    runs: list[dict[str, dict[str, float]]], qids: list[str], metric: str
) -> dict[str, float]:
    """question id -> mean of one metric across runs, for the questions that
    carry it (ranking metrics exist only for the scored categories)."""
    pooled = {}
    for qid in qids:
        values = [run[qid][metric] for run in runs if metric in run.get(qid, {})]
        if values:
            pooled[qid] = statistics.mean(values)
    return pooled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _signed_sums(values: list[float]) -> list[float]:
    """Every sum obtainable by assigning + or - to each value (2^len values)."""
    sums = [0.0]
    for value in values:
        sums = [s + value for s in sums] + [s - value for s in sums]
    return sums


def exact_sign_flip_p(diffs: list[float]) -> float:
    """Two-sided p-value of the paired sign-flip test, enumerating all 2^n sign
    assignments exactly.

    The statistic is the sum of the signed differences. Enumerating both halves
    separately and counting matches by binary search costs 2^(n/2) log 2^(n/2)
    instead of 2^n, so n = 20 questions is instant. The observed assignment is
    always counted, so the p-value can never be 0.
    """
    total = abs(math.fsum(diffs))
    if total <= 1e-12:
        return 1.0  # no difference to test: every assignment is at least as extreme
    observed = total - 1e-12
    half = len(diffs) // 2
    left = _signed_sums(diffs[:half])
    right = sorted(_signed_sums(diffs[half:]))
    hits = 0
    for value in left:
        # |value + other| >= observed  <=>  other >= observed - value
        #                              or   other <= -observed - value
        hits += len(right) - bisect.bisect_left(right, observed - value)
        hits += bisect.bisect_right(right, -observed - value)
    return hits / 2 ** len(diffs)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare_pooled(
    runs_a: list[dict[str, dict[str, float]]],
    runs_b: list[dict[str, dict[str, float]]],
    qids: list[str],
    metric: str,
    seed: int = 0,
) -> dict | None:
    """One pooled comparison row, or None when fewer than two questions carry
    the metric in both systems."""
    means_a = pooled_means(runs_a, qids, metric)
    means_b = pooled_means(runs_b, qids, metric)
    shared = sorted(means_a.keys() & means_b.keys())
    if len(shared) < 2:
        return None
    diffs = [means_a[qid] - means_b[qid] for qid in shared]
    lo, hi = bootstrap_diff_ci(diffs, DEFAULT_RESAMPLES, random.Random(seed))
    row = {
        "metric": metric,
        "n": len(shared),
        "mean_a": statistics.mean(means_a[qid] for qid in shared),
        "mean_b": statistics.mean(means_b[qid] for qid in shared),
        "diff": statistics.mean(diffs),
        "diff_ci95": (lo, hi),
        "p_value": exact_sign_flip_p(diffs),
        "wins": sum(1 for d in diffs if d > 1e-9),
        "ties": sum(1 for d in diffs if abs(d) <= 1e-9),
        "losses": sum(1 for d in diffs if d < -1e-9),
        "loo_p": None,
    }
    if len(runs_a) >= 3 and len(runs_a) == len(runs_b):
        # How much of the result rests on any single run.
        loo = [
            exact_sign_flip_p(
                [
                    pooled_means(rest_a, qids, metric)[qid]
                    - pooled_means(rest_b, qids, metric)[qid]
                    for qid in shared
                ]
            )
            for rest_a, rest_b in (
                (runs_a[:i] + runs_a[i + 1:], runs_b[:i] + runs_b[i + 1:])
                for i in range(len(runs_a))
            )
        ]
        row["loo_p"] = (min(loo), max(loo))
    return row


def out_of_catalog_rate(predictions: list[str]) -> tuple[float, int, int]:
    """(rate, misses, total) of recommended ids absent from the catalog
    snapshot — a system's hallucination/unavailability rate."""
    catalog = metrics._catalog_keys()  # package-internal: the committed snapshot
    misses = sum(1 for p in predictions if metrics._norm(p) not in catalog)
    total = len(predictions)
    return (misses / total if total else 0.0), misses, total


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_systems(
    systems: dict[str, list[dict[str, dict[str, float]]]],
    predictions: dict[str, list[str]],
    qids: list[str],
    metric_names: list[str],
) -> None:
    print(f"\n{'system':>14}" + "".join(f"{m:>13}" for m in metric_names) + f"{'out-of-cat':>17}")
    for system, runs in systems.items():
        cells = ""
        for metric in metric_names:
            pooled = pooled_means(runs, qids, metric)
            cells += f"{statistics.mean(pooled.values()):>13.3f}" if pooled else f"{'—':>13}"
        rate, misses, total = out_of_catalog_rate(predictions[system])
        print(f"{system:>14}{cells}{f'{rate:.1%} ({misses}/{total})':>17}")


def print_comparisons(rows: dict[str, list[dict]], reference: str) -> None:
    for metric, metric_rows in rows.items():
        adjusted = holm([row["p_value"] for row in metric_rows])
        print(f"\n--- {metric} (n={metric_rows[0]['n']}) ---")
        print(
            f"{'baseline':>14}{reference:>14}{'baseline':>10}{'diff':>8}{'95% CI':>18}"
            f"{'p':>8}{'holm':>8}{'W/T/L':>10}{'LOO p':>16}"
        )
        for row, holm_p in zip(metric_rows, adjusted):
            lo, hi = row["diff_ci95"]
            loo = (
                f"{row['loo_p'][0]:.3f}–{row['loo_p'][1]:.3f}" if row["loo_p"] else "—"
            )
            print(
                f"{row['baseline']:>14}{row['mean_a']:>14.3f}{row['mean_b']:>10.3f}"
                f"{row['diff']:>+8.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
                f"{row['p_value']:>8.4f}{holm_p:>8.4f}"
                f"{f'{row['wins']}/{row['ties']}/{row['losses']}':>10}{loo:>16}"
            )


def print_per_question(
    systems: dict[str, list[dict[str, dict[str, float]]]],
    categories: dict[str, str],
    qids: list[str],
) -> None:
    print("\n--- per-question pooled composite ---")
    names = list(systems)
    print(f"{'id':>6}{'category':>15}" + "".join(f"{n:>14}" for n in names))
    pooled = {s: pooled_means(runs, qids, "composite") for s, runs in systems.items()}
    for qid in qids:
        # impossible / off_topic carry no composite: graded qualitatively.
        cells = "".join(
            f"{pooled[n][qid]:>14.2f}" if qid in pooled[n] else f"{'—':>14}"
            for n in names
        )
        print(f"{qid:>6}{categories[qid]:>15}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pooled multi-run comparison of the evaluation systems."
    )
    parser.add_argument("stamp", nargs="?", help="Batch timestamp (default: the latest).")
    parser.add_argument(
        "--reference", default="agent", choices=sorted(SYSTEMS),
        help="System tested against all others (default: agent).",
    )
    parser.add_argument(
        "--metrics", nargs="+",
        help="Metrics to test (default: the ranking metrics and the composite; "
        "'all' adds latency).",
    )
    parser.add_argument("--per-question", action="store_true", help="Per-question composite table.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default: 0).")
    args = parser.parse_args()

    batches = find_batches()
    if not batches:
        raise SystemExit(f"No result files found in {RESULTS_DIR}")
    stamp = args.stamp or max(batches)
    if stamp not in batches:
        raise SystemExit(f"Unknown batch '{stamp}'; available: {sorted(batches)}")
    batch = batches[stamp]
    if args.reference not in batch:
        raise SystemExit(f"Batch '{stamp}' has no '{args.reference}' results.")

    first = json.loads(next(iter(batch[args.reference].values())).read_text(encoding="utf-8"))
    k = first["k"]
    systems: dict[str, list] = {}
    predictions: dict[str, list[str]] = {}
    for system in (args.reference, *(s for s in SYSTEMS if s != args.reference and s in batch)):
        paths = [batch[system][run] for run in sorted(batch[system])]
        systems[system], predictions[system] = load_runs(paths, k)

    # Only questions every system answered in every run can be paired.
    qids = sorted(
        set.intersection(*(set(run) for runs in systems.values() for run in runs))
    )
    categories = {
        result["id"]: result["category"]
        for result in json.loads(
            next(iter(batch[args.reference].values())).read_text(encoding="utf-8")
        )["results"]
    }
    metric_names = args.metrics or [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "composite"]
    if metric_names == ["all"]:
        metric_names = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "composite", "latency_s"]

    run_counts = {s: len(runs) for s, runs in systems.items()}
    print(
        f"\n=== pooled batch {stamp} — {first['chat_model']}, "
        f"{'/'.join(str(c) for c in dict.fromkeys(run_counts.values()))} runs × "
        f"{len(systems)} systems, {len(qids)} questions ==="
    )
    if len(set(run_counts.values())) > 1:
        print(f"  WARNING: unequal run counts per system: {run_counts}")

    print_systems(systems, predictions, qids, metric_names)

    rows: dict[str, list[dict]] = {}
    for metric in metric_names:
        for baseline in (s for s in systems if s != args.reference):
            row = compare_pooled(
                systems[args.reference], systems[baseline], qids, metric, args.seed
            )
            if row:
                rows.setdefault(metric, []).append({**row, "baseline": baseline})
    print_comparisons(rows, args.reference)

    if args.per_question:
        print_per_question(systems, categories, qids)
    print()


if __name__ == "__main__":
    main()
