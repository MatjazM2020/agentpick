"""Paired statistical comparison of two evaluation result files.

Pairs the per-question scores of two systems by question id and, for every
metric they share, reports the means, the mean paired difference with a
percentile-bootstrap 95% CI, and a two-sided p-value from a paired
randomization (sign-flip permutation) test — the standard significance test
for IR evaluations, valid for small n and non-normal score distributions.

Usage (from the repository root):

    python -m evaluation.compare results/A.json results/B.json
    python -m evaluation.compare results/A.json results/B.json --metrics ndcg@3 mrr
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

DEFAULT_RESAMPLES = 10_000


def paired_scores(report_a: dict, report_b: dict) -> dict[str, list[tuple[float, float]]]:
    """metric -> [(score_a, score_b), ...] over question ids present in both
    reports; a metric is included per question only when both systems have it."""
    rows_a = {r["id"]: r for r in report_a["results"]}
    rows_b = {r["id"]: r for r in report_b["results"]}
    pairs: dict[str, list[tuple[float, float]]] = {}
    for qid in rows_a.keys() & rows_b.keys():
        scores_a = {**rows_a[qid]["scores"], "latency_s": rows_a[qid]["latency_s"]}
        scores_b = {**rows_b[qid]["scores"], "latency_s": rows_b[qid]["latency_s"]}
        for metric, value_a in scores_a.items():
            value_b = scores_b.get(metric)
            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                pairs.setdefault(metric, []).append((float(value_a), float(value_b)))
    return pairs


def permutation_p_value(diffs: list[float], n_resamples: int, rng: random.Random) -> float:
    """Two-sided p-value of the paired sign-flip permutation test for
    H0: mean difference = 0."""
    observed = abs(statistics.mean(diffs))
    hits = sum(
        1
        for _ in range(n_resamples)
        if abs(statistics.mean(d * rng.choice((-1.0, 1.0)) for d in diffs)) >= observed - 1e-12
    )
    # +1 correction so the p-value is never exactly 0 (Monte Carlo estimate).
    return (hits + 1) / (n_resamples + 1)


def bootstrap_diff_ci(diffs: list[float], n_resamples: int, rng: random.Random) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI of the mean paired difference."""
    means = sorted(
        statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_resamples)
    )
    return (
        means[round(0.025 * (n_resamples - 1))],
        means[round(0.975 * (n_resamples - 1))],
    )


def compare(
    report_a: dict,
    report_b: dict,
    metrics_filter: list[str] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[dict]:
    """Per-metric comparison rows, sorted by metric name."""
    rows = []
    for metric, pairs in sorted(paired_scores(report_a, report_b).items()):
        if metrics_filter and metric not in metrics_filter:
            continue
        diffs = [a - b for a, b in pairs]
        rng = random.Random(seed)
        lo, hi = bootstrap_diff_ci(diffs, n_resamples, rng)
        rows.append(
            {
                "metric": metric,
                "n": len(pairs),
                "mean_a": round(statistics.mean(a for a, _ in pairs), 4),
                "mean_b": round(statistics.mean(b for _, b in pairs), 4),
                "diff": round(statistics.mean(diffs), 4),
                "diff_ci95": [round(lo, 4), round(hi, 4)],
                "p_value": round(permutation_p_value(diffs, n_resamples, rng), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two evaluation result files."
    )
    parser.add_argument("file_a", type=Path, help="Results JSON of system A.")
    parser.add_argument("file_b", type=Path, help="Results JSON of system B.")
    parser.add_argument("--metrics", nargs="+", help="Only these metrics.")
    parser.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"Permutation/bootstrap resamples (default: {DEFAULT_RESAMPLES}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = parser.parse_args()

    report_a = json.loads(args.file_a.read_text(encoding="utf-8"))
    report_b = json.loads(args.file_b.read_text(encoding="utf-8"))
    name_a, name_b = report_a["system"], report_b["system"]
    rows = compare(report_a, report_b, args.metrics, args.resamples, args.seed)

    print(f"\n=== {name_a} (A) vs {name_b} (B), paired by question id ===")
    header = f"{'metric':>24} {'n':>3} {'A':>7} {'B':>7} {'A-B':>7} {'95% CI':>18} {'p':>7}"
    print(header)
    for row in rows:
        lo, hi = row["diff_ci95"]
        print(
            f"{row['metric']:>24} {row['n']:>3} {row['mean_a']:>7} {row['mean_b']:>7} "
            f"{row['diff']:>7} {f'[{lo}, {hi}]':>18} {row['p_value']:>7}"
        )


if __name__ == "__main__":
    main()
