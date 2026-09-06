"""Aggregate every multi-run evaluation batch into one markdown statistics report.

Gathers all result batches in ``evaluation/results`` (one batch per chat model,
``<stamp>_r<i>_<system>.json`` as written by ``evaluation.run --runs N``),
pools each question's scores across the runs, and writes the numbers a thesis
chapter needs to a single ``.md`` file: per-system means, per-category means,
run-to-run spread, paired significance tests against the reference system, and
the per-question composite table.

Every statistic is computed by ``evaluation.pooled`` — this module only selects
and formats. Usage (from the repository root):

    python -m evaluation.thesis_stats
    python -m evaluation.thesis_stats --out thesis_v1/eval_stats.md
    python -m evaluation.thesis_stats --reference single_round
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from evaluation.pooled import (
    RANKED_CATEGORIES,
    SYSTEMS,
    compare_pooled,
    find_batches,
    holm,
    load_runs,
    out_of_catalog_rate,
    pooled_means,
)

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "statistics.md"
ABSTENTION_CATEGORIES = ("impossible", "off_topic")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _num(value: float | None, fmt: str = ".3f") -> str:
    return "—" if value is None else format(value, fmt)


def _abstention(paths: list[Path]) -> tuple[int, int]:
    """(items where the system named >= 1 model of its own, total items) over
    the impossible/off_topic questions. Ids the user named in the question are
    ignored — echoing them back in order to deny the request is not a pick."""
    named = total = 0
    for path in paths:
        for result in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if result["category"] not in ABSTENTION_CATEGORIES:
                continue
            asked = (result["question"] or "").casefold()
            picks = [
                m for m in result["scores"].get("predicted_models", [])
                if m.casefold() not in asked
            ]
            total += 1
            named += bool(picks)
    return named, total


def render_batch(stamp: str, batch: dict[str, dict[str, Path]], reference: str) -> list[str]:
    """The markdown section for one batch (= one chat model)."""
    paths = {s: [batch[s][r] for r in sorted(batch[s], key=int)] for s in SYSTEMS if s in batch}
    order = [reference, *(s for s in SYSTEMS if s != reference and s in paths)]
    first = json.loads(paths[reference][0].read_text(encoding="utf-8"))
    k = first["k"]
    categories = {r["id"]: r["category"] for r in first["results"]}

    systems, predictions = {}, {}
    for system in order:
        systems[system], predictions[system] = load_runs(paths[system], k)
    # Only questions every system answered in every run can be paired.
    qids = sorted(set.intersection(*(set(run) for runs in systems.values() for run in runs)))
    metrics = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "composite"]
    n_runs = {len(runs) for runs in systems.values()}

    lines = [
        f"## `{first['chat_model']}` — batch `{stamp}`",
        "",
        f"{'/'.join(str(n) for n in sorted(n_runs))} runs × {len(order)} systems × "
        f"{len(qids)} questions = {sum(len(p) for p in paths.values()) * len(qids)} scored answers. "
        f"k={k}. Reference system: `{reference}`.",
        "",
        "### System means",
        "",
        "Each question's score is averaged over the runs first, then averaged over the "
        f"questions that carry the metric (n={len(pooled_means(systems[reference], qids, f'ndcg@{k}'))} "
        f"for the ranking metrics, n={len(pooled_means(systems[reference], qids, 'composite'))} for the "
        f"composite, n={len(qids)} for latency).",
        "",
    ]

    rows = []
    for system in order:
        cells = []
        for metric in metrics:
            pooled = pooled_means(systems[system], qids, metric)
            cells.append(_num(statistics.mean(pooled.values()) if pooled else None))
        latency = pooled_means(systems[system], qids, "latency_s")
        rate, misses, total = out_of_catalog_rate(predictions[system])
        named, n_abs = _abstention(paths[system])
        rows.append([
            f"`{system}`", *cells,
            f"{statistics.mean(latency.values()):.2f}",
            f"{rate:.1%} ({misses}/{total})",
            f"{1 - named / n_abs:.0%} ({n_abs - named}/{n_abs})",
        ])
    lines += _table(
        ["system", *metrics, "latency (s)", "out-of-catalog", "silent on abstention"], rows
    )
    lines += [
        "",
        "*out-of-catalog* = share of recommended ids absent from the catalog snapshot "
        "(model ids the user named in the question are excluded). *silent on abstention* "
        "= share of the impossible/off_topic items on which the system named no model of "
        "its own; it is **descriptive only, not a score** — a baseline that retrieves "
        "nothing scores well on it, and a correct denial that offers a verified "
        "alternative scores badly, so those items are graded by hand.",
        "",
        "### Per-category means",
        "",
        f"Category primary metric: nDCG@{k} for deterministic/ranking/multi_turn, "
        "mentions_expected for ambiguous; impossible/off_topic carry no automatic score.",
        "",
    ]

    names = list(dict.fromkeys(categories.values()))
    composite = {s: pooled_means(systems[s], qids, "composite") for s in order}
    rows = []
    for category in names:
        members = [q for q in qids if categories[q] == category]
        cells = []
        for system in order:
            scored = [composite[system][q] for q in members if q in composite[system]]
            cells.append(_num(statistics.mean(scored) if scored else None, ".3f"))
        rows.append([category, str(len(members)), *cells])
    lines += _table(["category", "n", *(f"`{s}`" for s in order)], rows)

    lines += [
        "",
        "### Run-to-run spread",
        "",
        "Each run scored on its own, then summarized across the runs — the variability "
        "the pooling averages out. A spread wider than the differences being tested is "
        "why the headline numbers pool several repetitions.",
        "",
    ]
    rows = []
    for metric in (f"ndcg@{k}", "composite"):
        for system in order:
            per_run = [
                statistics.mean(run[q][metric] for q in qids if metric in run.get(q, {}))
                for run in systems[system]
            ]
            rows.append([
                metric, f"`{system}`",
                f"{statistics.mean(per_run):.3f}",
                f"{statistics.stdev(per_run):.3f}" if len(per_run) > 1 else "—",
                f"{min(per_run):.3f}–{max(per_run):.3f}",
            ])
    lines += _table(["metric", "system", "mean", "SD across runs", "range"], rows)

    lines += [
        "",
        f"### Paired comparisons — `{reference}` vs each baseline",
        "",
        "Unit of analysis is the question (scores pooled over runs first). *p* is the "
        "two-sided exact paired sign-flip permutation test over all 2^n sign "
        "assignments; *holm* is the Holm-Bonferroni adjustment within each metric's "
        "family of three baselines; *CI* is a percentile bootstrap (10,000 resamples, "
        "seed 0) of the mean paired difference; *LOO p* is the range of *p* when any "
        "single run is left out.",
        "",
    ]
    for metric in metrics:
        comparisons = []
        for baseline in (s for s in order if s != reference):
            row = compare_pooled(systems[reference], systems[baseline], qids, metric)
            if row:
                comparisons.append({**row, "baseline": baseline})
        if not comparisons:
            continue
        adjusted = holm([row["p_value"] for row in comparisons])
        lines += ["", f"**{metric}** (n={comparisons[0]['n']})", ""]
        rows = []
        for row, holm_p in zip(comparisons, adjusted):
            lo, hi = row["diff_ci95"]
            loo = f"{row['loo_p'][0]:.3f}–{row['loo_p'][1]:.3f}" if row["loo_p"] else "—"
            rows.append([
                f"`{row['baseline']}`",
                f"{row['mean_a']:.3f}", f"{row['mean_b']:.3f}", f"{row['diff']:+.3f}",
                f"[{lo:+.3f}, {hi:+.3f}]", f"{row['p_value']:.4f}", f"{holm_p:.4f}",
                f"{row['wins']}/{row['ties']}/{row['losses']}", loo,
            ])
        lines += _table(
            ["baseline", f"`{reference}`", "baseline", "diff", "95% CI", "p", "holm",
             "W/T/L", "LOO p"],
            rows,
        )

    lines += [
        "",
        "### Per-question pooled composite",
        "",
    ]
    # Grouped by category, then by the numeric part of the id (N2 before N11).
    ordered = sorted(
        qids, key=lambda q: (names.index(categories[q]), q.rstrip("0123456789"),
                             int(q.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or 0))
    )
    rows = [
        [q, categories[q], *(_num(composite[s].get(q), ".2f") for s in order)]
        for q in ordered
    ]
    lines += _table(["id", "category", *(f"`{s}`" for s in order)], rows)
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write pooled multi-run evaluation statistics to a markdown file."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output file (default: {DEFAULT_OUT}).")
    parser.add_argument(
        "--reference", default="agent", choices=sorted(SYSTEMS),
        help="System tested against all others (default: agent).",
    )
    args = parser.parse_args()

    batches = {
        stamp: batch
        for stamp, batch in sorted(find_batches().items())
        if all(system in batch for system in SYSTEMS)
    }
    if not batches:
        raise SystemExit("No batch with all four systems found in evaluation/results.")

    lines = [
        "# AgentPick evaluation — pooled statistics",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`python -m evaluation.thesis_stats` from the stored result files. "
        f"{len(batches)} batch(es): "
        + ", ".join(f"`{s}`" for s in batches)
        + ".",
        "",
    ]
    for stamp, batch in batches.items():
        lines += render_batch(stamp, batch, args.reference)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"statistics -> {args.out}")


if __name__ == "__main__":
    main()
