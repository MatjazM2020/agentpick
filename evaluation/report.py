"""Per-question markdown comparison of the four systems for one evaluation round.

Reads the four result files of a round (the latest round that has all four
systems, or the one named on the command line), takes each question's stored
scores, and writes a markdown table comparing the systems side by side.

Usage (from the repository root):

    python -m evaluation.report                    # latest complete round
    python -m evaluation.report 20260714_043202    # a specific round

The per-question score is the category's primary metric: nDCG@k for
deterministic/ranking/multi_turn and mentions_expected for ambiguous.
Impossible and off_topic questions carry no automatic score (a correct answer
names no model) and show as "—". Rescored files are preferred over the
originals when both exist.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SYSTEMS = ("agent", "llm_only", "single_round", "qdrant_only")

_FILE_RE = re.compile(
    rf"^(?P<stamp>.+?)_(?P<system>{'|'.join(SYSTEMS)})(?P<rescored>_rescored)?\.json$"
)


def find_rounds() -> dict[str, dict[str, Path]]:
    """stamp -> {system: path}, preferring _rescored files."""
    rounds: dict[str, dict[str, Path]] = {}
    for path in RESULTS_DIR.glob("*.json"):
        m = _FILE_RE.match(path.name)
        if not m:
            continue
        entry = rounds.setdefault(m["stamp"], {})
        if m["system"] not in entry or m["rescored"]:
            entry[m["system"]] = path
    return rounds


def primary_score(category: str, scores: dict, k: int) -> float | None:
    if category in ("deterministic", "ranking", "multi_turn"):
        return scores.get(f"ndcg@{k}")
    if category == "ambiguous":
        return scores.get("mentions_expected")
    return None


def build_report(stamp: str, files: dict[str, Path]) -> str:
    reports = {system: json.loads(files[system].read_text(encoding="utf-8")) for system in SYSTEMS}
    k = reports["agent"]["k"]
    # id -> {"category": ..., system: score}
    rows: dict[str, dict] = {}
    for system, report in reports.items():
        for result in report["results"]:
            row = rows.setdefault(result["id"], {"category": result["category"]})
            row[system] = primary_score(result["category"], result["scores"], k)

    def fmt(row: dict, system: str) -> str:
        value = row.get(system)
        if value is None:
            return "—"
        best = max(v for s in SYSTEMS if (v := row.get(s)) is not None)
        text = f"{value:.2f}"
        return f"**{text}**" if value == best and best > 0 else text

    lines = [
        f"# System comparison — round `{stamp}`",
        "",
        f"Model: `{reports['agent']['chat_model']}`, k={k}, "
        f"{len(rows)} questions. Score per question is the category's primary "
        f"metric (nDCG@{k} for deterministic/ranking/multi_turn, "
        "mentions_expected for ambiguous); impossible and off_topic questions "
        "carry no automatic score and show as \"—\". Best non-zero value per "
        "row in bold.",
        "",
        "| id | category | agent | llm_only | single_round | qdrant_only |",
        "|---|---|---|---|---|---|",
    ]
    for qid, row in rows.items():
        lines.append(
            f"| {qid} | {row['category']} | "
            + " | ".join(fmt(row, s) for s in SYSTEMS) + " |"
        )

    lines += ["", "## Means", "", "| | agent | llm_only | single_round | qdrant_only |", "|---|---|---|---|---|"]
    categories = list(dict.fromkeys(row["category"] for row in rows.values()))
    for category in categories:
        values = []
        for system in SYSTEMS:
            scores = [r[system] for r in rows.values() if r["category"] == category and r.get(system) is not None]
            values.append(f"{statistics.mean(scores):.2f}" if scores else "—")
        n = sum(1 for r in rows.values() if r["category"] == category)
        lines.append(f"| {category} (n={n}) | " + " | ".join(values) + " |")
    overall = [
        f"{statistics.mean([r[s] for r in rows.values() if r.get(s) is not None]):.2f}"
        for s in SYSTEMS
    ]
    scored = sum(1 for r in rows.values() if any(r.get(s) is not None for s in SYSTEMS))
    lines.append(
        f"| **all scored (n={scored})** | " + " | ".join(f"**{v}**" for v in overall) + " |"
    )
    latency = [
        f"{reports[s]['summary']['latency_s']['mean']:.1f}s" for s in SYSTEMS
    ]
    lines.append("| mean latency | " + " | ".join(latency) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rounds = find_rounds()
    complete = {stamp: files for stamp, files in rounds.items() if len(files) == len(SYSTEMS)}
    if not complete:
        raise SystemExit(f"No round with all four systems found in {RESULTS_DIR}")
    stamp = sys.argv[1] if len(sys.argv) > 1 else max(complete)
    if stamp not in complete:
        raise SystemExit(f"Round '{stamp}' is missing systems; complete rounds: {sorted(complete)}")
    out_path = RESULTS_DIR / f"{stamp}_comparison.md"
    out_path.write_text(build_report(stamp, complete[stamp]), encoding="utf-8")
    print(f"report -> {out_path}")


if __name__ == "__main__":
    main()
