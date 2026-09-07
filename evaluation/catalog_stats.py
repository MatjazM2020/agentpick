"""Descriptive statistics of the catalog snapshot, as reported in the thesis.

Regenerates every number in the two tables of the thesis section *Lastnosti
podatkov o modelih* — catalog size and text lengths, and per-attribute
coverage — plus the figures quoted in that section's prose (task-tag
mismatches, quantized re-uploads, model families, test artifacts).

Where ``thesis_stats.py`` reports how the systems performed, this module
describes the data they ran against. It reads the ``models`` table only;
nothing is written and no network call is made, so the report always describes
the snapshot currently loaded in PostgreSQL. Re-run it after re-capturing the
catalog and the thesis tables can be checked against it row by row.

Definitions that are not self-evident from the column names:

* **card length** — ``model_card`` with the YAML header stripped, measured in
  characters and in whitespace-separated words.
* **parameter count from safetensors** — ``parameter_count`` mixes exact counts
  read from safetensors metadata by ``enrich_models.py`` with nominal sizes
  parsed out of the repo id by ``backfill_param_counts.py``, and the column
  does not record which is which. A row counts as exact when its value differs
  from the nominal size its id would yield.
* **language** — a tag equal to one of the codes observed under the ``language``
  key of some card header in this snapshot. The portal writes languages as bare
  tags (``en``, ``eng``), which cannot otherwise be told apart from tags naming
  an architecture or a format.
* **family**, **quantized re-upload**, **test artifact** — taken from
  ``src.catalog``, so the report uses exactly the logic the query layer applies
  at run time.

Needs PyYAML to read the card headers; it is already in the backend
virtualenv, pulled in by ``transformers``.

Usage (from the repository root, with PostgreSQL reachable per ``.env``):

    docker compose up -d postgres
    backend/.venv/bin/python -m evaluation.catalog_stats
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

import yaml

from src import catalog

# YAML header at the top of a model card, as written by the card template. The
# closing "---" may end the file, so the trailing newline has to be optional:
# a handful of cards are a header and nothing else.
_YAML_HEADER_RE = re.compile(r"\A---\r?\n.*?\r?\n---[^\S\n]*(?:\r?\n|\Z)", re.S)
# A size token like "7b", "1.5B", "125m" — same rule as backfill_param_counts.py.
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([bBmM])(?![A-Za-z0-9])")
# Language codes are two or three letters, optionally with a region suffix.
_LANG_CODE_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]+)?")

COLUMNS = ("model_id", "pipeline_tag", "library_name", "tags", "parameter_count",
           "model_card", "num_chunks", "downloads", "likes")


def card_body(model_card: str | None) -> str:
    """Card text without the YAML header."""
    return _YAML_HEADER_RE.sub("", model_card or "").strip()


def card_header(model_card: str | None) -> dict:
    """Parsed YAML header, or an empty dict if it is absent or malformed."""
    match = _YAML_HEADER_RE.match(model_card or "")
    if not match:
        return {}
    try:
        header = yaml.safe_load(match.group(0).strip().strip("-").strip())
    except yaml.YAMLError:
        return {}
    return header if isinstance(header, dict) else {}


def nominal_params(model_id: str) -> int | None:
    """Parameter count implied by a size token in the repo id, or None."""
    match = _SIZE_RE.search(model_id.split("/", 1)[-1])
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).lower()
    return int(value * (1e9 if unit == "b" else 1e6))


def language_codes(rows: list[dict]) -> set[str]:
    """Every code declared under a ``language`` key anywhere in the snapshot."""
    codes: set[str] = set()
    for row in rows:
        declared = card_header(row["model_card"]).get("language")
        if isinstance(declared, str):
            codes.add(declared.lower())
        elif isinstance(declared, list):
            codes.update(str(code).lower() for code in declared if code)
    return {code for code in codes if _LANG_CODE_RE.fullmatch(code)}


def _thousands(value: float) -> str:
    """1709 -> '1.709', the thousands separator the thesis uses."""
    return f"{round(value):,}".replace(",", ".")


def _spread(values) -> str:
    """Median with the first and third quartile, as the tables print it."""
    data = list(values)
    low, _, high = statistics.quantiles(data, n=4, method="inclusive")
    return f"{_thousands(statistics.median(data))} ({_thousands(low)}–{_thousands(high)})"


def _share(matching: int, total: int) -> str:
    """Percentage with the decimal comma the thesis uses."""
    return f"{100 * matching / total:.1f} %".replace(".", ",")


def size_and_lengths(rows: list[dict]) -> list[tuple[str, str]]:
    """The rows of the catalog size / text length table."""
    total = len(rows)
    bodies = [card_body(row["model_card"]) for row in rows]
    short = sum(1 for body in bodies if len(body) < 200)
    empty = sum(1 for body in bodies if not body)
    return [
        ("modeli", _thousands(total)),
        ("besedilni segmenti", _thousands(sum(r["num_chunks"] for r in rows))),
        ("segmentov na model", _spread(r["num_chunks"] for r in rows)),
        ("oznak na model", _spread(len(r["tags"] or []) for r in rows)),
        ("dolžina kartice v znakih", _spread(len(b) for b in bodies)),
        ("dolžina kartice v besedah", _spread(len(b.split()) for b in bodies)),
        ("najdaljša kartica v znakih", _thousands(max(len(b) for b in bodies))),
        ("kartice, krajše od 200 znakov", f"{short} ({_share(short, total)})"),
        ("kartice brez besedila", f"{empty} ({_share(empty, total)})"),
        ("prenosi v zadnjih 30 dneh", _spread(r["downloads"] for r in rows)),
        ("všečki", _spread(r["likes"] for r in rows)),
    ]


def attribute_coverage(rows: list[dict]) -> list[tuple[str, str, str]]:
    """The rows of the attribute coverage table."""
    total = len(rows)
    codes = language_codes(rows)

    def tagged(row: dict, prefix: str) -> bool:
        return any(tag.startswith(prefix) for tag in (row["tags"] or []))

    # parameter_count carries no provenance, so exact counts are the non-null
    # rows whose value is not simply the nominal size read off the repo id.
    def exact_params(row: dict) -> bool:
        return (row["parameter_count"] is not None
                and nominal_params(row["model_id"]) != row["parameter_count"])

    attributes = (
        ("pipeline_tag", lambda r: bool(r["pipeline_tag"])),
        ("oznake (tags)", lambda r: bool(r["tags"])),
        ("licenca", lambda r: tagged(r, "license:")),
        ("library_name", lambda r: bool(r["library_name"])),
        ("število parametrov (safetensors)", exact_params),
        ("izvorni model in razmerje", lambda r: tagged(r, "base_model:")),
        ("jezik", lambda r: any(str(t).lower() in codes for t in (r["tags"] or []))),
        ("navedba članka", lambda r: tagged(r, "arxiv:")),
        ("učne zbirke", lambda r: tagged(r, "dataset:")),
    )
    out = []
    for label, predicate in attributes:
        matching = sum(1 for row in rows if predicate(row))
        out.append((label, _thousands(matching), _share(matching, total)))
    return out


def prose_figures(rows: list[dict]) -> list[tuple[str, str]]:
    """The numbers the section quotes in prose rather than in a table."""
    total = len(rows)
    mismatched = [r for r in rows if r["pipeline_tag"] != "text-generation"]
    top_tags = Counter(r["pipeline_tag"] for r in mismatched).most_common(2)

    reuploads = [r for r in rows if catalog._is_quantized_reupload(r["model_id"])]
    declared = [r for r in reuploads if any(
        t.startswith("base_model:quantized:") for t in (r["tags"] or []))]
    unsourced = [r for r in reuploads if not any(
        t.startswith("base_model:") for t in (r["tags"] or []))]

    families = Counter(catalog._family_key(r["model_id"]) for r in rows)
    largest, largest_size = families.most_common(1)[0]
    artifacts = [r for r in rows if catalog._TEST_ARTIFACT_RE.search(r["model_id"])]
    headerless = sum(1 for r in rows
                     if not _YAML_HEADER_RE.match(r["model_card"] or ""))

    return [
        ("drugačen pipeline_tag",
         f"{len(mismatched)} ({_share(len(mismatched), total)}), "
         f"najpogosteje {', '.join(f'{t} {n}' for t, n in top_tags)}"),
        ("ime nosi oznako formata ali kvantizacije",
         f"{len(reuploads)} ({_share(len(reuploads), total)})"),
        ("  od tega z razmerjem quantized",
         f"{len(declared)} ({_share(len(declared), len(reuploads))} teh objav)"),
        ("  od tega brez navedbe izvornega modela",
         f"{len(unsourced)} ({_share(len(unsourced), len(reuploads))} teh objav)"),
        ("družine modelov",
         f"{_thousands(len(families))}, največja {largest} ({largest_size} objav)"),
        ("testni artefakti",
         f"{len(artifacts)} ({_share(len(artifacts), total)})"),
        ("kartice brez glave YAML",
         f"{headerless} (atributov iz glave pri njih ni mogoče prebrati)"),
    ]


def main() -> None:
    rows = catalog._select(f"SELECT {', '.join(COLUMNS)} FROM models", [])
    if not rows:
        raise SystemExit("The models table is empty — load the catalog first.")

    print("\nObseg posnetka kataloga in dolžine besedil")
    print("-" * 62)
    for label, value in size_and_lengths(rows):
        print(f"  {label:<32}{value:>28}")

    print("\nDelež modelov z izpolnjenim atributom")
    print("-" * 62)
    for label, models, share in attribute_coverage(rows):
        print(f"  {label:<32}{models:>14}{share:>14}")

    print("\nŠtevilke, navedene v besedilu razdelka")
    print("-" * 62)
    for label, value in prose_figures(rows):
        print(f"  {label:<42}{value}")
    print()


if __name__ == "__main__":
    main()
