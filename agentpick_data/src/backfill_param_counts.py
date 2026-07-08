#!/usr/bin/env python3
"""
Backfill missing ``models.parameter_count`` from the size in the model id.

``enrich_models.py`` reads exact counts from HuggingFace safetensors metadata,
but many repos (GGUF re-uploads, old pytorch-bin models) have none, leaving
``parameter_count`` NULL. Most of those ids carry a nominal size ("7b",
"125m", "Qwen3-32B-GGUF"); this script parses it and stores the nominal count
(7B -> 7_000_000_000). Only NULL rows are touched — exact safetensors counts
are never overwritten.

Validated against the 1,175 models with known counts: the parsed nominal size
agrees with the exact count for ~89% of ids that contain one (the rest are
mostly draft/speculator repos whose stored count covers only a small artifact).

Usage:
    cd agentpick_data
    source venv/bin/activate
    POSTGRES_PORT=5433 python src/backfill_param_counts.py --dry-run
    POSTGRES_PORT=5433 python src/backfill_param_counts.py
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Optional

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# A size token like "7b", "1.5B", "125m", standing alone between separators.
# The lookarounds reject tokens inside longer runs ("8x7B", "Q8_0", "8bit")
# and version numbers ("v1.5"). In MoE ids ("30B-A3B") the first token is the
# total parameter count, so we take the first match.
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([bBmM])(?![A-Za-z0-9])")


def parse_params_from_id(model_id: str) -> Optional[int]:
    """Nominal parameter count parsed from a model id, or None."""
    name = model_id.split("/", 1)[-1]
    m = _SIZE_RE.search(name)
    if not m:
        return None
    value = float(m.group(1))
    mult = 1_000_000_000 if m.group(2) in "bB" else 1_000_000
    count = int(value * mult)
    return count or None


def connect_postgres() -> psycopg2.extensions.connection:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB", "agentpick")
    user = os.getenv("POSTGRES_USER", "agentpick")
    password = os.getenv("POSTGRES_PASSWORD", "agentpick_password")
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=db,
        user=user,
        password=password,
        connect_timeout=10,
    )
    logger.info("Connected to PostgreSQL at %s:%s/%s", host, port, db)
    return conn


def backfill(conn: psycopg2.extensions.connection, dry_run: bool = False) -> dict:
    """Fill parameter_count for NULL rows whose id contains a size. Returns stats."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT model_id FROM models WHERE parameter_count IS NULL ORDER BY model_id"
        )
        missing = [r[0] for r in cur.fetchall()]

    updates = [
        (model_id, count)
        for model_id in missing
        if (count := parse_params_from_id(model_id)) is not None
    ]
    logger.info(
        "%d models missing parameter_count, %d have a size in their id (dry_run=%s)",
        len(missing), len(updates), dry_run,
    )

    for model_id, count in updates:
        logger.info("%s -> %s", model_id, f"{count:,}")
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE models SET parameter_count = %s WHERE model_id = %s",
                    (count, model_id),
                )
    if not dry_run:
        conn.commit()

    stats = {"missing": len(missing), "updated": 0 if dry_run else len(updates)}
    logger.info("Backfill done: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill NULL parameter_count values from sizes in model ids",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log parsed counts without updating PostgreSQL",
    )
    args = parser.parse_args()

    conn = connect_postgres()
    try:
        backfill(conn, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
