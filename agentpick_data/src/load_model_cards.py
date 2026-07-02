#!/usr/bin/env python3
"""
Fetch HuggingFace model card markdown and update PostgreSQL ``models`` rows.

Reads model IDs from ``processed_models.txt`` (or ``--models-file``), downloads
each repo's ``README.md``, and stores the raw markdown in ``models.model_card``.

Docker / networking
-------------------
PostgreSQL runs in docker-compose as service ``postgres`` (container
``agentpick-postgres``). From your **host machine** (where this script runs),
connect via the published port:

    docker compose up -d postgres
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 python src/load_model_cards.py

Inside the Docker network (e.g. backend container), Postgres is ``postgres:5432`` —
this script is intended to run from ``agentpick_data/`` on the host with the
mapped port, same as ``initialize_postgres.py``.

Optional ``HF_TOKEN`` improves rate limits and unlocks gated models.

Usage:
    cd agentpick_data
    source venv/bin/activate
    POSTGRES_PORT=5433 python src/load_model_cards.py
    POSTGRES_PORT=5433 python src/load_model_cards.py --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import psycopg2
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIGRATION_SQL = """
ALTER TABLE models
ADD COLUMN IF NOT EXISTS model_card TEXT;
"""


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


def ensure_model_card_column(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(MIGRATION_SQL)
    conn.commit()
    logger.info("Ensured models.model_card column exists")


def read_model_ids(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Models file not found: {path}")

    ids: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            model_id = line.strip()
            if not model_id or model_id.startswith("#"):
                continue
            if model_id not in seen:
                seen.add(model_id)
                ids.append(model_id)
    logger.info("Loaded %d unique model IDs from %s", len(ids), path)
    return ids


def fetch_existing_model_cards(conn: psycopg2.extensions.connection) -> dict[str, Optional[str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT model_id, model_card FROM models")
        return {row[0]: row[1] for row in cur.fetchall()}


def fetch_model_card(
    model_id: str,
    token: Optional[str],
    retries: int = 3,
) -> Optional[str]:
    """Download README.md for a model and return its markdown content."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
                token=token,
            )
            return Path(path).read_text(encoding="utf-8")
        except EntryNotFoundError:
            logger.debug("No README.md for %s", model_id)
            return None
        except RepositoryNotFoundError:
            logger.warning("Repository not found: %s", model_id)
            return None
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "HF download retry %d/%d for %s in %ss: %s",
                    attempt,
                    retries,
                    model_id,
                    wait,
                    e,
                )
                time.sleep(wait)
    logger.error("HF download failed for %s after %d attempts: %s", model_id, retries, last_error)
    return None


def update_model_card(
    conn: psycopg2.extensions.connection,
    model_id: str,
    model_card: Optional[str],
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE models
            SET model_card = %s
            WHERE model_id = %s
            """,
            (model_card, model_id),
        )
        updated = cur.rowcount > 0
    conn.commit()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load HuggingFace model card markdown into PostgreSQL models table",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="Path to processed_models.txt (default: ../data/processed_models.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N models (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between HuggingFace downloads (default: 0.2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even when model_card is already set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch from HF and log results without updating PostgreSQL",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_file = args.models_file or os.path.join(script_dir, "../data/processed_models.txt")
    models_file = os.path.abspath(models_file)

    logger.info("=" * 70)
    logger.info("Load model cards → PostgreSQL")
    logger.info("=" * 70)
    logger.info("Models file: %s", models_file)
    logger.info(
        "PostgreSQL target: %s:%s/%s",
        os.getenv("POSTGRES_HOST", "localhost"),
        os.getenv("POSTGRES_PORT", "5432"),
        os.getenv("POSTGRES_DB", "agentpick"),
    )
    if args.dry_run:
        logger.info("DRY RUN — no database writes")

    model_ids = read_model_ids(models_file)
    if args.limit:
        model_ids = model_ids[: args.limit]
        logger.info("Limited to first %d models", len(model_ids))

    token = os.getenv("HF_TOKEN") or None

    conn = connect_postgres()
    try:
        ensure_model_card_column(conn)

        db_rows = fetch_existing_model_cards(conn)
        db_model_ids = set(db_rows)

        stats = {
            "fetched": 0,
            "with_card": 0,
            "missing_card": 0,
            "updated": 0,
            "not_in_db": 0,
            "skipped_existing": 0,
            "errors": 0,
        }

        for i, model_id in enumerate(model_ids, 1):
            if model_id not in db_model_ids:
                logger.warning("[%d/%d] %s — not in PostgreSQL, skipping", i, len(model_ids), model_id)
                stats["not_in_db"] += 1
                continue

            existing = db_rows.get(model_id)
            if existing is not None and not args.force:
                logger.debug("[%d/%d] %s — already set (%d chars), skipping", i, len(model_ids), model_id, len(existing))
                stats["skipped_existing"] += 1
                continue

            model_card = fetch_model_card(model_id, token=token)
            stats["fetched"] += 1
            if model_card is not None:
                stats["with_card"] += 1
                logger.info(
                    "[%d/%d] %s → %d chars",
                    i,
                    len(model_ids),
                    model_id,
                    len(model_card),
                )
            else:
                stats["missing_card"] += 1
                logger.info("[%d/%d] %s → no README.md", i, len(model_ids), model_id)

            if not args.dry_run:
                if update_model_card(conn, model_id, model_card):
                    stats["updated"] += 1
                    db_rows[model_id] = model_card
                else:
                    stats["errors"] += 1
                    logger.error("[%d/%d] %s — UPDATE matched 0 rows", i, len(model_ids), model_id)

            if args.delay > 0 and i < len(model_ids):
                time.sleep(args.delay)

        logger.info("=" * 70)
        logger.info(
            "Done: fetched=%d with_card=%d missing_card=%d updated=%d "
            "skipped_existing=%d not_in_db=%d errors=%d",
            stats["fetched"],
            stats["with_card"],
            stats["missing_card"],
            stats["updated"],
            stats["skipped_existing"],
            stats["not_in_db"],
            stats["errors"],
        )
        logger.info("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
