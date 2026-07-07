#!/usr/bin/env python3
"""
Enrich PostgreSQL ``models`` rows with HuggingFace model cards and parameter counts.

For every model already loaded into the ``models`` table, downloads the repo's
``README.md`` (stored in ``models.model_card``) and reads ``safetensors.total``
from ``model_info()`` (stored in ``models.parameter_count``). Fields that are
already set are skipped unless ``--force`` is given, so re-runs are cheap.

Runs automatically as the last step of ``initialize_postgres.py``; use this CLI
to re-run enrichment on its own.

Docker / networking
-------------------
PostgreSQL runs in docker-compose as service ``postgres`` (container
``agentpick-postgres``). From your **host machine** (where this script runs),
connect via the published port:

    docker compose up -d postgres
    POSTGRES_PORT=5433 python src/enrich_models.py

Optional ``HF_TOKEN`` improves rate limits and unlocks gated models.

Usage:
    cd agentpick_data
    source venv/bin/activate
    POSTGRES_PORT=5433 python src/enrich_models.py
    POSTGRES_PORT=5433 python src/enrich_models.py --limit 20 --dry-run
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
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
                    attempt, retries, model_id, wait, e,
                )
                time.sleep(wait)
    logger.error("HF download failed for %s after %d attempts: %s", model_id, retries, last_error)
    return None


def fetch_parameter_count(api: HfApi, model_id: str, retries: int = 3) -> Optional[int]:
    """Read the total parameter count from HuggingFace safetensors metadata."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            info = api.model_info(model_id)
            st = info.safetensors
            if st is None or st.total is None:
                return None
            return int(st.total)
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "HF API retry %d/%d for %s in %ss: %s",
                    attempt, retries, model_id, wait, e,
                )
                time.sleep(wait)
    logger.error("HF API failed for %s after %d attempts: %s", model_id, retries, last_error)
    return None


def enrich_models(
    conn: psycopg2.extensions.connection,
    limit: Optional[int] = None,
    delay: float = 0.2,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Fill in model_card and parameter_count for models missing them.

    Returns a stats dict.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_id, model_card IS NOT NULL, parameter_count IS NOT NULL
            FROM models
            ORDER BY downloads DESC NULLS LAST
            """
        )
        rows = cur.fetchall()

    if not force:
        rows = [r for r in rows if not (r[1] and r[2])]
    if limit:
        rows = rows[:limit]

    logger.info("Enriching %d models (force=%s, dry_run=%s)", len(rows), force, dry_run)

    token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=token)

    stats = {
        "models": len(rows),
        "cards_fetched": 0,
        "cards_found": 0,
        "counts_fetched": 0,
        "counts_found": 0,
        "updated": 0,
    }

    for i, (model_id, has_card, has_count) in enumerate(rows, 1):
        updates = {}

        if force or not has_card:
            model_card = fetch_model_card(model_id, token=token)
            stats["cards_fetched"] += 1
            if model_card is not None:
                stats["cards_found"] += 1
            updates["model_card"] = model_card

        if force or not has_count:
            count = fetch_parameter_count(api, model_id)
            stats["counts_fetched"] += 1
            if count is not None:
                stats["counts_found"] += 1
            updates["parameter_count"] = count

        card_msg = "kept"
        if "model_card" in updates:
            card = updates["model_card"]
            card_msg = f"{len(card)} chars" if card is not None else "none"
        count_msg = "kept"
        if "parameter_count" in updates:
            count = updates["parameter_count"]
            count_msg = f"{count:,}" if count is not None else "none"
        logger.info("[%d/%d] %s → card=%s params=%s", i, len(rows), model_id, card_msg, count_msg)

        if updates and not dry_run:
            set_clause = ", ".join(f"{col} = %s" for col in updates)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE models SET {set_clause} WHERE model_id = %s",
                    (*updates.values(), model_id),
                )
            conn.commit()
            stats["updated"] += 1

        if delay > 0 and i < len(rows):
            time.sleep(delay)

    logger.info(
        "Enrichment done: models=%d cards_fetched=%d cards_found=%d "
        "counts_fetched=%d counts_found=%d updated=%d",
        stats["models"], stats["cards_fetched"], stats["cards_found"],
        stats["counts_fetched"], stats["counts_found"], stats["updated"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich PostgreSQL models table with HuggingFace model cards and parameter counts",
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
        help="Seconds to wait between HuggingFace requests (default: 0.2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even when model_card / parameter_count are already set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch from HF and log results without updating PostgreSQL",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Enrich models → PostgreSQL")
    logger.info("=" * 70)
    logger.info(
        "PostgreSQL target: %s:%s/%s",
        os.getenv("POSTGRES_HOST", "localhost"),
        os.getenv("POSTGRES_PORT", "5432"),
        os.getenv("POSTGRES_DB", "agentpick"),
    )
    if args.dry_run:
        logger.info("DRY RUN — no database writes")

    conn = connect_postgres()
    try:
        enrich_models(
            conn,
            limit=args.limit,
            delay=args.delay,
            force=args.force,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
