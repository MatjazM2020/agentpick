"""
Catalog data access — the ground truth the agent's tools read from.

Two sources, one interface:
- Qdrant: semantic (vector) search over model-card chunks for fuzzy intent.
- PostgreSQL ``models`` table: authoritative metadata (downloads, likes,
  parameter_count, pipeline_tag, tags) and the full ``model_card`` markdown.

Everything blocking (DB / vector queries) lives here; the async tool layer in
``src/tools.py`` offloads these to a thread pool.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from typing import Any, Optional

from src.core import config

logger = logging.getLogger(__name__)

# Columns exposed to the agent (model_card handled separately due to size).
_META_COLUMNS = (
    "model_id, downloads, likes, pipeline_tag, library_name, "
    "tags, parameter_count, last_modified"
)

_CARD_EXCERPT_CHARS = 400
_CARD_FULL_CHARS = 4000


# ---------------------------------------------------------------------------
# PostgreSQL (lazy, pooled)
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()


class CatalogUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached; surfaced to the LLM as an error."""


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError as e:  # pragma: no cover
            raise CatalogUnavailable(f"psycopg2 not installed: {e}") from e
        try:
            _pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=int(os.getenv("POSTGRES_POOL_SIZE", "5")),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                dbname=os.getenv("POSTGRES_DB", "agentpick"),
                user=os.getenv("POSTGRES_USER", "agentpick"),
                password=os.getenv("POSTGRES_PASSWORD", "agentpick_password"),
                connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
            )
        except Exception as e:
            raise CatalogUnavailable(f"Could not connect to PostgreSQL: {e}") from e
    return _pool


def _select(sql: str, params: list) -> list[dict]:
    try:
        from psycopg2.extras import RealDictCursor
    except ImportError as e:  # pragma: no cover
        raise CatalogUnavailable(f"psycopg2 not installed: {e}") from e
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        raise CatalogUnavailable(f"PostgreSQL query failed: {e}") from e
    finally:
        pool.putconn(conn)


def _shape(row: dict, *, card_excerpt: Optional[str] = None) -> dict:
    """Normalize a DB row into a compact, LLM-friendly dict."""
    out = {
        "model_id": row.get("model_id"),
        "pipeline_tag": row.get("pipeline_tag"),
        "parameter_count": row.get("parameter_count"),
        "downloads": row.get("downloads") or 0,
        "likes": row.get("likes") or 0,
        "tags": list(row.get("tags") or [])[:12],
    }
    if card_excerpt is not None:
        out["card_excerpt"] = card_excerpt
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def _meta_for_ids(model_ids: list[str]) -> dict[str, dict]:
    """Authoritative metadata + short card excerpt for a set of model ids."""
    if not model_ids:
        return {}
    sql = (
        f"SELECT {_META_COLUMNS}, LEFT(model_card, %s) AS card_excerpt "
        "FROM models WHERE model_id = ANY(%s)"
    )
    rows = _select(sql, [_CARD_EXCERPT_CHARS, list(model_ids)])
    return {
        r["model_id"]: _shape(r, card_excerpt=(r.get("card_excerpt") or "").strip())
        for r in rows
        if r.get("model_id")
    }


# ---------------------------------------------------------------------------
# Qdrant (lazy)
# ---------------------------------------------------------------------------

_qclient = None
_qclient_lock = threading.Lock()


def _qdrant():
    global _qclient
    if _qclient is not None:
        return _qclient
    with _qclient_lock:
        if _qclient is None:
            from qdrant_client import QdrantClient

            _qclient = QdrantClient(url=config.qdrant_url())
            logger.info("[catalog] Qdrant client -> %s", config.qdrant_url())
    return _qclient


# ---------------------------------------------------------------------------
# Public catalog operations (used by tools)
# ---------------------------------------------------------------------------


def semantic_search(query: str, limit: int = 8) -> list[dict]:
    """
    Vector search over model-card chunks; returns ranked, de-duplicated models.

    Each result carries authoritative PostgreSQL metadata plus the best-matching
    card excerpt, so the agent can judge relevance without a follow-up call.
    """
    from src.core.llm import embed

    embedding = embed(query)
    kwargs: dict[str, Any] = {
        "collection_name": config.QDRANT_COLLECTION,
        "query": embedding,
        "limit": config.QDRANT_TOP_K_CHUNKS,
        "with_payload": True,
    }
    if config.QDRANT_QUERY_USING:
        kwargs["using"] = config.QDRANT_QUERY_USING
    points = list(_qdrant().query_points(**kwargs).points)

    agg: dict[str, dict] = defaultdict(
        lambda: {"score_sum": 0.0, "count": 0, "best_score": float("-inf"), "excerpt": ""}
    )
    for p in points:
        payload = p.payload or {}
        mid = payload.get("model_id")
        if not mid:
            continue
        a = agg[mid]
        a["score_sum"] += p.score
        a["count"] += 1
        if p.score >= a["best_score"]:
            a["best_score"] = p.score
            text = (payload.get("text") or "").strip()
            if text:
                a["excerpt"] = text[:_CARD_EXCERPT_CHARS]

    # Rank models by their best-matching chunk: averaging chunk scores would
    # penalize models with long cards (many retrieved, partially relevant chunks).
    ranked = sorted(
        agg.items(),
        key=lambda kv: kv[1]["best_score"],
        reverse=True,
    )[:limit]

    ids = [mid for mid, _ in ranked]
    try:
        meta = _meta_for_ids(ids)
    except CatalogUnavailable as e:
        logger.warning("[catalog] metadata enrich skipped: %s", e)
        meta = {}

    results: list[dict] = []
    for mid, a in ranked:
        item = meta.get(mid, {"model_id": mid})
        excerpt = a["excerpt"] or item.get("card_excerpt")
        if excerpt:
            item = {**item, "card_excerpt": excerpt}
        results.append(item)
    logger.info("[catalog] semantic_search '%s' -> %d models", query[:60], len(results))
    return results


_SORT_SQL = {
    "downloads": "downloads DESC NULLS LAST",
    "likes": "likes DESC NULLS LAST",
    "smallest": "parameter_count ASC NULLS LAST",
    "largest": "parameter_count DESC NULLS LAST",
    "newest": "last_modified DESC NULLS LAST",
}


# A tag filter matching this few models is more likely a wrong tag than a
# genuinely rare capability; the result then carries a warning + similar tags.
_SPARSE_TAG_THRESHOLD = 25


def _similar_tags(tag: str) -> list[str]:
    """Existing tags containing ``tag`` as a substring, with model counts."""
    # Namespaced tags (license:, dataset:, region:, ...) are metadata, not
    # capability tags, so they make poor suggestions.
    rows = _select(
        "SELECT t AS tag, COUNT(*) AS n FROM models, unnest(tags) AS t "
        "WHERE t ILIKE %s AND t NOT LIKE '%%:%%' "
        "GROUP BY t ORDER BY n DESC LIMIT 6",
        [f"%{tag}%"],
    )
    return [f"{r['tag']} ({r['n']} models)" for r in rows]


def filter_models(
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
    name_contains: Optional[str] = None,
    min_params_b: Optional[float] = None,
    max_params_b: Optional[float] = None,
    sort_by: str = "downloads",
    limit: int = 8,
) -> dict:
    """
    Structured SQL query over the catalog for precise constraints and rankings.

    Returns ``{"models": [...], "total_matches": N, "warnings": [...]}`` so the
    agent can tell "nothing/few match" apart from "the filter is too narrow":
    warnings flag sparse tags (with similar existing tags) and models excluded
    by a size filter only because their parameter count is unknown.
    """
    where: list[str] = []
    wparams: list[Any] = []
    size_clauses: list[str] = []
    size_params: list[Any] = []

    tt = (task_type or "").strip().lower()
    if tt and tt != "general":
        where.append("LOWER(pipeline_tag) = %s")
        wparams.append(tt)
    if tag and tag.strip():
        where.append("%s = ANY(tags)")
        wparams.append(tag.strip())
    if name_contains and name_contains.strip():
        where.append("model_id ILIKE %s")
        wparams.append(f"%{name_contains.strip()}%")
    if min_params_b is not None:
        size_clauses.append("parameter_count >= %s")
        size_params.append(int(float(min_params_b) * 1_000_000_000))
    if max_params_b is not None:
        size_clauses.append("parameter_count <= %s")
        size_params.append(int(float(max_params_b) * 1_000_000_000))

    order = _SORT_SQL.get((sort_by or "").strip().lower(), _SORT_SQL["downloads"])
    limit = max(1, min(int(limit), 25))

    all_clauses = where + size_clauses
    where_sql = (" WHERE " + " AND ".join(all_clauses)) if all_clauses else ""
    sql = (
        f"SELECT {_META_COLUMNS}, LEFT(model_card, %s) AS card_excerpt FROM models"
        f"{where_sql} ORDER BY {order} LIMIT %s"
    )
    rows = _select(sql, [_CARD_EXCERPT_CHARS] + wparams + size_params + [limit])

    total = _select(
        f"SELECT COUNT(*) AS n FROM models{where_sql}", wparams + size_params
    )[0]["n"]

    warnings: list[str] = []
    if tag and tag.strip() and total < _SPARSE_TAG_THRESHOLD:
        similar = _similar_tags(tag.strip())
        msg = (
            f"The tag filter '{tag.strip()}' matched only {total} of the whole "
            "catalog. Tags are sparse and inconsistently applied, so this "
            "filter may be excluding relevant models — consider name_contains "
            "or search_models instead."
        )
        if similar:
            msg += " Similar existing tags: " + ", ".join(similar) + "."
        warnings.append(msg)
    if size_clauses:
        unknown_sql = " AND ".join(where + ["parameter_count IS NULL"])
        unknown = _select(
            f"SELECT COUNT(*) AS n FROM models WHERE {unknown_sql}", wparams
        )[0]["n"]
        if unknown:
            warnings.append(
                f"Note: {unknown} models matching the other filters have an "
                "unknown parameter count and are not included in this "
                "size-filtered result."
            )

    logger.info(
        "[catalog] filter_models(task=%s, tag=%s, name=%s, %s-%sB, sort=%s) -> %d/%d",
        task_type, tag, name_contains, min_params_b, max_params_b, sort_by,
        len(rows), total,
    )
    return {
        "models": [_shape(r, card_excerpt=(r.get("card_excerpt") or "").strip()) for r in rows],
        "total_matches": total,
        "warnings": warnings,
    }


def get_model_details(model_id: str) -> Optional[dict]:
    """Full metadata + truncated model card (README) for one model, or None."""
    sql = (
        f"SELECT {_META_COLUMNS}, library_name, LEFT(model_card, %s) AS model_card "
        "FROM models WHERE model_id = %s"
    )
    rows = _select(sql, [_CARD_FULL_CHARS, model_id])
    if not rows:
        return None
    row = rows[0]
    detail = _shape(row)
    detail["library_name"] = row.get("library_name")
    detail["last_modified"] = str(row.get("last_modified") or "") or None
    detail["model_card"] = (row.get("model_card") or "").strip() or None
    return {k: v for k, v in detail.items() if v not in (None, "")}
