"""
PostgreSQL metadata access for popularity-based routing.

Qdrant handles semantic retrieval; PostgreSQL (the ``models`` table loaded by
``agentpick_data/src/load_parquet_to_postgres.py``) handles structured popularity
queries: global top-N by downloads/likes, threshold filtering, and authoritative
download/like counts for hybrid rerank.

All queries return candidate dicts shaped exactly like the Qdrant retriever
output (``id``, ``score``, ``metadata``, ``num_chunks``, ``point_ids``) so the
rest of the pipeline (ranker) is unchanged. On any connection /
driver / empty-table problem a ``PostgresUnavailable`` is raised so callers can
fall back to Qdrant.
"""

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.core.state import RecommendationState

logger = logging.getLogger(__name__)


# Columns selected for every candidate query (mirrors init_postgres.sql)
_SELECT_COLUMNS = (
    "model_id, downloads, likes, pipeline_tag, library_name, "
    "created_at, last_modified, tags, chunk_ids, num_chunks, parameter_count, "
    "model_card"
)


class PostgresUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached or queried; callers fall back to Qdrant."""


# --- Connection pool (lazy-init, thread-safe) ---

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Return a process-wide ThreadedConnectionPool (created on first call)."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError as e:  # pragma: no cover
            raise PostgresUnavailable(f"psycopg2 not installed: {e}") from e

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
            raise PostgresUnavailable(f"Could not create PostgreSQL pool: {e}") from e
    return _pool


def _run_select(sql: str, params: list) -> List[dict]:
    """Run a SELECT and return rows as dicts. Raises PostgresUnavailable on failure."""
    try:
        from psycopg2.extras import RealDictCursor
    except ImportError as e:  # pragma: no cover
        raise PostgresUnavailable(f"psycopg2 not installed: {e}") from e

    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        raise PostgresUnavailable(f"PostgreSQL query failed: {e}") from e
    finally:
        pool.putconn(conn)


def _row_to_candidate(row: dict, score: float = 0.0) -> Dict[str, Any]:
    """Map a ``models`` row to the Qdrant-retriever candidate shape."""
    created = row.get("created_at")
    modified = row.get("last_modified")
    metadata = {
        "model_id": row.get("model_id"),
        "downloads": row.get("downloads") or 0,
        "likes": row.get("likes") or 0,
        "pipeline_tag": row.get("pipeline_tag"),
        "library_name": row.get("library_name"),
        "tags": list(row.get("tags") or []),
        "parameter_count": row.get("parameter_count"),
        "model_card": row.get("model_card"),
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "last_modified": modified.isoformat() if hasattr(modified, "isoformat") else modified,
    }
    return {
        "id": row.get("model_id"),
        "score": score,
        "metadata": metadata,
        "num_chunks": int(row.get("num_chunks") or 0),
        "point_ids": list(row.get("chunk_ids") or []),
    }


def query_top_models(
    task_type: Optional[str] = None,
    tags: Optional[Any] = None,
    sort_by: Optional[str] = None,
    min_downloads: Optional[int] = None,
    min_likes: Optional[int] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Global popularity query against the ``models`` table.

    Orders by downloads (default) or likes, applies optional download/like
    thresholds, and optionally narrows by ``pipeline_tag`` / ``tags`` — relaxing
    only the pipeline_tag/tags narrowing if it would return nothing (explicit
    thresholds are always kept). Returns retriever-shaped candidates.
    """
    order_col = "likes" if (sort_by or "").strip().lower() == "likes" else "downloads"

    threshold_clauses: List[str] = []
    threshold_params: List[Any] = []
    if min_downloads is not None:
        threshold_clauses.append("downloads >= %s")
        threshold_params.append(int(min_downloads))
    if min_likes is not None:
        threshold_clauses.append("likes >= %s")
        threshold_params.append(int(min_likes))

    narrow_clauses: List[str] = []
    narrow_params: List[Any] = []
    tt = (task_type or "").strip().lower()
    if tt and tt != "general":
        narrow_clauses.append("pipeline_tag = %s")
        narrow_params.append(tt)
    if tags:
        tag_list = [tags] if isinstance(tags, str) else list(tags)
        if tag_list:
            narrow_clauses.append("tags && %s")
            narrow_params.append(tag_list)

    def _select(where_clauses: List[str], params: List[Any]) -> List[dict]:
        sql = f"SELECT {_SELECT_COLUMNS} FROM models"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += f" ORDER BY {order_col} DESC NULLS LAST LIMIT %s"
        return _run_select(sql, params + [int(limit)])

    rows = _select(threshold_clauses + narrow_clauses, threshold_params + narrow_params)
    if not rows and narrow_clauses:
        logger.info(
            "[postgres.query_top_models] task/tag narrowing returned 0 rows; "
            "relaxing to thresholds only"
        )
        rows = _select(threshold_clauses, threshold_params)

    candidates = [_row_to_candidate(r) for r in rows]
    logger.info(
        f"[postgres.query_top_models] returned {len(candidates)} models "
        f"(order_by={order_col}, min_downloads={min_downloads}, min_likes={min_likes})"
    )
    return candidates


def fetch_metadata(model_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Authoritative downloads/likes (and other metadata) for ``model_ids``.

    Used by hybrid mode to filter/rerank Qdrant candidates. Returns
    ``{model_id: metadata}`` for the models found in PostgreSQL.
    """
    if not model_ids:
        return {}

    sql = (
        f"SELECT {_SELECT_COLUMNS} FROM models WHERE model_id = ANY(%s)"
    )
    rows = _run_select(sql, [list(model_ids)])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        candidate = _row_to_candidate(row)
        out[candidate["id"]] = candidate["metadata"]
    logger.info(
        f"[postgres.fetch_metadata] resolved {len(out)}/{len(model_ids)} model ids"
    )
    return out


def enrich_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge authoritative PostgreSQL metadata into retriever-shaped candidates."""
    if not candidates:
        return candidates

    model_ids = [m["id"] for m in candidates if m.get("id")]
    try:
        pg_meta = fetch_metadata(model_ids)
    except PostgresUnavailable as e:
        logger.warning(f"[postgres.enrich_candidates] unavailable: {e}")
        return candidates

    if not pg_meta:
        return candidates

    for m in candidates:
        meta = pg_meta.get(m["id"])
        if not meta:
            continue
        md = m.setdefault("metadata", {})
        for key in (
            "downloads",
            "likes",
            "pipeline_tag",
            "library_name",
            "tags",
            "parameter_count",
            "last_modified",
            "model_card",
        ):
            if meta.get(key) is not None:
                md[key] = meta[key]
    return candidates


def fetch_model_cards(model_ids: List[str]) -> Dict[str, str]:
    """Return ``{model_id: model_card markdown}`` for picked models (PostgreSQL source of truth)."""
    if not model_ids:
        return {}

    sql = "SELECT model_id, model_card FROM models WHERE model_id = ANY(%s)"
    rows = _run_select(sql, [list(model_ids)])
    out: Dict[str, str] = {}
    for row in rows:
        mid = row.get("model_id")
        card = (row.get("model_card") or "").strip()
        if mid and card:
            out[mid] = card
    logger.info(
        f"[postgres.fetch_model_cards] resolved {len(out)}/{len(model_ids)} model cards"
    )
    return out


def enrich_and_filter(state: "RecommendationState") -> "RecommendationState":
    """
    Enrich Qdrant candidates with authoritative PostgreSQL metadata and apply
    popularity thresholds when in hybrid or popularity-aware mode.
    """
    if not state.retrieved_models:
        return state

    pop = state.popularity or {}
    mode = pop.get("mode", "none")
    min_downloads = pop.get("min_downloads")
    min_likes = pop.get("min_likes")
    model_ids = [m["id"] for m in state.retrieved_models if m.get("id")]

    try:
        pg_meta = fetch_metadata(model_ids)
    except PostgresUnavailable as e:
        logger.warning(f"[postgres.enrich_and_filter] unavailable: {e}")
        state.agent_logs.append(f"PostgreSQL enrich skipped: {e}")
        return state

    if not pg_meta:
        return state

    kept = []
    for m in state.retrieved_models:
        meta = pg_meta.get(m["id"])
        if meta:
            m["metadata"]["downloads"] = meta.get("downloads", m["metadata"].get("downloads", 0))
            m["metadata"]["likes"] = meta.get("likes", m["metadata"].get("likes", 0))
            for key in ("pipeline_tag", "library_name", "tags", "last_modified"):
                if meta.get(key) is not None:
                    m["metadata"][key] = meta[key]

        if mode in ("hybrid", "popularity_only"):
            dl = m["metadata"].get("downloads", 0) or 0
            lk = m["metadata"].get("likes", 0) or 0
            if min_downloads is not None and dl < min_downloads:
                continue
            if min_likes is not None and lk < min_likes:
                continue

        kept.append(m)

    if kept:
        state.retrieved_models = kept
        state.agent_logs.append(f"PostgreSQL: enriched and filtered to {len(kept)} models")
    return state
