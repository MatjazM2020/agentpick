"""
PostgreSQL metadata access for popularity-based routing.

Qdrant handles semantic retrieval; PostgreSQL (the ``models`` table loaded by
``agentpick_data/src/load_parquet_to_postgres.py``) handles structured popularity
queries: global top-N by downloads/likes, threshold filtering, and authoritative
download/like counts for hybrid rerank.

All queries return candidate dicts shaped exactly like the Qdrant retriever
output (``id``, ``score``, ``metadata``, ``num_chunks``, ``point_ids``) so the
rest of the pipeline (evaluator, synthesizer) is unchanged. On any connection /
driver / empty-table problem a ``PostgresUnavailable`` is raised so callers can
fall back to Qdrant.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Columns selected for every candidate query (mirrors init_postgres.sql)
_SELECT_COLUMNS = (
    "model_id, downloads, likes, pipeline_tag, library_name, "
    "created_at, last_modified, tags, chunk_ids, num_chunks"
)


class PostgresUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached or queried; callers fall back to Qdrant."""


def _connect():
    """Open a short-lived PostgreSQL connection from env (see docker-compose.yaml)."""
    try:
        import psycopg2
    except ImportError as e:  # pragma: no cover - depends on install
        raise PostgresUnavailable(f"psycopg2 not installed: {e}") from e

    try:
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "agentpick"),
            user=os.getenv("POSTGRES_USER", "agentpick"),
            password=os.getenv("POSTGRES_PASSWORD", "agentpick_password"),
            connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
        )
    except Exception as e:
        raise PostgresUnavailable(f"Could not connect to PostgreSQL: {e}") from e


def _run_select(sql: str, params: list) -> List[dict]:
    """Run a SELECT and return rows as dicts. Raises PostgresUnavailable on failure."""
    conn = _connect()
    try:
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        raise PostgresUnavailable(f"PostgreSQL query failed: {e}") from e
    finally:
        conn.close()


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
