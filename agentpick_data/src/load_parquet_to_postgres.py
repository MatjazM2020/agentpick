#!/usr/bin/env python3
"""
Load embeddings parquet into PostgreSQL metadata table.

Aggregates embeddings by model_id and loads one row per model into
the PostgreSQL `models` table, mapping chunk IDs to Qdrant point IDs.

Architecture:
- Qdrant: vector embeddings + chunk text + semantic retrieval
- PostgreSQL: structured metadata + relational ranking + filtering
- Shared: model_id and chunk_ids for stable interop

Usage:
    python load_parquet_to_postgres.py [--parquet-path /path/to/embeddings.parquet]

Environment variables:
    PARQUET_PATH: Path to embeddings parquet (default: agentpick_data/data/embeddings.parquet)
    POSTGRES_HOST: PostgreSQL host (default: localhost)
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_DB: Database name (default: agentpick)
    POSTGRES_USER: Database user (default: agentpick)
    POSTGRES_PASSWORD: Database password (default: agentpick_password)
"""

import sys
import os
import logging
import argparse
from typing import Optional
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_postgres_connection(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str
) -> psycopg2.extensions.connection:
    """Create PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db,
            user=user,
            password=password
        )
        logger.info(f"Connected to PostgreSQL at {host}:{port}/{db}")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def load_parquet(parquet_path: str) -> pd.DataFrame:
    """Load embeddings parquet file."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from parquet")
    return df


def validate_chunk_ids(chunk_ids: list) -> list:
    """Validate and ensure chunk IDs are integers, sorted ascending."""
    if not chunk_ids:
        return []

    try:
        # Convert to int, deduplicate, sort
        int_ids = list(set([int(cid) for cid in chunk_ids]))
        int_ids.sort()
        return int_ids
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid chunk ID: {e}")
        raise


def aggregate_by_model(df: pd.DataFrame) -> list:
    """
    Group parquet by model_id and aggregate into model records.

    Returns list of dicts, one per model:
    {
        'model_id': str,
        'downloads': int,
        'likes': int,
        'pipeline_tag': str,
        'library_name': str,
        'created_at': datetime,
        'last_modified': datetime,
        'tags': list[str],
        'chunk_ids': list[int],
        'num_chunks': int
    }
    """
    logger.info("Aggregating by model_id...")

    models_dict = {}

    for _, row in df.iterrows():
        model_id = row.get('model_id')
        if pd.isna(model_id):
            logger.warning("Skipping row with missing model_id")
            continue

        model_id = str(model_id).strip()

        if model_id not in models_dict:
            # First occurrence of this model
            models_dict[model_id] = {
                'model_id': model_id,
                'downloads': int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else None,
                'likes': int(row.get('likes', 0)) if pd.notna(row.get('likes')) else None,
                'pipeline_tag': str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
                'library_name': str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
                'created_at': pd.Timestamp(row.get('created_at')).to_pydatetime() if pd.notna(row.get('created_at')) else None,
                'last_modified': pd.Timestamp(row.get('last_modified')).to_pydatetime() if pd.notna(row.get('last_modified')) else None,
                'tags': [t.strip() for t in str(row.get('tags', '')).split(',') if t.strip()] if pd.notna(row.get('tags')) else [],
                'chunk_ids': []
            }

        # Collect chunk ID from this row
        chunk_id = row.get('chunk_id')
        if pd.notna(chunk_id):
            try:
                models_dict[model_id]['chunk_ids'].append(int(chunk_id))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid chunk_id for model {model_id}: {e}")

    # Deduplicate and sort chunk_ids
    logger.info("Deduplicating and sorting chunk IDs...")
    for model_id in models_dict:
        chunk_ids = models_dict[model_id]['chunk_ids']
        models_dict[model_id]['chunk_ids'] = validate_chunk_ids(chunk_ids)
        models_dict[model_id]['num_chunks'] = len(models_dict[model_id]['chunk_ids'])

    logger.info(f"Aggregated into {len(models_dict)} models")
    return list(models_dict.values())


def insert_models_batch(conn: psycopg2.extensions.connection, models: list, batch_size: int = 100):
    """
    Batch insert/upsert models using PostgreSQL ON CONFLICT.

    Uses UPSERT to safely handle re-runs.
    """
    cursor = conn.cursor()
    logger.info(f"Inserting {len(models)} models with batch_size={batch_size}...")

    insert_sql = """
        INSERT INTO models (
            model_id,
            downloads,
            likes,
            pipeline_tag,
            library_name,
            created_at,
            last_modified,
            tags,
            chunk_ids,
            num_chunks
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_id) DO UPDATE SET
            downloads = EXCLUDED.downloads,
            likes = EXCLUDED.likes,
            pipeline_tag = EXCLUDED.pipeline_tag,
            library_name = EXCLUDED.library_name,
            created_at = EXCLUDED.created_at,
            last_modified = EXCLUDED.last_modified,
            tags = EXCLUDED.tags,
            chunk_ids = EXCLUDED.chunk_ids,
            num_chunks = EXCLUDED.num_chunks
    """

    # Prepare batch data
    batch_data = [
        (
            m['model_id'],
            m['downloads'],
            m['likes'],
            m['pipeline_tag'],
            m['library_name'],
            m['created_at'],
            m['last_modified'],
            m['tags'],
            m['chunk_ids'],
            m['num_chunks']
        )
        for m in models
    ]

    try:
        execute_batch(cursor, insert_sql, batch_data, page_size=batch_size)
        conn.commit()
        logger.info(f"Successfully inserted/updated {len(models)} models")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Batch insert failed: {e}")
        raise
    finally:
        cursor.close()


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Load embeddings parquet into PostgreSQL'
    )
    parser.add_argument(
        '--parquet-path',
        type=str,
        default=None,
        help='Path to embeddings parquet file'
    )
    args = parser.parse_args()

    # Resolve parquet path
    parquet_path = args.parquet_path or os.getenv(
        'PARQUET_PATH',
        os.path.join(os.path.dirname(__file__), '../data/embeddings.parquet')
    )

    # PostgreSQL config from environment
    pg_host = os.getenv('POSTGRES_HOST', 'localhost')
    pg_port = int(os.getenv('POSTGRES_PORT', '5432'))
    pg_db = os.getenv('POSTGRES_DB', 'agentpick')
    pg_user = os.getenv('POSTGRES_USER', 'agentpick')
    pg_password = os.getenv('POSTGRES_PASSWORD', 'agentpick_password')

    logger.info("=" * 60)
    logger.info("PostgreSQL Parquet Loader")
    logger.info("=" * 60)
    logger.info(f"Parquet: {parquet_path}")
    logger.info(f"PostgreSQL: {pg_host}:{pg_port}/{pg_db}")

    try:
        # 1. Load parquet
        df = load_parquet(parquet_path)

        # 2. Validate required columns
        required_cols = ['model_id', 'chunk_id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 3. Aggregate by model_id
        models = aggregate_by_model(df)

        # 4. Connect to PostgreSQL
        conn = get_postgres_connection(pg_host, pg_port, pg_db, pg_user, pg_password)

        # 5. Batch insert/upsert
        insert_models_batch(conn, models)

        conn.close()
        logger.info("=" * 60)
        logger.info("✓ Load complete")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
