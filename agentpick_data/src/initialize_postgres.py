"""
Unified PostgreSQL initialization and data loading script.

Orchestrates:
1. Waits for PostgreSQL to be ready (health check with retries)
2. Initializes schema from init_postgres.sql
3. Loads embeddings parquet into models table

Usage:
    python initialize_postgres.py [--parquet-path /path/to/embeddings.parquet]

Environment variables:
    POSTGRES_HOST: PostgreSQL host (default: localhost)
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_DB: Database name (default: agentpick)
    POSTGRES_USER: Database user (default: agentpick)
    POSTGRES_PASSWORD: Database password (default: agentpick_password)
    PARQUET_PATH: Path to embeddings parquet (default: agentpick_data/data/embeddings.parquet)
    POSTGRES_RETRY_INTERVAL: Seconds between connection retries (default: 5)
    POSTGRES_MAX_RETRIES: Max connection retries (default: 30)
"""

import sys
import os
import logging
import argparse
import time
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


def wait_for_postgres(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str,
    retry_interval: int = 5,
    max_retries: int = 30
) -> psycopg2.extensions.connection:
    """
    Wait for PostgreSQL to be ready with retries.
    
    Returns connection once ready.
    """
    logger.info(f"Waiting for PostgreSQL at {host}:{port} (max {max_retries} retries)...")
    
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=db,
                user=user,
                password=password,
                connect_timeout=5
            )
            logger.info(f"✓ PostgreSQL ready on attempt {attempt}")
            return conn
        except psycopg2.Error as e:
            if attempt == max_retries:
                logger.error(f"✗ PostgreSQL connection failed after {max_retries} retries")
                raise
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)


def initialize_schema(conn: psycopg2.extensions.connection, schema_path: str):
    """Execute SQL schema initialization."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    logger.info(f"Initializing schema from {schema_path}...")
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    cursor = conn.cursor()
    try:
        cursor.execute(schema_sql)
        conn.commit()
        logger.info("✓ Schema initialized")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Schema initialization failed: {e}")
        raise
    finally:
        cursor.close()


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
        int_ids = list(set([int(cid) for cid in chunk_ids]))
        int_ids.sort()
        return int_ids
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid chunk ID: {e}")
        raise


def aggregate_by_model(df: pd.DataFrame) -> list:
    """
    Group parquet by model_id and aggregate into model records.

    Returns list of dicts, one per model.
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
            # Handle tags - can be numpy array or list
            tags_val = row.get('tags', [])
            if isinstance(tags_val, (list, tuple)):
                tags_list = [str(t).strip() for t in tags_val if t]
            elif hasattr(tags_val, '__iter__') and not isinstance(tags_val, str):
                # numpy array or similar
                tags_list = [str(t).strip() for t in tags_val if t]
            else:
                tags_list = []
            
            models_dict[model_id] = {
                'model_id': model_id,
                'downloads': int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else None,
                'likes': int(row.get('likes', 0)) if pd.notna(row.get('likes')) else None,
                'pipeline_tag': str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
                'library_name': str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
                'created_at': pd.Timestamp(row.get('created_at')).to_pydatetime() if pd.notna(row.get('created_at')) else None,
                'last_modified': pd.Timestamp(row.get('last_modified')).to_pydatetime() if pd.notna(row.get('last_modified')) else None,
                'tags': tags_list,
                'chunk_ids': []
            }

        chunk_id = row.get('id')
        if pd.notna(chunk_id):
            try:
                models_dict[model_id]['chunk_ids'].append(int(chunk_id))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid chunk ID for model {model_id}: {e}")

    logger.info("Deduplicating and sorting chunk IDs...")
    for model_id in models_dict:
        chunk_ids = models_dict[model_id]['chunk_ids']
        models_dict[model_id]['chunk_ids'] = validate_chunk_ids(chunk_ids)
        models_dict[model_id]['num_chunks'] = len(models_dict[model_id]['chunk_ids'])

    logger.info(f"Aggregated into {len(models_dict)} models")
    return list(models_dict.values())


def insert_models_batch(conn: psycopg2.extensions.connection, models: list, batch_size: int = 100):
    """Batch insert/upsert models using PostgreSQL ON CONFLICT."""
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
        logger.info(f"✓ Successfully inserted/updated {len(models)} models")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Batch insert failed: {e}")
        raise
    finally:
        cursor.close()


def main():
    """Main entry point - orchestrate all initialization steps."""
    parser = argparse.ArgumentParser(
        description='Initialize PostgreSQL schema and load embeddings'
    )
    parser.add_argument(
        '--parquet-path',
        type=str,
        default=None,
        help='Path to embeddings parquet file'
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(script_dir, 'init_postgres.sql')
    
    parquet_path = args.parquet_path or os.getenv(
        'PARQUET_PATH',
        os.path.join(script_dir, '../data/embeddings.parquet')
    )

    # PostgreSQL config
    pg_host = os.getenv('POSTGRES_HOST', 'localhost')
    pg_port = int(os.getenv('POSTGRES_PORT', '5432'))
    pg_db = os.getenv('POSTGRES_DB', 'agentpick')
    pg_user = os.getenv('POSTGRES_USER', 'agentpick')
    pg_password = os.getenv('POSTGRES_PASSWORD', 'agentpick_password')
    retry_interval = int(os.getenv('POSTGRES_RETRY_INTERVAL', '5'))
    max_retries = int(os.getenv('POSTGRES_MAX_RETRIES', '30'))

    logger.info("=" * 70)
    logger.info("PostgreSQL Initialization & Data Loading")
    logger.info("=" * 70)
    logger.info(f"PostgreSQL: {pg_host}:{pg_port}/{pg_db}")
    logger.info(f"Parquet: {parquet_path}")
    logger.info(f"Schema: {schema_path}")

    try:
        # Step 1: Wait for PostgreSQL
        conn = wait_for_postgres(
            pg_host, pg_port, pg_db, pg_user, pg_password,
            retry_interval=retry_interval,
            max_retries=max_retries
        )

        # Step 2: Initialize schema
        initialize_schema(conn, schema_path)

        # Step 3: Load parquet
        df = load_parquet(parquet_path)
        
        required_cols = ['model_id', 'id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        models = aggregate_by_model(df)
        insert_models_batch(conn, models)

        conn.close()
        
        logger.info("=" * 70)
        logger.info("✓ All initialization steps completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
