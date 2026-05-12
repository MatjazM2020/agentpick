"""Parquet storage and schema management."""

import logging
from typing import List, Dict, Any
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize record to ensure schema consistency.
    
    Args:
        record: Record dictionary
        
    Returns:
        Normalized record with all expected fields
    """
    record.setdefault('downloads', 0)
    record.setdefault('likes', 0)
    record.setdefault('tags', [])
    record.setdefault('pipeline_tag', None)
    record.setdefault('library_name', None)
    record.setdefault('created_at', None)
    record.setdefault('last_modified', None)
    record.setdefault('chunk_index', 0)
    return record


class ParquetWriter:
    """Streaming writer for embeddings to Parquet format.
    
    Uses PyArrow's ParquetWriter for efficient appending without
    reloading entire dataset at each flush. Avoids O(n²) performance
    degradation that occurs with concat_tables.
    """
    
    def __init__(self, path: Path):
        """Initialize Parquet writer.
        
        Args:
            path: Path to output Parquet file
        """
        self.path = path
        self.buffer: List[Dict[str, Any]] = []
        self.writer = None
        self.schema = None
    
    def add(self, records: List[Dict[str, Any]]):
        """Add records to buffer and flush if threshold reached.
        
        Args:
            records: List of record dictionaries
        """
        self.buffer.extend(records)
        if len(self.buffer) >= 1000:
            self.flush()
    
    def flush(self):
        """Write buffered records to Parquet file."""
        if not self.buffer:
            return
        
        # Enforce schema to prevent inconsistencies
        if self.schema is None:
            temp_table = pa.Table.from_pylist(self.buffer)
            self.schema = temp_table.schema
        
        # Apply enforced schema to ensure consistency
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        
        # Initialize writer on first flush
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="snappy"
            )
        
        self.writer.write_table(table)
        logger.debug(f"Flushed {len(self.buffer)} records to {self.path}")
        self.buffer = []
    
    def close(self):
        """Close the Parquet writer and finalize the file."""
        if self.writer is not None:
            self.writer.close()
            logger.info(f"Parquet file closed: {self.path}")
            self.writer = None
