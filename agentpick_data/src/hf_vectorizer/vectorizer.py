"""Main vectorization pipeline."""

import os
import sys
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq

from .hf_client import HFClient
from .embeddings import EmbeddingModel
from .parsing import ReadmeParser
from .storage import ParquetWriter, normalize_record
from .config import (
    DEFAULT_DATA_DIR, DEFAULT_EMBEDDINGS_FILE, DEFAULT_PROCESSED_FILE,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS_PER_CHUNK, DEFAULT_CHUNK_OVERLAP_TOKENS,
    MIN_DOWNLOADS_THRESHOLD
)
from .utils import setup_logging

logger = logging.getLogger(__name__)


class HFModelVectorizer:
    """Main class for downloading and vectorizing Hugging Face models."""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hf_token: Optional[str] = None,
    ):
        """Initialize the vectorizer.
        
        Args:
            data_dir: Directory to store downloaded model cards and embeddings
            embedding_model: HuggingFace model ID for embeddings
            batch_size: Batch size for embedding generation
            hf_token: HuggingFace API token
        """
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir}")
        
        # Initialize components
        self.hf_client = HFClient(token=hf_token)
        self.embedding_model = EmbeddingModel(embedding_model, batch_size)
        self.parser = ReadmeParser(embedding_model)
        
        # Initialize Parquet writer
        embeddings_path = self.data_dir / "embeddings.parquet"
        if embeddings_path.exists():
            logger.info(f"Existing embeddings file found: {embeddings_path}")
            logger.info("Resuming without deleting existing data")
        
        self.writer = ParquetWriter(embeddings_path)
        logger.info(f"Embeddings will be written to {embeddings_path}")
        
        # Resume capability: track processed models
        self.processed_file = self.data_dir / "processed_models.txt"
        self.processed = set()
        if self.processed_file.exists():
            self.processed = set(self.processed_file.read_text().splitlines())
            logger.info(f"Resuming: {len(self.processed)} models already processed")
    
    def process_model(self, model_id: str, point_id_counter: int) -> int:
        """Process a single model: download, parse, embed, and store.
        
        Filters by downloads BEFORE downloading README to avoid wasted I/O.
        
        Args:
            model_id: Model ID
            point_id_counter: Current point ID counter for embeddings
            
        Returns:
            Updated point ID counter
        """
        try:
            # Skip if already processed
            if model_id in self.processed:
                logger.debug(f"Skipping {model_id}: already processed")
                return point_id_counter
            
            # Get model info BEFORE downloading README (filter early)
            model_info = self.hf_client.get_model_info(model_id)
            
            # Distinguish metadata failure from real 0 downloads
            if model_info is None:
                logger.warning(f"Skipping {model_id}: metadata fetch failed")
                return point_id_counter
            
            # Skip if downloads < threshold
            downloads = model_info.get('downloads', 0)
            if downloads < MIN_DOWNLOADS_THRESHOLD:
                logger.debug(f"Skipping {model_id}: only {downloads} downloads")
                return point_id_counter
            
            # Download README
            model_dir = self.data_dir / model_id.replace("/", "_")
            readme_path = self.hf_client.download_readme(model_id, model_dir)
            if not readme_path or not readme_path.exists():
                logger.debug(f"Skipping {model_id}: no README")
                return point_id_counter
            
            # Parse sections
            sections = self.parser.parse_readme(readme_path)
            if not sections:
                logger.debug(f"Skipping {model_id}: no sections found")
                return point_id_counter
            
            # Prepare texts using token-based chunking with overlap
            texts = []
            chunk_metadata = []
            
            for i, section in enumerate(sections):
                chunks = self.parser.chunk_text_tokens(
                    section['content'],
                    max_tokens=DEFAULT_MAX_TOKENS_PER_CHUNK,
                    overlap=DEFAULT_CHUNK_OVERLAP_TOKENS
                )
                
                for j, chunk in enumerate(chunks):
                    texts.append(chunk)
                    chunk_metadata.append({
                        'section_header': section['header'],
                        'section_index': i,
                        'chunk_index': j,
                        'num_chunks_in_section': len(chunks),
                    })
            
            # Generate embeddings with per-batch encoding to prevent total failure
            embeddings_list = []
            for i in range(0, len(texts), self.embedding_model.batch_size):
                batch = texts[i:i + self.embedding_model.batch_size]
                try:
                    emb = self.embedding_model.embed_texts(batch)
                    embeddings_list.append(emb)
                except Exception as e:
                    logger.error(f"Batch failed for {model_id}, skipping batch: {e}")
                    continue
            
            if not embeddings_list:
                logger.warning(f"No embeddings generated for {model_id}, skipping model")
                return point_id_counter
            
            embeddings = np.vstack(embeddings_list)
            
            # Prepare records for Parquet
            records = []
            for i, (embedding, meta) in enumerate(zip(embeddings, chunk_metadata)):
                point_id = point_id_counter + i
                
                record = normalize_record({
                    'id': point_id,
                    'vector': embedding.tolist(),
                    'model_id': model_id,
                    'section_header': meta['section_header'],
                    'section_index': meta['section_index'],
                    'chunk_index': meta['chunk_index'],
                    'num_sections': len(sections),
                    'text': texts[i],
                    **model_info,
                })
                records.append(record)
            
            # Write to Parquet
            self.writer.add(records)
            
            # Flush to ensure durability before marking as processed
            self.writer.flush()
            
            # Mark as processed with atomic fsync
            self.processed.add(model_id)
            with open(self.processed_file, "a") as f:
                f.write(model_id + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            logger.info(f"Processed {model_id}: {len(sections)} sections")
            return point_id_counter + len(records)
            
        except Exception as e:
            logger.error(f"Error processing {model_id}: {e}")
            traceback.print_exc()
            return point_id_counter
    
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run the full vectorization pipeline.
        
        Args:
            limit: Maximum number of models to process (None for all)
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        
        try:
            # Fetch models
            models = self.hf_client.fetch_all_models(limit=limit)
            
            # Resume capability: find max ID to continue from
            embeddings_path = self.data_dir / "embeddings.parquet"
            point_id = 0
            if embeddings_path.exists():
                pfile = pq.ParquetFile(embeddings_path)
                if pfile.metadata.num_rows > 0:
                    max_id = 0
                    for batch in pfile.iter_batches(columns=['id']):
                        col = batch.column(0).to_pylist()
                        if col:
                            max_id = max(max_id, max(col))
                    point_id = max_id + 1
                    logger.info(f"Resuming from point_id {point_id}")
            
            processed_count = 0
            skipped_count = 0
            
            with tqdm(total=len(models), desc="Processing models") as pbar:
                for model_id in models:
                    time.sleep(0.05)  # Adaptive delay to avoid API throttling
                    
                    old_point_id = point_id
                    point_id = self.process_model(model_id, point_id)
                    
                    if point_id > old_point_id:
                        processed_count += 1
                    else:
                        skipped_count += 1
                    
                    pbar.update(1)
            
            # Final flush and close
            try:
                self.writer.flush()
                self.writer.close()
            finally:
                if self.writer is not None:
                    self.writer.close()
            
            elapsed = time.time() - start_time
            
            # Get final stats
            if embeddings_path.exists():
                pfile = pq.ParquetFile(embeddings_path)
                total_vectors = pfile.metadata.num_rows
            else:
                total_vectors = 0
            
            logger.info("=" * 60)
            logger.info("VECTORIZATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total models fetched: {len(models)}")
            logger.info(f"Models successfully processed: {processed_count}")
            logger.info(f"Models skipped: {skipped_count}")
            logger.info(f"Total vectors stored: {total_vectors}")
            logger.info(f"Embedding dimension: {self.embedding_model.embedding_dim}")
            logger.info(f"Time taken: {elapsed:.2f}s ({elapsed/60:.2f} min)")
            logger.info(f"Output file: {embeddings_path}")
            
            return {
                'models_processed': len(models),
                'vectors_stored': total_vectors,
                'elapsed_seconds': elapsed,
                'output_file': str(embeddings_path),
            }
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            traceback.print_exc()
            sys.exit(1)


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and vectorize Hugging Face text-generation models"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to store downloaded model cards and embeddings",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of models to process",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    vectorizer = HFModelVectorizer(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
    )
    
    result = vectorizer.run(limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
