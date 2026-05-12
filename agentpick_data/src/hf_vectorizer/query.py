"""Query utilities for Parquet and Qdrant vector databases."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from .config import QDRANT_URL, QDRANT_COLLECTION_NAME, PARQUET_MAX_SIZE_GB

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None


class ParquetQueryEngine:
    """Query interface for Parquet-backed vector database."""
    
    def __init__(self, parquet_path: str = "embeddings.parquet"):
        """Initialize Parquet-based vector database client.
        
        Args:
            parquet_path: Path to embeddings.parquet file
        """
        self.parquet_path = Path(parquet_path)
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        
        # Scalability check: prevent out-of-memory on large files
        file_size = self.parquet_path.stat().st_size / (1024**3)
        if file_size > PARQUET_MAX_SIZE_GB:
            raise RuntimeError(
                f"Parquet file too large ({file_size:.1f}GB) for in-memory loading. "
                "Use Qdrant backend instead: --qdrant-backend"
            )
        
        logger.info(f"Loading embeddings from {self.parquet_path}")
        self.table = pq.read_table(self.parquet_path)
        self.df = self.table.to_pandas()
        
        # Convert vector column from list to numpy arrays
        self.vectors = np.array([np.array(v) for v in self.df['vector']])
        logger.info(f"Loaded {len(self.df)} vectors")
    
    def _safe_get(self, row, col_name, default=None, dtype=None):
        """Safely extract value from pandas Series.
        
        Args:
            row: pandas Series
            col_name: Column name
            default: Default value if not found or NaN
            dtype: Type to convert to ('int', 'str', 'list', or None)
        
        Returns:
            Extracted and converted value
        """
        try:
            if col_name not in row.index:
                return default
            val = row[col_name]
            if pd.isna(val):
                return default
            if dtype == 'int':
                return int(val)
            elif dtype == 'str':
                return str(val)
            elif dtype == 'list':
                if isinstance(val, list):
                    return val
                return default if default is not None else []
            return val
        except (TypeError, ValueError):
            return default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_vectors': len(self.df),
            'vector_dimension': self.vectors.shape[1],
            'unique_models': self.df['model_id'].nunique(),
            'total_sections': len(self.df),
        }
    
    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search for models using natural language.
        
        Args:
            query: Natural language query
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of results with model info and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Compute similarity scores (cosine similarity)
        similarities = np.dot(self.vectors, query_embedding) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit * 3]
        
        # Format results, deduplicating by model_id
        formatted = []
        seen_models = set()
        
        for idx in top_indices:
            score = float(similarities[idx])
            
            if score < score_threshold:
                break
            
            row = self.df.iloc[idx]
            model_id = self._safe_get(row, 'model_id', '', 'str')
            
            if model_id not in seen_models:
                formatted.append({
                    'score': round(score, 4),
                    'model_id': model_id,
                    'section': self._safe_get(row, 'section_header', '', 'str'),
                    'downloads': self._safe_get(row, 'downloads', 0, 'int'),
                    'likes': self._safe_get(row, 'likes', 0, 'int'),
                    'tags': self._safe_get(row, 'tags', [], 'list'),
                })
                seen_models.add(model_id)
            
            if len(formatted) >= limit:
                break
        
        return formatted
    
    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search models by tag.
        
        Args:
            tag: Model tag to filter by
            limit: Number of unique models to return
        
        Returns:
            List of models with this tag
        """
        results = []
        seen_models = set()
        
        for idx, row in self.df.iterrows():
            tags = self._safe_get(row, 'tags', [], 'list')
            model_id = self._safe_get(row, 'model_id', '', 'str')
            
            if model_id not in seen_models and tag in tags:
                results.append({
                    'model_id': model_id,
                    'section': self._safe_get(row, 'section_header', '', 'str'),
                    'downloads': self._safe_get(row, 'downloads', 0, 'int'),
                    'tags': tags,
                })
                seen_models.add(model_id)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Get all information for a specific model."""
        model_rows = self.df[self.df['model_id'] == model_id]
        
        if len(model_rows) == 0:
            return {'model_id': model_id, 'sections': [], 'metadata': {}}
        
        model_data = {
            'model_id': model_id,
            'sections': [],
            'metadata': {},
        }
        
        # Collect sections
        for idx, row in model_rows.iterrows():
            model_data['sections'].append({
                'header': self._safe_get(row, 'section_header', '', 'str'),
                'index': self._safe_get(row, 'section_index', 0, 'int'),
            })
            
            # Store metadata from first section
            if not model_data['metadata']:
                model_data['metadata'] = {
                    'downloads': self._safe_get(row, 'downloads', 0, 'int'),
                    'likes': self._safe_get(row, 'likes', 0, 'int'),
                    'tags': self._safe_get(row, 'tags', [], 'list'),
                    'pipeline_tag': self._safe_get(row, 'pipeline_tag', '', 'str'),
                    'library_name': self._safe_get(row, 'library_name', '', 'str'),
                    'created_at': self._safe_get(row, 'created_at', '', 'str'),
                    'last_modified': self._safe_get(row, 'last_modified', '', 'str'),
                }
        
        return model_data
    
    def get_top_models(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top models by download count."""
        models = {}
        
        for idx, row in self.df.iterrows():
            model_id = self._safe_get(row, 'model_id', '', 'str')
            if model_id not in models:
                models[model_id] = {
                    'model_id': model_id,
                    'downloads': self._safe_get(row, 'downloads', 0, 'int'),
                    'likes': self._safe_get(row, 'likes', 0, 'int'),
                    'tags': self._safe_get(row, 'tags', [], 'list'),
                }
        
        sorted_models = sorted(
            models.values(),
            key=lambda x: x['downloads'],
            reverse=True
        )
        
        return sorted_models[:limit]
    
    def export_collection(self, output_file: str):
        """Export collection to JSONL file."""
        with open(output_file, 'w') as f:
            for idx, row in self.df.iterrows():
                record = {
                    'id': self._safe_get(row, 'id', idx, 'int'),
                    'payload': {
                        'model_id': self._safe_get(row, 'model_id', '', 'str'),
                        'section_header': self._safe_get(row, 'section_header', '', 'str'),
                        'section_index': self._safe_get(row, 'section_index', 0, 'int'),
                        'text': self._safe_get(row, 'text', '', 'str'),
                        'downloads': self._safe_get(row, 'downloads', 0, 'int'),
                        'likes': self._safe_get(row, 'likes', 0, 'int'),
                        'tags': self._safe_get(row, 'tags', [], 'list'),
                        'pipeline_tag': self._safe_get(row, 'pipeline_tag', '', 'str'),
                        'library_name': self._safe_get(row, 'library_name', '', 'str'),
                        'created_at': self._safe_get(row, 'created_at', '', 'str'),
                        'last_modified': self._safe_get(row, 'last_modified', '', 'str'),
                    }
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported {len(self.df)} vectors to {output_file}")
    
    def print_stats(self):
        """Print collection statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("VECTOR DATABASE STATISTICS (Parquet)")
        print("=" * 60)
        print(f"Parquet File: {self.parquet_path}")
        print(f"Total Vectors: {stats['total_vectors']}")
        print(f"Unique Models: {stats['unique_models']}")
        print(f"Vector Dimension: {stats['vector_dimension']}")
        print("=" * 60 + "\n")


class QdrantQueryEngine:
    """Query interface for Qdrant vector database."""
    
    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ):
        """Initialize Qdrant-based vector database client.
        
        Args:
            qdrant_url: URL to Qdrant server
            collection_name: Name of collection in Qdrant
        """
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )
        
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        logger.info(f"Connecting to Qdrant at {qdrant_url}...")
        try:
            self.client = QdrantClient(url=qdrant_url)
            self.client.get_collection(collection_name)
            logger.info(f"Connected to Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {qdrant_url}")
            raise
        
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'total_vectors': collection_info.points_count,
                'vector_dimension': collection_info.config.params.vectors.size,
                'unique_models': None,
                'total_sections': collection_info.points_count,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
    
    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search for models using natural language.
        
        Args:
            query: Natural language query
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of results with model info and scores
        """
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            query_vector = query_embedding.tolist()
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 3,
            )
            
            formatted = []
            seen_models = set()
            
            for scored_point in results:
                score = float(scored_point.score)
                
                if score < score_threshold:
                    break
                
                payload = scored_point.payload
                model_id = payload.get("model_id", "")
                
                if model_id not in seen_models:
                    formatted.append({
                        'score': round(score, 4),
                        'model_id': model_id,
                        'section': payload.get('section_header', ''),
                        'downloads': payload.get('downloads', 0),
                        'likes': payload.get('likes', 0),
                        'tags': payload.get('tags', []) or [],
                    })
                    seen_models.add(model_id)
                
                if len(formatted) >= limit:
                    break
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise
    
    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search models by tag.
        
        Args:
            tag: Model tag to filter by
            limit: Number of unique models to return
        
        Returns:
            List of models with this tag
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )
            
            formatted = []
            seen_models = set()
            
            for point in results[0]:
                payload = point.payload
                model_id = payload.get("model_id", "")
                tags = payload.get("tags", []) or []
                
                if model_id not in seen_models and tag in tags:
                    formatted.append({
                        'model_id': model_id,
                        'section': payload.get('section_header', ''),
                        'downloads': payload.get('downloads', 0),
                        'tags': tags,
                    })
                    seen_models.add(model_id)
                
                if len(formatted) >= limit:
                    break
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error searching by tag: {e}")
            raise
    
    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Get all information for a specific model."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )
            
            model_data = {
                'model_id': model_id,
                'sections': [],
                'metadata': {},
            }
            
            found_metadata = False
            for point in results[0]:
                payload = point.payload
                if payload.get("model_id") == model_id:
                    model_data['sections'].append({
                        'header': payload.get('section_header', ''),
                        'index': payload.get('section_index', 0),
                    })
                    
                    if not found_metadata:
                        model_data['metadata'] = {
                            'downloads': payload.get('downloads', 0),
                            'likes': payload.get('likes', 0),
                            'tags': payload.get('tags', []) or [],
                            'pipeline_tag': payload.get('pipeline_tag', ''),
                            'library_name': payload.get('library_name', ''),
                            'created_at': payload.get('created_at', ''),
                            'last_modified': payload.get('last_modified', ''),
                        }
                        found_metadata = True
            
            return model_data
        
        except Exception as e:
            logger.error(f"Error getting model details: {e}")
            raise
    
    def get_top_models(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top models by download count."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )
            
            models = {}
            for point in results[0]:
                payload = point.payload
                model_id = payload.get("model_id", "")
                if model_id not in models:
                    models[model_id] = {
                        'model_id': model_id,
                        'downloads': payload.get('downloads', 0),
                        'likes': payload.get('likes', 0),
                        'tags': payload.get('tags', []) or [],
                    }
            
            sorted_models = sorted(
                models.values(),
                key=lambda x: x['downloads'],
                reverse=True
            )
            
            return sorted_models[:limit]
        
        except Exception as e:
            logger.error(f"Error getting top models: {e}")
            raise
    
    def export_collection(self, output_file: str):
        """Export collection to JSONL file."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )
            
            with open(output_file, 'w') as f:
                for point in results[0]:
                    record = {
                        'id': point.id,
                        'payload': point.payload
                    }
                    f.write(json.dumps(record) + '\n')
            
            logger.info(f"Exported {len(results[0])} vectors to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            raise
    
    def print_stats(self):
        """Print collection statistics."""
        try:
            stats = self.get_stats()
            collection_info = self.client.get_collection(self.collection_name)
            
            print("\n" + "=" * 60)
            print("VECTOR DATABASE STATISTICS (Qdrant)")
            print("=" * 60)
            print(f"Qdrant URL: {self.qdrant_url}")
            print(f"Collection: {self.collection_name}")
            print(f"Total Vectors: {stats['total_vectors']}")
            print(f"Vector Dimension: {stats['vector_dimension']}")
            print(f"Distance Metric: {collection_info.config.params.vectors.distance}")
            print("=" * 60 + "\n")
        
        except Exception as e:
            logger.error(f"Error printing stats: {e}")
            raise


def main():
    """Command-line entry point for querying."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Query HF models vector database (Parquet or Qdrant backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query Parquet file
  python -m hf_vectorizer.query --backend parquet --parquet-path embeddings.parquet stats
  python -m hf_vectorizer.query --backend parquet search "text generation"
  
  # Query Qdrant
  python -m hf_vectorizer.query --backend qdrant stats
  python -m hf_vectorizer.query --backend qdrant search "text generation"
        """
    )
    
    # Backend selection
    parser.add_argument(
        '--backend',
        choices=['parquet', 'qdrant'],
        default='parquet',
        help='Backend to use (default: parquet)'
    )
    
    # Parquet options
    parser.add_argument(
        '--parquet-path',
        default='data/embeddings.parquet',
        help='Path to embeddings.parquet file (Parquet backend)'
    )
    
    # Qdrant options
    parser.add_argument(
        '--qdrant-url',
        default=QDRANT_URL,
        help='Qdrant server URL (Qdrant backend)'
    )
    
    parser.add_argument(
        '--collection-name',
        default=QDRANT_COLLECTION_NAME,
        help='Qdrant collection name (Qdrant backend)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Stats command
    subparsers.add_parser('stats', help='Show collection statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Semantic search')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Max results')
    search_parser.add_argument('--threshold', type=float, default=0.0, help='Min score')
    
    # Tag search
    tag_parser = subparsers.add_parser('tag', help='Search by tag')
    tag_parser.add_argument('tag', help='Tag to filter by')
    tag_parser.add_argument('--limit', type=int, default=10, help='Max results')
    
    # Top models
    top_parser = subparsers.add_parser('top', help='Get top models by downloads')
    top_parser.add_argument('--limit', type=int, default=20, help='Number of models')
    
    # Model details
    details_parser = subparsers.add_parser('details', help='Get model details')
    details_parser.add_argument('model_id', help='Model ID')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export collection')
    export_parser.add_argument('--output', default='hf_models_export.jsonl', help='Output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize backend
    try:
        if args.backend == 'parquet':
            utils = ParquetQueryEngine(parquet_path=args.parquet_path)
        elif args.backend == 'qdrant':
            utils = QdrantQueryEngine(
                qdrant_url=args.qdrant_url,
                collection_name=args.collection_name,
            )
        else:
            print(f"Unknown backend: {args.backend}")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing {args.backend} backend: {e}")
        sys.exit(1)
    
    try:
        if args.command == 'stats':
            utils.print_stats()
        
        elif args.command == 'search':
            results = utils.search_semantic(
                args.query,
                limit=args.limit,
                score_threshold=args.threshold,
            )
            print(f"\nTop results for: '{args.query}'")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['model_id']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Section: {result['section']}")
                print(f"   Downloads: {result['downloads']}, Likes: {result['likes']}")
                print()
        
        elif args.command == 'tag':
            results = utils.search_by_tag(args.tag, limit=args.limit)
            print(f"\nModels with tag '{args.tag}':")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['model_id']}")
                print(f"   Downloads: {result['downloads']}")
                print()
        
        elif args.command == 'top':
            results = utils.get_top_models(limit=args.limit)
            print(f"\nTop {args.limit} models by downloads:")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['model_id']}")
                print(f"   Downloads: {result['downloads']}, Likes: {result['likes']}")
                print()
        
        elif args.command == 'details':
            result = utils.get_model_details(args.model_id)
            print(json.dumps(result, indent=2))
        
        elif args.command == 'export':
            utils.export_collection(args.output)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
