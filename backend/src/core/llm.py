"""
LLM utilities for embeddings and text processing.

This module provides embedding functions using SentenceTransformer.
Used by the retriever to embed queries and retrieve similar models from Qdrant.
"""

from typing import List
from sentence_transformers import SentenceTransformer


# Model: BAAI/bge-large-en-v1.5 - 1024-dimensional embeddings
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load embedding model on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _embedding_model


def embed(text: str) -> List[float]:
    """
    Embed a text query into a vector.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding (1024 dimensions)
    """
    model = _get_embedding_model()
    embeddings = model.encode([text], convert_to_numpy=False)
    return embeddings[0].tolist()
