"""Embedding model handling and operations."""

import logging
from typing import List

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for SentenceTransformer embedding models."""
    
    def __init__(self, model_id: str = "BAAI/bge-large-en-v1.5", batch_size: int = 32):
        """Initialize embedding model.
        
        Args:
            model_id: HuggingFace model ID
            batch_size: Batch size for encoding
        """
        logger.info("Loading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_id, device=device)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Using device: {device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts with CUDA error handling.
        
        Implements retry with smaller batch size and CPU fallback.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of shape (len(texts), embedding_dim) in float32
        """
        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            ).astype(np.float32)
        
        except RuntimeError as e:
            if "CUDA" not in str(e):
                raise
            
            logger.warning(f"CUDA failure, retrying with smaller batch: {e}")
            torch.cuda.empty_cache()
            
            # Retry with smaller batch
            try:
                return self.model.encode(
                    texts,
                    batch_size=max(1, self.batch_size // 2),
                    convert_to_numpy=True,
                ).astype(np.float32)
            
            except RuntimeError as e2:
                logger.error(f"Retry failed, falling back to CPU: {e2}")
                
                # CPU fallback
                self.model = self.model.to("cpu")
                return self.model.encode(
                    texts,
                    batch_size=8,
                    convert_to_numpy=True,
                ).astype(np.float32)
    
    def reset_to_gpu(self):
        """Reset model to GPU if available.
        
        Used to restore GPU operations after CPU fallback.
        """
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("Model reset to GPU")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embeddings in float32
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.astype(np.float32)
