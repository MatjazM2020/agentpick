"""Configuration and environment settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Default paths
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DEFAULT_EMBEDDINGS_FILE = DEFAULT_DATA_DIR / "embeddings.parquet"
DEFAULT_PROCESSED_FILE = DEFAULT_DATA_DIR / "processed_models.txt"

# API settings
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Vectorization settings
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS_PER_CHUNK = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64

# Download filtering
MIN_DOWNLOADS_THRESHOLD = 10000

# Vector DB settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "hf_models"

# File size limits (in GB)
PARQUET_MAX_SIZE_GB = 5.0
