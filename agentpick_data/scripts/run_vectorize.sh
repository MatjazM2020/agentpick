#!/bin/bash
# Run Hugging Face model vectorization locally
# 
# This script downloads, parses, and vectorizes Hugging Face model cards.
# Output is stored in data/embeddings.parquet

set -e

echo "Starting HF Model Vectorizer..."
echo ""

# Create data directory
mkdir -p data logs

# Run vectorization
python -m hf_vectorizer.vectorizer \
    --data-dir ./data \
    --embedding-model "BAAI/bge-large-en-v1.5" \
    --batch-size 32 \
    "$@"

echo ""
echo "Vectorization complete!"
