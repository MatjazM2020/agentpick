# HuggingFace Model Vectorizer

A Python tool for downloading, parsing, and generating embeddings for HuggingFace model cards using semantic embeddings. This project vectorizes model metadata and README content, storing results in Parquet format or syncing to a Qdrant vector database.

## Features

- **Download model metadata** from HuggingFace Hub with filtering by popularity (downloads threshold)
- **Parse README content** using chunking with overlap for better semantic representation
- **Generate embeddings** using state-of-the-art sentence transformers (BAAI/bge-large-en-v1.5 by default)
- **Store embeddings** in Parquet format for local analysis
- **Resume capability** - tracks processed models and resumes interrupted vectorization
- **Flexible deployment**:
  - Local execution via bash script
  - HPC cluster submission via SLURM
  - Docker integration with Qdrant vector database

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Data directory (created at runtime)
│   ├── embeddings.parquet      # Generated embeddings
│   └── processed_models.txt    # Resume tracking
├── scripts/
│   ├── run_vectorize.sh        # Local execution script
│   ├── start_qdrant.sh         # Start Qdrant Docker container
│   └── submit_vectorize.sh     # SLURM HPC job submission
└── src/hf_vectorizer/          # Main package
    ├── __init__.py
    ├── __main__.py
    ├── config.py               # Configuration and defaults
    ├── vectorizer.py           # Main vectorization pipeline
    ├── hf_client.py            # HuggingFace API client
    ├── embeddings.py           # Embedding model wrapper
    ├── parsing.py              # README parser and chunker
    ├── storage.py              # Parquet storage manager
    ├── query.py                # Query vector database
    └── utils.py                # Utility functions
```

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for Qdrant setup
- (Optional) SLURM cluster access (for HPC submission)

### 1. Clone the Repository

```bash
cd /path/to/agentpick_data
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- `sentence-transformers` - Semantic embedding models
- `transformers` - HuggingFace model architecture
- `torch` - Deep learning framework
- `qdrant-client` - Vector database client (optional)
- `pandas` / `pyarrow` - Data processing and Parquet format
- `huggingface-hub` - HuggingFace API client

### 4. (Optional) Setup HuggingFace Token

For higher rate limits on HuggingFace API, create a `.env` file:

```bash
# Create .env file
cat > .env << EOF
HF_TOKEN=your_huggingface_token_here
DATA_DIR=./data
EOF
```

**To get your HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy and paste into `.env`

## Configuration

Edit `src/hf_vectorizer/config.py` to customize settings:

```python
# Embedding model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Processing parameters
DEFAULT_BATCH_SIZE = 32              # Embeddings per batch
DEFAULT_MAX_TOKENS_PER_CHUNK = 512   # Tokens per README chunk
DEFAULT_CHUNK_OVERLAP_TOKENS = 64    # Overlap between chunks

# Filtering
MIN_DOWNLOADS_THRESHOLD = 1000       # Only process models with >1000 downloads

# Vector database (Qdrant)
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "hf_models"
```

## Running Vectorization

### Option 1: Local Execution (Recommended for Small Runs)

This runs on your local machine using all available CPU/GPU resources.

```bash
# Activate virtual environment first
source venv/bin/activate

# Run vectorization with default settings
bash scripts/run_vectorize.sh

# Or run directly with custom options
python -m hf_vectorizer.vectorizer \
    --data-dir ./data \
    --embedding-model "BAAI/bge-large-en-v1.5" \
    --batch-size 32
```

**Command-line options:**
- `--data-dir PATH` - Directory for storing embeddings and metadata
- `--embedding-model MODEL_ID` - HuggingFace model ID for embeddings
- `--batch-size SIZE` - Batch size for embedding generation (default: 32)
- `--hf-token TOKEN` - HuggingFace API token (or use `.env` file)

**Output:**
- `data/embeddings.parquet` - Parquet file with embeddings and metadata
- `data/processed_models.txt` - List of processed model IDs for resuming

### Option 2: HPC Cluster (SLURM)

For processing large numbers of models on a compute cluster:

```bash
# Submit job to SLURM queue
sbatch scripts/submit_vectorize.sh

# Check job status
squeue -u $USER

# View job output (while running or after completion)
tail -f logs/vectorize_<job_id>.log
```

**SLURM Configuration (in `scripts/submit_vectorize.sh`):**
- Time limit: 12 hours
- CPUs: 8 per task
- Memory: 32 GB
- GPU: 1x A100 (optional, for faster embeddings)
- Partition: `frida` (customize for your cluster)

Modify the `#SBATCH` directives to match your cluster's requirements.

### Option 3: Interactive Python

For debugging or running specific models:

```bash
source venv/bin/activate
python

from hf_vectorizer.vectorizer import HFModelVectorizer

vectorizer = HFModelVectorizer(
    data_dir="./data",
    embedding_model="BAAI/bge-large-en-v1.5",
    batch_size=32
)

# Process a single model
vectorizer.process_model("meta-llama/Llama-2-7b", point_id_counter=0)
```

## Storage Options

### Parquet (Local Storage - Default)

Embeddings are stored in `data/embeddings.parquet` (Apache Parquet format).

**Advantages:**
- No external database needed
- Easy to read with pandas/polars
- Efficient columnar compression

**Access embeddings:**
```python
import pandas as pd
import pyarrow.parquet as pq

# Read all embeddings
df = pd.read_parquet("data/embeddings.parquet")
print(df.columns)  # Model ID, embedding, content, metadata...

# Read specific columns
table = pq.read_table("data/embeddings.parquet", columns=["model_id", "embedding"])
```

### Qdrant Vector Database (Optional)

For semantic search across embeddings, sync to a Qdrant instance:

#### Step 1: Start Qdrant Docker Container

```bash
# Start Qdrant locally (requires Docker)
bash scripts/start_qdrant.sh

# Qdrant will be available at http://localhost:6333
# Web UI: http://localhost:6333/dashboard
```

#### Step 2: Configure Qdrant URL (Optional)

Default configuration connects to localhost. For remote Qdrant, update `.env`:

```bash
QDRANT_URL=http://your-qdrant-server:6333
```

#### Step 3: Run Vectorization

Vectorization automatically syncs to Qdrant if `QDRANT_URL` is configured.

## Resume and Checkpointing

The vectorizer automatically tracks processed models in `data/processed_models.txt`:

- If interrupted, subsequent runs skip already-processed models
- No data loss - new chunks are appended to `embeddings.parquet`
- Safe to run multiple times on the same data directory

**To clear and restart:**
```bash
# Remove progress tracking (keeps embeddings)
rm data/processed_models.txt

# Remove all data (start fresh)
rm -rf data/
```

## Example Workflow

### Complete vectorization pipeline:

```bash
# 1. Setup environment
source venv/bin/activate

# 2. (Optional) Start Qdrant if using vector search
bash scripts/start_qdrant.sh  # Run in separate terminal

# 3. Run vectorization locally
bash scripts/run_vectorize.sh

# Expected output:
# Starting HF Model Vectorizer...
# 
# Downloading and parsing HF models...
# Models processed: 100%|████████| 5000/5000 [02:45:30<00:00, 0.51 it/s]
# 
# Vectorization complete!
# Embeddings saved to: data/embeddings.parquet
```

### Query embeddings:

```python
import pandas as pd

# Load embeddings
df = pd.read_parquet("data/embeddings.parquet")

# Find embedding statistics
print(f"Total chunks: {len(df)}")
print(f"Unique models: {df['model_id'].nunique()}")
print(f"Embedding dimensions: {len(df['embedding'][0])}")

# Filter by model type
llm_models = df[df['tags'].str.contains("llama|gpt", na=False)]
print(f"LLM models: {len(llm_models)}")
```

## Troubleshooting

### Issue: "No module named 'hf_vectorizer'"

**Solution:** Ensure virtual environment is activated and working directory is correct:
```bash
source venv/bin/activate
cd /path/to/agentpick_data
```

### Issue: Out of memory during embedding generation

**Solution:** Reduce batch size:
```bash
python -m hf_vectorizer.vectorizer \
    --data-dir ./data \
    --batch-size 8  # Smaller batch
```

### Issue: HuggingFace rate limiting

**Solution:** Add API token to `.env` for higher rate limits:
```bash
echo "HF_TOKEN=hf_your_token_here" >> .env
```

### Issue: Qdrant connection failed

**Solution:** Check if Qdrant is running:
```bash
# Check if Docker container is running
docker ps | grep qdrant

# If not, start it
bash scripts/start_qdrant.sh

# Or disable Qdrant by commenting QDRANT_URL in .env
```

### Issue: "CUDA out of memory" (GPU usage)

**Solution:** Use CPU instead or reduce batch size:
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python -m hf_vectorizer.vectorizer --batch-size 8
```

## Performance Notes

- **Embedding generation:** ~0.5 models/second on modern hardware
- **Parquet size:** ~25-50 MB per 1,000 models (depending on README length)
- **Memory usage:** 8-16 GB RAM for 32-batch embeddings
- **GPU acceleration:** ~2-3x faster with CUDA-capable GPU

**For 100,000 models:** ~24-48 hours on single machine, ~2-4 hours on HPC cluster with GPU

## Contributing

To extend this project:

1. **Custom embedding models:** Modify `config.py` and `embeddings.py`
2. **Different parsing strategies:** Edit `parsing.py` for alternative chunking
3. **Additional storage backends:** Extend `storage.py` (e.g., PostgreSQL, Elasticsearch)
4. **Query improvements:** Enhance `query.py` for better search

## License

[Add your license here]

## References

- [HuggingFace Hub](https://huggingface.co/)
- [Sentence Transformers](https://www.sbert.net/)
- [BAAI bge-large-en embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Qdrant Vector Database](https://qdrant.tech/)
- [Apache Parquet](https://parquet.apache.org/)
