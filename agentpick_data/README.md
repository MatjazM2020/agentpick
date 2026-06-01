# AgentPick Data Pipeline

The data layer for **AgentPick**. It downloads HuggingFace model cards, generates
semantic embeddings, and loads them into the two stores the backend queries:

- **Qdrant** — vector embeddings + chunk text for semantic retrieval.
- **PostgreSQL** — structured model metadata for relational filtering and ranking.

A portable **Parquet** file (`data/embeddings.parquet`) is the intermediate
artifact produced by the vectorizer and consumed by both loaders.

```
HuggingFace Hub
      │  download + parse README + embed (BAAI/bge-large-en-v1.5, 1024-dim)
      ▼
data/embeddings.parquet   ← one row per chunk
      ├──────────────► Qdrant      (vectors + chunk text, collection: hf_models)
      └──────────────► PostgreSQL  (one row per model, table: models)
```

---

## Quickstart

End-to-end on a local machine. Run everything from the `agentpick_data/` directory
unless noted. The package isn't pip-installed, so `PYTHONPATH` must point at `src/`.

```bash
# 0. From the repo root, start the databases (Qdrant + PostgreSQL)
docker compose up -d qdrant postgres
#    Qdrant     → localhost:6333  (dashboard at http://localhost:6333/dashboard)
#    PostgreSQL → localhost:5433  (mapped to the container's 5432)

# 1. Set up the Python environment (in agentpick_data/)
cd agentpick_data
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Make the package importable for the whole session
export PYTHONPATH="$PWD/src"

# 3. Generate embeddings (writes data/embeddings.parquet).
#    Start small with --limit while you verify the setup.
python -m hf_vectorizer.vectorizer --data-dir ./data --batch-size 8 --limit 50

# 4. Load vectors into Qdrant (collection: hf_models)
python src/load_embeddings_to_qdrant.py

# 5. Initialize the PostgreSQL schema and load metadata.
#    POSTGRES_PORT=5433 because docker-compose maps the container's 5432 → host 5433.
POSTGRES_PORT=5433 python src/initialize_postgres.py

# 6. Verify with a semantic query against Qdrant
python -m hf_vectorizer.query --backend qdrant search "small instruction-tuned chat model"
```

That's it — Qdrant holds the vectors and PostgreSQL holds the metadata, ready for
the AgentPick backend.

> **Tip:** Already have an `embeddings.parquet`? Skip step 3 and go straight to the
> loaders (steps 4–5).

---

## Prerequisites

- Python 3.8+
- Docker (for Qdrant + PostgreSQL via `docker compose`)
- (Optional) A HuggingFace token for higher API rate limits
- (Optional) SLURM cluster access for large vectorization jobs

## Project Structure

```
agentpick_data/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/                           # Runtime artifacts (gitignored)
│   ├── embeddings.parquet          # Generated embeddings (one row per chunk)
│   └── processed_models.txt        # Resume tracking
├── scripts/
│   ├── run_vectorize.sh            # Local vectorization run
│   ├── vectorize_only.sh           # SLURM: embed already-downloaded models
│   ├── submit_vectorize.sh         # SLURM: full download + vectorize job
│   └── start_qdrant.sh             # Start a standalone Qdrant container
└── src/
    ├── load_embeddings_to_qdrant.py  # Parquet → Qdrant (vectors + payload)
    ├── initialize_postgres.py        # Wait + create schema + load Parquet → Postgres
    ├── load_parquet_to_postgres.py   # Metadata loader (schema must already exist)
    ├── init_postgres.sql             # PostgreSQL `models` table + indexes
    └── hf_vectorizer/                # Vectorization package
        ├── __main__.py
        ├── config.py                 # Defaults + env-var configuration
        ├── vectorizer.py             # Main pipeline (download → parse → embed → write)
        ├── hf_client.py              # HuggingFace Hub client
        ├── embeddings.py             # Sentence-transformer wrapper
        ├── parsing.py                # README parsing + token chunking
        ├── storage.py                # Streaming Parquet writer
        ├── query.py                  # Query CLI (Parquet or Qdrant backend)
        └── utils.py                  # Logging + retry helpers
```

## Configuration

Vectorizer defaults live in `src/hf_vectorizer/config.py`:

```python
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim, cosine distance
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS_PER_CHUNK = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
MIN_DOWNLOADS_THRESHOLD = 1000          # Skip models below this download count
QDRANT_COLLECTION_NAME = "hf_models"
```

Environment variables (read via `.env` for the vectorizer/Qdrant loader, or
exported directly for the PostgreSQL scripts):

| Variable | Default | Used by |
|---|---|---|
| `HF_TOKEN` | _none_ | HuggingFace API (higher rate limits) |
| `DATA_DIR` | `./data` | Vectorizer + Qdrant loader |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant loader + query |
| `POSTGRES_HOST` | `localhost` | PostgreSQL scripts |
| `POSTGRES_PORT` | `5432` | PostgreSQL scripts (use `5433` for docker-compose) |
| `POSTGRES_DB` | `agentpick` | PostgreSQL scripts |
| `POSTGRES_USER` | `agentpick` | PostgreSQL scripts |
| `POSTGRES_PASSWORD` | `agentpick_password` | PostgreSQL scripts |

Optional `.env` (in `agentpick_data/`) for the vectorizer and Qdrant loader:

```bash
cat > .env << 'EOF'
HF_TOKEN=your_huggingface_token_here
DATA_DIR=./data
QDRANT_URL=http://localhost:6333
EOF
```

Get a token at https://huggingface.co/settings/tokens (read access is enough).

## Step 1 — Generate Embeddings

The vectorizer fetches text-generation models (sorted by downloads, stopping below
`MIN_DOWNLOADS_THRESHOLD`), downloads each README, chunks it with overlap, embeds the
chunks, and streams the result to `data/embeddings.parquet`.

### Local run

```bash
export PYTHONPATH="$PWD/src"

# Convenience wrapper (creates data/ and logs/, uses batch size 32)
bash scripts/run_vectorize.sh

# Or call the module directly with custom options
python -m hf_vectorizer.vectorizer \
    --data-dir ./data \
    --embedding-model "BAAI/bge-large-en-v1.5" \
    --batch-size 8 \
    --limit 100
```

**Options:** `--data-dir`, `--embedding-model`, `--batch-size`, `--limit`,
`--hf-token` (or set `HF_TOKEN`).

**Output:** `data/embeddings.parquet` and `data/processed_models.txt` (resume log).

Each Parquet row is one README chunk with these columns:
`id`, `vector`, `model_id`, `section_header`, `section_index`, `chunk_index`,
`num_sections`, `text`, `downloads`, `likes`, `tags`, `pipeline_tag`,
`library_name`, `created_at`, `last_modified`.

### HPC run (SLURM)

```bash
sbatch scripts/submit_vectorize.sh   # full download + vectorize
sbatch scripts/vectorize_only.sh     # embed already-downloaded models
squeue -u "$USER"                    # check status
```

Adjust the `#SBATCH` directives (partition, GPU, time, paths) for your cluster.

## Step 2 — Load Vectors into Qdrant

Start Qdrant if it isn't already running (`docker compose up -d qdrant`, or the
standalone `bash scripts/start_qdrant.sh`), then:

```bash
export PYTHONPATH="$PWD/src"

# Uses DATA_DIR/embeddings.parquet, QDRANT_URL, and collection "hf_models" by default
python src/load_embeddings_to_qdrant.py

# Or pass explicit args: <parquet_path> <qdrant_url> <collection_name>
python src/load_embeddings_to_qdrant.py data/embeddings.parquet http://localhost:6333 hf_models
```

The collection is created automatically (cosine distance, dimension inferred from
the data). The Parquet `id` column becomes the Qdrant point ID; every other column
is stored in the point payload.

## Step 3 — Load Metadata into PostgreSQL

`initialize_postgres.py` is the one-shot setup: it waits for PostgreSQL, applies the
schema from `init_postgres.sql` (the `models` table + indexes), then aggregates the
Parquet by `model_id` and upserts one row per model (with the chunk IDs that link
back to Qdrant points).

```bash
# docker-compose exposes PostgreSQL on host port 5433
POSTGRES_PORT=5433 python src/initialize_postgres.py

# Custom Parquet path
POSTGRES_PORT=5433 python src/initialize_postgres.py --parquet-path data/embeddings.parquet
```

`load_parquet_to_postgres.py` is a metadata-only loader for when the schema already
exists; otherwise prefer `initialize_postgres.py`.

The `models` table stores: `model_id` (PK), `downloads`, `likes`, `pipeline_tag`,
`library_name`, `created_at`, `last_modified`, `tags[]`, `chunk_ids[]`, `num_chunks`.

## Step 4 — Query

The query CLI works against either the Parquet file or Qdrant.

```bash
export PYTHONPATH="$PWD/src"

# Qdrant backend
python -m hf_vectorizer.query --backend qdrant stats
python -m hf_vectorizer.query --backend qdrant search "multilingual summarization model" --limit 5

# Parquet backend (no database needed; loads the file into memory)
python -m hf_vectorizer.query --backend parquet --parquet-path data/embeddings.parquet stats
python -m hf_vectorizer.query --backend parquet --parquet-path data/embeddings.parquet search "code generation"
```

**Commands:** `stats`, `search <query>`, `tag <tag>`, `top`, `details <model_id>`,
`export`.

Inspect the Parquet directly with pandas:

```python
import pandas as pd

df = pd.read_parquet("data/embeddings.parquet")
print(f"Chunks: {len(df)}  |  Models: {df['model_id'].nunique()}  |  Dim: {len(df['vector'].iloc[0])}")
```

## Resume & Checkpointing

The vectorizer tracks processed models in `data/processed_models.txt`:

- Interrupted runs resume automatically, skipping already-processed models.
- New chunks are appended; point IDs continue from the current max.
- Both loaders use upserts, so re-running them is safe.

```bash
rm data/processed_models.txt   # re-process models (keeps existing Parquet)
rm -rf data/                   # start completely fresh
```

## Troubleshooting

**`No module named 'hf_vectorizer'`** — activate the venv and export the path:
```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src"   # from the agentpick_data/ directory
```

**PostgreSQL connection refused / timeout** — confirm the container is up and use the
mapped host port:
```bash
docker compose ps postgres
POSTGRES_PORT=5433 python src/initialize_postgres.py
```

**Qdrant connection failed** — verify it's running:
```bash
docker compose ps qdrant       # or: docker ps | grep qdrant
docker compose up -d qdrant
```

**Out of memory during embedding** — reduce the batch size:
```bash
python -m hf_vectorizer.vectorizer --batch-size 8
```

**`CUDA out of memory`** — force CPU or shrink the batch:
```bash
export CUDA_VISIBLE_DEVICES=""
python -m hf_vectorizer.vectorizer --batch-size 8
```

**HuggingFace rate limiting** — add a token:
```bash
echo "HF_TOKEN=hf_your_token_here" >> .env
```

## Performance Notes

- **Embedding model:** `BAAI/bge-large-en-v1.5` → 1024 dimensions, cosine distance.
- **Throughput:** ~0.5 models/sec on CPU; ~2–3× faster with a CUDA GPU.
- **Parquet size:** ~25–50 MB per 1,000 models (depends on README length).
- **Memory:** 8–16 GB RAM for typical batch sizes.

## References

- [HuggingFace Hub](https://huggingface.co/)
- [Sentence Transformers](https://www.sbert.net/)
- [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Qdrant Vector Database](https://qdrant.tech/)
- [PostgreSQL](https://www.postgresql.org/)
- [Apache Parquet](https://parquet.apache.org/)
