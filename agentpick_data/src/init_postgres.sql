-- PostgreSQL schema for AgentPick metadata and relational retrieval
-- One row per model with aggregated chunk IDs for stable interop with Qdrant

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,

    downloads BIGINT,
    likes INT,

    pipeline_tag TEXT,
    library_name TEXT,

    created_at TIMESTAMP NULL,
    last_modified TIMESTAMP NULL,

    tags TEXT[],

    -- Stable mapping to semantic chunks in Qdrant
    -- Stores all chunk IDs (Qdrant point IDs) belonging to this model
    chunk_ids BIGINT[],

    -- Number of semantic chunks
    num_chunks INT,

    -- Total parameter count from HuggingFace safetensors metadata (null if unavailable)
    parameter_count BIGINT,

    -- Full markdown model card (README.md) from HuggingFace Hub
    model_card TEXT
);

-- Migration for databases created before these columns existed
-- (CREATE TABLE IF NOT EXISTS does not add columns to an existing table)
ALTER TABLE models ADD COLUMN IF NOT EXISTS parameter_count BIGINT;
ALTER TABLE models ADD COLUMN IF NOT EXISTS model_card TEXT;

-- Indexes for common filtering and ranking operations
CREATE INDEX IF NOT EXISTS idx_models_downloads
ON models(downloads DESC);

CREATE INDEX IF NOT EXISTS idx_models_pipeline_tag
ON models(pipeline_tag);

CREATE INDEX IF NOT EXISTS idx_models_likes
ON models(likes DESC);

CREATE INDEX IF NOT EXISTS idx_models_tags
ON models USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_models_parameter_count
ON models(parameter_count);
