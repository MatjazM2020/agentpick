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
    num_chunks INT
);

-- Indexes for common filtering and ranking operations
CREATE INDEX IF NOT EXISTS idx_models_downloads
ON models(downloads DESC);

CREATE INDEX IF NOT EXISTS idx_models_pipeline_tag
ON models(pipeline_tag);

CREATE INDEX IF NOT EXISTS idx_models_likes
ON models(likes DESC);

CREATE INDEX IF NOT EXISTS idx_models_tags
ON models USING GIN(tags);
