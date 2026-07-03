"""Runtime configuration read from environment variables."""

import os


def qdrant_url() -> str:
    """Qdrant HTTP URL (Docker: qdrant:6333; local dev: localhost:6333)."""
    explicit = (os.getenv("QDRANT_URL") or "").strip()
    if explicit:
        return explicit
    host = (os.getenv("QDRANT_HOST") or "localhost").strip() or "localhost"
    port = (os.getenv("QDRANT_PORT") or "6333").strip() or "6333"
    return f"http://{host}:{port}"


QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION_NAME") or "hf_models").strip() or "hf_models"
QDRANT_QUERY_USING = (os.getenv("QDRANT_QUERY_USING") or "").strip() or None

# Semantic search: how many chunks to pull before de-duplicating to models.
QDRANT_TOP_K_CHUNKS = int(os.getenv("QDRANT_TOP_K_CHUNKS", "60"))

# LLM used for the recommendation agent.
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5.4-nano")

# Bound the agent's tool-calling loop so a single turn stays fast and cheap.
MAX_TOOL_ITERATIONS = int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "8"))
