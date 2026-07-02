"""
Configuration for the recommendation system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _default_qdrant_url() -> str:
    """Qdrant HTTP URL from env (Docker: qdrant:6333; local dev: localhost)."""
    explicit = (os.getenv("QDRANT_URL") or "").strip()
    if explicit:
        return explicit
    host = (os.getenv("QDRANT_HOST") or "localhost").strip() or "localhost"
    port = (os.getenv("QDRANT_PORT") or "6333").strip() or "6333"
    return f"http://{host}:{port}"


def _default_qdrant_collection_name() -> str:
    return (os.getenv("QDRANT_COLLECTION_NAME") or "hf_models").strip() or "hf_models"


def _default_qdrant_query_using() -> Optional[str]:
    """Named vector for ``query_points(..., using=...)``; unset uses the collection default vector."""
    u = (os.getenv("QDRANT_QUERY_USING") or "").strip()
    return u or None


@dataclass
class RankerConfig:
    """Configuration for the LLM ranker (staged hard-filter -> re-rank -> explain)."""

    candidate_pool_size: int = 30
    top_k: int = 3
    max_retries: int = 2

    # Deterministic composite weights (must sum to 1.0)
    w_task_match: float = 0.40
    w_objective_evidence: float = 0.50
    w_community_signal: float = 0.10

    # Community signal normalization (downloads/likes from PostgreSQL)
    max_downloads_cap: int = 10_000_000
    max_likes_cap: int = 10_000

    def __post_init__(self) -> None:
        total = self.w_task_match + self.w_objective_evidence + self.w_community_signal
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Ranker weights must sum to 1.0, got {total:.4f} "
                f"(task={self.w_task_match}, objective={self.w_objective_evidence}, "
                f"community={self.w_community_signal})"
            )


@dataclass
class RetrieverConfig:
    """Configuration for Qdrant semantic retrieval."""

    qdrant_url: str = field(default_factory=_default_qdrant_url)
    qdrant_collection_name: str = field(default_factory=_default_qdrant_collection_name)
    qdrant_query_using: Optional[str] = field(default_factory=_default_qdrant_query_using)

    top_k_chunks: int = 90
    top_k_models: int = 30
    min_similarity_threshold: float = 0.0
    apply_qdrant_structured_filter: bool = False


@dataclass
class AgentConfig:
    """Configuration for the orchestrator and recommendation output."""

    recommendation_top_k: int = 3
    orchestrator_max_steps: int = 5


@dataclass
class APIConfig:
    """HTTP API tuning (reserved for future use)."""

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    enable_cors: bool = True
    cors_origins: list[str] | None = None
    max_query_length: int = 2000
    request_timeout_seconds: int = 60


@dataclass
class SystemConfig:
    """Overall system configuration."""

    ranker: RankerConfig
    retriever: RetrieverConfig
    agent: AgentConfig
    api: APIConfig
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> "SystemConfig":
        return cls(
            ranker=RankerConfig(),
            retriever=RetrieverConfig(),
            agent=AgentConfig(),
            api=APIConfig(),
        )

    def validate(self) -> None:
        if self.agent.orchestrator_max_steps < 1:
            raise ValueError("orchestrator_max_steps must be >= 1")
        if self.retriever.top_k_models < 1:
            raise ValueError("top_k_models must be >= 1")
        if self.retriever.top_k_chunks < self.retriever.top_k_models:
            raise ValueError("top_k_chunks must be >= top_k_models")
        if self.ranker.top_k < 1:
            raise ValueError("ranker.top_k must be >= 1")


config = SystemConfig.default()
