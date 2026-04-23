"""
Configuration for the recommendation system.

Defines:
- Scoring weights and thresholds
- Agent iteration limits
- API endpoints
- Model retrieval parameters
- System behavior tuning
"""

from dataclasses import dataclass


@dataclass
class ScoringConfig:
    """Weights and parameters for the deterministic scoring function."""
    
    # Scoring weights (must sum to 1.0)
    w_semantic_similarity: float = 0.35
    w_popularity: float = 0.20
    w_recency: float = 0.15
    w_hardware_fit: float = 0.15
    w_license_match: float = 0.10
    w_benchmark_score: float = 0.05
    
    # Scoring component thresholds
    min_similarity_score: float = 0.3
    max_age_days: int = 730  # 2 years
    
    # Popularity normalization
    max_downloads_cap: int = 10_000_000
    max_likes_cap: int = 10_000
    
    # Hardware fit scoring
    cpu_only_penalty: float = 0.8
    small_model_bonus: float = 1.1
    large_model_penalty: float = 0.9
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = (
            self.w_semantic_similarity +
            self.w_popularity +
            self.w_recency +
            self.w_hardware_fit +
            self.w_license_match +
            self.w_benchmark_score
        )
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total}"
            )


@dataclass
class RetrieverConfig:
    """Configuration for Qdrant retrieval."""
    
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "hf_models"
    
    # Retrieval parameters
    top_k_chunks: int = 20  # Retrieve this many chunks
    top_k_models: int = 10  # Return this many deduplicated models
    min_similarity_threshold: float = 0.3
    
    # Metadata filtering
    enabled_tags_filter: list[str] = None  # If set, only include these tags
    excluded_tags_filter: list[str] = None
    enabled_licenses: list[str] = None


@dataclass
class AgentConfig:
    """Configuration for agent behavior and limits."""
    
    # Iteration limits (for bounded autonomy)
    max_iterations: int = 3
    max_analyst_retries: int = 2
    max_synthesizer_retries: int = 2
    
    # Agent timeout (seconds)
    agent_timeout_seconds: int = 30
    
    # Requirements extraction
    require_all_constraints: bool = False  # If False, extract what's possible
    
    # Supervisor behavior
    quality_threshold: float = 0.5  # Confidence threshold for stopping
    auto_refine_on_low_confidence: bool = True


@dataclass
class APIConfig:
    """Flask API configuration."""
    
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # CORS
    enable_cors: bool = True
    cors_origins: list[str] = None  # None = allow all
    
    # Request/Response
    max_query_length: int = 2000
    request_timeout_seconds: int = 60


@dataclass
class SystemConfig:
    """Overall system configuration."""
    
    scoring: ScoringConfig
    retriever: RetrieverConfig
    agent: AgentConfig
    api: APIConfig
    
    # Logging
    log_level: str = "INFO"
    log_agent_decisions: bool = True
    
    @classmethod
    def default(cls) -> "SystemConfig":
        """Return default configuration."""
        return cls(
            scoring=ScoringConfig(),
            retriever=RetrieverConfig(),
            agent=AgentConfig(),
            api=APIConfig(),
        )
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.agent.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.retriever.top_k_models < 1:
            raise ValueError("top_k_models must be >= 1")
        if self.retriever.top_k_chunks < self.retriever.top_k_models:
            raise ValueError("top_k_chunks must be >= top_k_models")


# Global configuration instance
config = SystemConfig.default()
