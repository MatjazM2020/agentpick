"""
Recommendation pipeline state management.

This module defines the shared state that flows through all agents.
All agents read from and write to this state object.
No hidden memory inside agents.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScoredModel(BaseModel):
    """A model with its computed score and breakdown."""
    model_id: str
    score: float
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Component scores: similarity, popularity, recency, etc."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Raw model metadata from Qdrant"
    )


class RecommendationState(BaseModel):
    """
    Shared state for the entire recommendation pipeline.
    
    Single source of truth for:
    - User query and conversation history
    - Extracted constraints and preferences
    - Retrieved and ranked models
    - Agent decisions and iteration tracking
    """
    
    # Input & Conversation
    user_query: str = Field(
        description="Original user query"
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="Conversation history (user + assistant messages)"
    )
    
    # Extracted Requirements
    task_type: Optional[str] = Field(
        default=None,
        description="Inferred task (e.g., 'summarization', 'qa', 'code_generation')"
    )
    constraints: dict = Field(
        default_factory=dict,
        description="Structured constraints: latency, memory, license, etc."
    )
    preferences: dict = Field(
        default_factory=dict,
        description="User preferences: speed vs accuracy, hardware, etc."
    )
    
    # Retrieval & Ranking
    retrieved_models: list[dict] = Field(
        default_factory=list,
        description="Raw chunks/sections retrieved from Qdrant"
    )
    scored_models: list[ScoredModel] = Field(
        default_factory=list,
        description="Models with computed scores, sorted descending"
    )
    
    # Final Output
    final_recommendations: list[ScoredModel] = Field(
        default_factory=list,
        description="Top-K models with explanations"
    )
    explanations: dict[str, str] = Field(
        default_factory=dict,
        description="Why each model was recommended (model_id -> explanation)"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Clarifying questions for interactive refinement"
    )
    
    # Pipeline Tracking
    iteration: int = Field(
        default=0,
        description="Current iteration number (for bounded autonomy)"
    )
    requirements_extracted: bool = Field(
        default=False,
        description="Whether requirements analyst has run successfully"
    )
    retrieval_complete: bool = Field(
        default=False,
        description="Whether retrieval has completed"
    )
    evaluation_complete: bool = Field(
        default=False,
        description="Whether evaluation has completed"
    )
    
    # Logging & Diagnostics
    agent_logs: list[str] = Field(
        default_factory=list,
        description="Agent decision logs for observability"
    )
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
