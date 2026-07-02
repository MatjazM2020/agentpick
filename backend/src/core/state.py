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
    """A model with its computed score."""
    model_id: str
    score: float
    semantic_score: float = Field(
        default=0.0,
        description="Normalized Qdrant cosine similarity [0, 1]",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Raw model metadata from Qdrant/PostgreSQL",
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
        description="Original user query (typically latest user turn)"
    )
    conversation_text: str = Field(
        default="",
        description="All user turns joined; used for retrieval and requirements when set",
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
    popularity: dict = Field(
        default_factory=lambda: {"mode": "none"},
        description="Popularity-based DB routing: mode (none/popularity_only/hybrid), "
                    "sort_by, min_downloads, min_likes",
    )
    intent_summary: Optional[str] = Field(
        default=None,
        description="LLM-synthesized retrieval text: primary intent plus all constraints",
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
    
    recommendation_top_k: int = Field(
        default=3,
        description="How many models to return this turn (1 for single-pick, 3 default)",
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
    response_intro: Optional[str] = Field(
        default=None,
        description='One-line opener, e.g. "For your task (…), the best options are:"',
    )
    model_summaries: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-model summary: reasons (model_id -> dict)",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Clarifying questions for interactive refinement"
    )
    requirements_confidence: Optional[float] = Field(
        default=None,
        description="0–1 from requirements analyst; used for underspecified-query gating",
    )
    refinement_assistant_text: Optional[str] = Field(
        default=None,
        description="Full assistant message when pipeline stops for clarification (no top-3 yet)",
    )
    stopped_for_query_refinement: bool = Field(
        default=False,
        description="True when returning only clarification questions (no ranked top-3)",
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
        description="Agent decision logs for observability",
    )

    # Microsoft Agent Framework session (multi-turn LLM context)
    agent_session_id: Optional[str] = Field(
        default=None,
        description="AgentSession id for Requirements Analyst / Ranker",
    )
    agent_session_data: Optional[dict] = Field(
        default=None,
        description="Serialized AgentSession (session.to_dict()) for conversation continuity",
    )
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def natural_language_context_for_requirements(self) -> str:
        """User-side text for requirements and refinement (no assistant turns)."""
        ct = (self.conversation_text or "").strip()
        if ct:
            return ct
        parts = [m.content.strip() for m in self.messages if m.role == "user" and m.content]
        if parts:
            return "\n\n".join(parts)
        return (self.user_query or "").strip()

    def effective_search_query(self) -> str:
        """Text used for embedding / Qdrant search."""
        parts: list[str] = []
        nl = self.natural_language_context_for_requirements()
        if nl:
            parts.append(nl)

        summary = (self.intent_summary or "").strip()
        if summary and summary.lower() not in (nl or "").lower():
            parts.append(summary)

        tt = (self.task_type or "").strip()
        if tt and tt.lower() != "general":
            parts.append(f"Task: {tt}")

        for key in ("domain", "use_case", "language", "hardware", "license"):
            val = self.constraints.get(key)
            if val is not None and str(val).strip():
                label = f"{key}: {val}"
                blob = " ".join(parts).lower()
                if str(val).lower() not in blob:
                    parts.append(label)

        for key in ("model_size", "speed_vs_accuracy"):
            val = self.preferences.get(key)
            if val is not None and str(val).strip():
                label = f"{key}: {val}"
                if str(val).lower() not in " ".join(parts).lower():
                    parts.append(label)

        pop_pref = self.preferences.get("popularity")
        if pop_pref is True or str(pop_pref).lower() == "true":
            if "popular" not in " ".join(parts).lower():
                parts.append("popular well-known models")

        if parts:
            return " | ".join(parts)
        return (self.user_query or "").strip()
