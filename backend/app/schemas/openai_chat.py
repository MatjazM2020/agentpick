"""
OpenAI-compatible chat completion request schema.

Only the fields AgentPick acts on are declared; all other OpenAI parameters
(temperature, top_p, penalties, ...) are accepted and ignored.
"""

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="Message role: 'user', 'assistant', 'system'")
    content: Any = Field(..., description="Message text, or multimodal content parts")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(False, description="Enable streaming responses")
