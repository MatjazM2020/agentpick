"""
OpenAI-compatible chat completion schemas.

Implements request/response models that match OpenAI's API specification,
allowing Open WebUI to use the recommendation engine as a standard LLM provider.
"""

from typing import List, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="Message role: 'user', 'assistant', 'system'")
    content: Union[str, Any] = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name of the message sender")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    top_p: Optional[float] = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Enable streaming responses")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0, ge=-2, le=2, description="Frequency penalty")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = Field(..., description="Index of this choice")
    message: ChatMessage = Field(..., description="The completion message")
    finish_reason: str = Field(
        ...,
        description="Why generation stopped: 'stop', 'length', 'content_filter'",
    )


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp when created")
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: Optional[dict] = Field(None, description="Token usage statistics")
