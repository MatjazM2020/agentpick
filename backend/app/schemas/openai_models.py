"""
OpenAI-compatible models list schema.

Defines the response format for /v1/models endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Model(BaseModel):
    """Represents a single available model."""
    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type (always 'model')")
    created: int = Field(..., description="Unix timestamp when model was released")
    owned_by: str = Field(..., description="Organization owning this model")
    permission: List[dict] = Field(default_factory=list, description="Permission details")
    root: Optional[str] = Field(None, description="Root model if this is a fine-tune")
    parent: Optional[str] = Field(None, description="Parent model if this is a fine-tune")


class ModelsListResponse(BaseModel):
    """Response listing available models."""
    object: str = Field("list", description="Object type")
    data: List[Model] = Field(..., description="List of available models")
