"""
Models endpoint (/v1/models).

Advertises available recommendation models to Open WebUI.
"""

import time

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1", tags=["models"])

# Model registry — advertised to Open WebUI via GET /v1/models
AVAILABLE_MODELS = [
    {
        "id": "agentpick-recommender",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "agentpick",
        "name": "AgentPick Recommender",
        "description": "Hugging Face model recommendations from natural-language intent",
        "permission": [],
        "root": None,
        "parent": None,
    }
]


@router.get("/models")
async def list_models():
    """
    List available models.
    
    Compatible with OpenAI /v1/models endpoint.
    
    Returns:
        List of available models in OpenAI format
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Get details for a specific model.

    Args:
        model_id: Model identifier

    Returns:
        Model details or 404 if not found
    """
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model

    raise HTTPException(
        status_code=404,
        detail={"message": f"Model '{model_id}' not found", "type": "not_found_error"},
    )
