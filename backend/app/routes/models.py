"""
Models endpoint (/v1/models).

Advertises the recommender to Open WebUI as a selectable model.
"""

import time

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models():
    """List available models (OpenAI format)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agentpick-recommender",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agentpick",
                "name": "AgentPick Recommender",
                "description": "Hugging Face model recommendations from natural-language intent",
            }
        ],
    }
