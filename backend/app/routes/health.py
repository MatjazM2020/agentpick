"""
Health check endpoint.

Provides basic liveness check for deployment monitoring.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information
    """
    return {
        "status": "ok",
        "service": "agentpick-recommendation-api"
    }
