"""Health check endpoint (liveness for Docker healthchecks and monitoring)."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "agentpick-recommendation-api"}
