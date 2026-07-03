"""
FastAPI application factory.

Creates and configures the FastAPI app with OpenAI-compatible API endpoints.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.core.agent_activity_log import configure_activity_log

# Import route modules
from app.routes import chat, models, health


# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle manager — serve immediately; warm slow resources in background."""
    logger.info("Starting AgentPick Recommendation API")
    load_dotenv()
    log_path = configure_activity_log()
    logger.info(f"Agent activity log: {log_path}")
    from src.core.agent_activity_log import log_activity
    log_activity(f"=== AgentPick started | log={log_path} ===")

    async def _warmup() -> None:
        try:
            from src.core.llm import warmup as _warmup_embeddings
            from src.agent import get_client

            await asyncio.to_thread(_warmup_embeddings)
            get_client()
            logger.info("Embedding model and OpenAI client pre-warmed")
        except Exception as e:
            logger.warning(f"Background warmup failed (first request may be slower): {e}")

    warmup_task = asyncio.create_task(_warmup())

    yield

    warmup_task.cancel()
    try:
        await warmup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down AgentPick Recommendation API")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Sets up:
    - OpenAI-compatible endpoints
    - CORS middleware
    - Route registration
    - Error handlers
    
    Returns:
        Configured FastAPI app
    """
    
    app = FastAPI(
        title="AgentPick Recommendation API",
        description="OpenAI-compatible API for model recommendations",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # === CORS Configuration ===
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow Open WebUI and other clients
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # === Register Routes ===
    # Health check
    app.include_router(health.router)
    
    # Models endpoint
    app.include_router(models.router)
    
    # Chat completions endpoint
    app.include_router(chat.router)
    
    # === Root Endpoint ===
    @app.get("/")
    async def root():
        """API root."""
        return {
            "status": "ok",
            "service": "agentpick-recommendation-api",
            "openai_compatible": True,
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    
    logger.info("FastAPI application created successfully")
    return app


# Create app instance for uvicorn
app = create_app()
