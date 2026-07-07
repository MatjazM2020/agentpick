"""FastAPI application factory (OpenAI-compatible recommendation API)."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Before the route imports below: src.core.config reads env at import time.
load_dotenv()

from app.routes import chat, health, models  # noqa: E402
from src.core.agent_activity_log import configure_activity_log, log_activity  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Serve immediately; warm slow resources (embedder, LLM client) in background."""
    log_activity(f"=== AgentPick started | log={configure_activity_log()} ===")

    async def _warmup() -> None:
        try:
            from src.agent import get_client
            from src.core.llm import warmup

            await asyncio.to_thread(warmup)
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


def create_app() -> FastAPI:
    """Build the FastAPI app: CORS plus the health, models, and chat routes."""
    app = FastAPI(
        title="AgentPick Recommendation API",
        description="OpenAI-compatible API for model recommendations",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Open WebUI runs on a different origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    return app


app = create_app()
