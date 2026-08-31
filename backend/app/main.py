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
            from src import catalog
            from src.agent import get_client
            from src.core.llm import warmup

            get_client()
            # Embedder load and catalog connections are independent — warm both at once.
            await asyncio.gather(
                asyncio.to_thread(warmup),
                asyncio.to_thread(catalog.warm),
            )
            # One end-to-end pass per tool path: the first real Qdrant query pays a
            # one-off segment load (~0.7s) and the first SQL scan warms PG buffers —
            # pay those here instead of on the first chat request.
            await asyncio.to_thread(catalog.semantic_search, "general purpose chat assistant", 1)
            await asyncio.to_thread(catalog.filter_models)
            logger.info("Embedding model, OpenAI client, and catalog connections pre-warmed")
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
