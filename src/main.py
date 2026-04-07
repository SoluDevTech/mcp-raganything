"""Main entry point for the RAGAnything API."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alembic import command
from application.api.health_routes import health_router
from application.api.indexing_routes import indexing_router
from application.api.mcp_tools import mcp
from application.api.query_routes import query_router
from dependencies import app_config, bm25_adapter

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"


def _run_alembic_upgrade() -> None:
    """Run Alembic migrations to head (called via asyncio.to_thread)."""
    alembic_dir = Path(__file__).parent
    cfg = Config(str(alembic_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(alembic_dir / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def db_lifespan(_app: FastAPI):
    """Database migrations and cleanup lifecycle.

    - Runs Alembic migrations on startup
    - Closes BM25 connection pool on shutdown
    """
    logger.info("Application startup initiated")

    try:
        logger.info("Running database migrations...")
        await asyncio.to_thread(_run_alembic_upgrade)
        logger.info("Database migrations completed")
    except Exception:
        logger.exception("Failed to run migrations — refusing to start")
        raise

    yield

    # Cleanup on shutdown
    logger.info("Application shutdown initiated")
    if bm25_adapter is not None:
        try:
            await bm25_adapter.close()
        except Exception:
            logger.exception("Failed to close BM25 adapter")
    logger.info("Application shutdown complete")


# Create FastAPI app with appropriate lifespan
if app_config.MCP_TRANSPORT == "streamable":
    mcp_app = mcp.http_app(path="/")

    @asynccontextmanager
    async def combined_lifespan(app: FastAPI):
        """Combine database lifecycle with MCP lifecycle for streamable transport."""
        async with db_lifespan(app), mcp_app.lifespan(app):
            yield

    app = FastAPI(
        title="RAG Anything API",
        lifespan=combined_lifespan,
    )
    app.mount(MCP_PATH, mcp_app)
else:
    app = FastAPI(
        title="RAG Anything API",
        lifespan=db_lifespan,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REST_PATH = "/api/v1"
app.include_router(indexing_router, prefix=REST_PATH)
app.include_router(health_router, prefix=REST_PATH)
app.include_router(query_router, prefix=REST_PATH)


def run_fastapi():
    """Run FastAPI server with uvicorn."""
    uvicorn.run(
        app,
        host=app_config.HOST,
        port=app_config.PORT,
        log_level=app_config.UVICORN_LOG_LEVEL,
        access_log=False,
        ws="none",
    )


if __name__ == "__main__":
    if app_config.MCP_TRANSPORT == "stdio":
        api_thread = threading.Thread(target=run_fastapi, daemon=True)
        api_thread.start()
        mcp.run(transport="stdio")
    else:
        run_fastapi()
