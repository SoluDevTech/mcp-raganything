"""Main entry point for the RAGAnything API.
Simplified following hexagonal architecture pattern from pickpro_indexing_api.
"""

import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.api.health_routes import health_router
from application.api.indexing_routes import indexing_router
from application.api.mcp_tools import mcp
from application.api.query_routes import query_router
from dependencies import app_config

logger = logging.getLogger(__name__)


MCP_PATH = "/mcp"

if app_config.MCP_TRANSPORT == "streamable":
    mcp_app = mcp.http_app(path="/")
    app = FastAPI(
        title="RAG Anything API",
        lifespan=mcp_app.lifespan,
    )
    app.mount(MCP_PATH, mcp_app)
else:
    app = FastAPI(title="RAG Anything API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= REST API ROUTES =============

REST_PATH = "/api/v1"

app.include_router(indexing_router, prefix=REST_PATH)
app.include_router(health_router, prefix=REST_PATH)
app.include_router(query_router, prefix=REST_PATH)

# ============= MAIN =============


def run_fastapi():
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
