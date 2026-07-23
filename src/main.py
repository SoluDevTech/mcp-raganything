"""Main entry point for the MCP service."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.api.classical_indexing_routes import classical_indexing_router
from application.api.classical_query_routes import classical_query_router
from application.api.file_routes import file_router
from application.api.health_routes import health_router
from application.api.mcp_classical_tools import mcp_classical
from application.api.mcp_file_tools import mcp_files
from application.error_handlers import (
    classical_error_handler,
    config_error_handler,
    dependency_not_initialized_handler,
    document_error_handler,
    domain_error_handler,
    file_error_handler,
    file_too_large_handler,
    file_validation_handler,
    indexing_error_handler,
    invalid_api_key_handler,
    storage_error_handler,
    storage_not_found_handler,
    unsupported_format_handler,
    vector_bad_request_handler,
    vector_store_error_handler,
)
from dependencies import (
    app_config,
    classical_bm25_adapter,
    classical_vector_store,
)
from domain.errors.base import DomainError
from domain.errors.classical import ClassicalConfigError
from domain.errors.config import ConfigError, DependencyNotInitializedError
from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.file import FileError, FileTooLargeError, FileValidationError
from domain.errors.indexing import IndexingError
from domain.errors.security import InvalidApiKeyError
from domain.errors.storage import StorageError, StorageNotFoundError
from domain.errors.vector_store import VectorStoreConfigError, VectorStoreError
from domain.logging.messages import LogMessage
from infrastructure.logging import RequestIdMiddleware, configure_logging
from security import ComposableAgentsSecurity, McpApiKeyMiddleware

configure_logging(app_config)

logger = logging.getLogger(__name__)

security = ComposableAgentsSecurity(master_key=app_config.API_KEY)


@asynccontextmanager
async def db_lifespan(_app: FastAPI):
    """Closes connection pools and HTTP clients on shutdown."""
    yield

    logger.info(LogMessage.APP_SHUTDOWN_INITIATED)
    if classical_bm25_adapter is not None:
        try:
            await classical_bm25_adapter.close()
        except Exception:
            logger.exception(LogMessage.APP_BM25_CLOSE_FAILED)
    if classical_vector_store is not None:
        try:
            await classical_vector_store.close()
        except Exception:
            logger.exception(LogMessage.APP_CLASSICAL_VS_CLOSE_FAILED)
    logger.info(LogMessage.APP_SHUTDOWN_COMPLETE)


mcp_middleware = McpApiKeyMiddleware(master_key=app_config.API_KEY)
mcp_files.add_middleware(mcp_middleware)
mcp_classical.add_middleware(mcp_middleware)

mcp_files_app = mcp_files.http_app(path="/")
mcp_classical_app = mcp_classical.http_app(path="/")


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combine database lifecycle with MCP lifespans."""
    async with (
        db_lifespan(app),
        mcp_files_app.lifespan(app),
        mcp_classical_app.lifespan(app),
    ):
        yield


app = FastAPI(
    title="RAG Anything API",
    lifespan=combined_lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REST_PATH = "/api/v1"
app.include_router(health_router, prefix=REST_PATH)

_api_key_dep = [Depends(security.verify_api_key)]
app.include_router(file_router, prefix=REST_PATH, dependencies=_api_key_dep)
app.include_router(
    classical_indexing_router, prefix=REST_PATH, dependencies=_api_key_dep
)
app.include_router(classical_query_router, prefix=REST_PATH, dependencies=_api_key_dep)

app.mount("/files/mcp", mcp_files_app)
app.mount("/classical/mcp", mcp_classical_app)

app.add_exception_handler(StorageNotFoundError, storage_not_found_handler)
app.add_exception_handler(StorageError, storage_error_handler)
app.add_exception_handler(UnsupportedFormatError, unsupported_format_handler)
app.add_exception_handler(DocumentReadError, document_error_handler)
app.add_exception_handler(VectorStoreConfigError, vector_bad_request_handler)
app.add_exception_handler(VectorStoreError, vector_store_error_handler)
app.add_exception_handler(ClassicalConfigError, classical_error_handler)
app.add_exception_handler(IndexingError, indexing_error_handler)
app.add_exception_handler(FileValidationError, file_validation_handler)
app.add_exception_handler(FileTooLargeError, file_too_large_handler)
app.add_exception_handler(FileError, file_error_handler)
app.add_exception_handler(
    DependencyNotInitializedError, dependency_not_initialized_handler
)
app.add_exception_handler(ConfigError, config_error_handler)
app.add_exception_handler(InvalidApiKeyError, invalid_api_key_handler)
app.add_exception_handler(DomainError, domain_error_handler)


def run_fastapi():
    """Run FastAPI server with uvicorn."""
    uvicorn.run(
        "main:app",
        host=app_config.HOST,
        port=app_config.PORT,
        ws="none",
    )


if __name__ == "__main__":
    run_fastapi()
