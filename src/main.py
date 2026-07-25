"""Main entry point for the MCP service."""

import logging
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.api.classical_indexing_routes import classical_indexing_router
from application.api.classical_query_routes import classical_query_router
from application.api.file_routes import file_router
from application.api.health_routes import health_router
from application.api.mcp_classical_tools import mcp_classical
from application.api.mcp_file_tools import mcp_files
from application.api.mcp_registry_routes import mcp_registry_router
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
    db_config,
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
from infrastructure.database.mcp_registry_store import McpRegistryStore
from infrastructure.logging import RequestIdMiddleware, configure_logging
from infrastructure.openapi_mcp.rehydration import rehydrate_openapi_servers
from infrastructure.openapi_mcp.runner import GeneratedMcpRunner
from infrastructure.security.crypto import FernetSecretCipher
from security import ComposableAgentsSecurity, McpApiKeyMiddleware

configure_logging(app_config)

logger = logging.getLogger(__name__)

security = ComposableAgentsSecurity(master_key=app_config.API_KEY)

#: asyncpg pool for the MCP registry store; created in the lifespan, closed on
#: shutdown. None until the lifespan runs.
_mcp_pool: asyncpg.Pool | None = None


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
    """Combine database lifecycle with MCP lifespans and registry rehydration."""
    # --- MCP registry: build pool, store, ensure schema, rehydrate servers ---
    global _mcp_pool
    runner = getattr(app.state, "generated_mcp_runner", None)
    cipher = None
    store = None
    try:
        if app_config.SECRET_ENCRYPTION_KEY:
            cipher = FernetSecretCipher(app_config.SECRET_ENCRYPTION_KEY)
            _mcp_pool = await asyncpg.create_pool(
                dsn=db_config.asyncpg_url,
                statement_cache_size=db_config.POSTGRES_STATEMENT_CACHE_SIZE or 100,
            )
            store = McpRegistryStore(pool=_mcp_pool, cipher=cipher)
            _deps.mcp_registry_store = store
            if runner is not None:
                await rehydrate_openapi_servers(
                    store=store,
                    runner=runner,
                    factory=_deps.get_mcp_openapi_factory(),
                )
        else:
            logger.warning(
                "SECRET_ENCRYPTION_KEY not set; MCP server registry is disabled."
            )
    except Exception:
        logger.exception("MCP registry initialization failed; continuing startup.")

    async with (
        db_lifespan(app),
        mcp_files_app.lifespan(app),
        mcp_classical_app.lifespan(app),
    ):
        yield

    # --- Shutdown: close generated MCP runner and the registry pool ---
    if runner is not None:
        try:
            await runner.close()
        except Exception:
            logger.exception("Failed to close generated MCP runner.")
    if _mcp_pool is not None:
        await _mcp_pool.close()


app = FastAPI(
    title="RAG Anything API",
    lifespan=combined_lifespan,
)

# Wire the generated MCP runner now that the app exists. The runner holds the
# app reference so it can mount in-process FastMCP servers under
# /generated/{name}/mcp at runtime. Stored on app.state so the lifespan can
# access it, and on the dependencies module so the use-case providers resolve it.
generated_mcp_runner = GeneratedMcpRunner(
    app=app,
    base_url=app_config.GENERATED_MCP_BASE_URL,
    middleware=[mcp_middleware],
)
app.state.generated_mcp_runner = generated_mcp_runner
import dependencies as _deps  # noqa: E402

_deps.generated_mcp_runner = generated_mcp_runner

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
# mcp_registry_router already carries the full /api/v1/mcp/servers prefix, so
# it is included without an additional prefix (only the API key dependency).
app.include_router(mcp_registry_router, dependencies=_api_key_dep)

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
