"""FastAPI exception handlers for all domain errors.

Each handler reads the exception's own ``status_code`` and ``detail``,
logs at the appropriate level, and returns a uniform JSON response.
Imported and registered individually in ``main.py``.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from domain.errors.base import DomainError
from domain.errors.bricks import (
    BricksApiError,
    BricksConnectionError,
    BricksNotFoundError,
    BricksPermissionError,
    BricksTimeoutError,
)
from domain.errors.classical import ClassicalConfigError
from domain.errors.config import ConfigError, DependencyNotInitializedError
from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.file import FileError, FileTooLargeError, FileValidationError
from domain.errors.indexing import IndexingError
from domain.errors.rag import RagConfigError, RagEngineError, RagUnavailableError
from domain.errors.storage import StorageError, StorageNotFoundError
from domain.errors.vector_store import VectorStoreConfigError, VectorStoreError
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)


def _error_response(exc: DomainError) -> JSONResponse:
    """Build a JSON response from any domain error."""
    return JSONResponse(status_code=int(exc.status_code), content={"detail": exc.detail})


async def storage_not_found_handler(_request: Request, exc: StorageNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_STORAGE_ERROR, exc.detail)
    return _error_response(exc)


async def storage_error_handler(_request: Request, exc: StorageError) -> JSONResponse:
    logger.error(LogMessage.LOG_STORAGE_ERROR, exc.detail)
    return _error_response(exc)


async def bricks_not_found_handler(_request: Request, exc: BricksNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_BRICKS_NOT_FOUND, exc.detail)
    return _error_response(exc)


async def bricks_permission_handler(_request: Request, exc: BricksPermissionError) -> JSONResponse:
    logger.warning(LogMessage.LOG_BRICKS_PERMISSION, exc.detail)
    return _error_response(exc)


async def bricks_connection_handler(_request: Request, exc: BricksConnectionError) -> JSONResponse:
    logger.error(LogMessage.LOG_BRICKS_CONNECTION, exc.detail)
    return _error_response(exc)


async def bricks_timeout_handler(_request: Request, exc: BricksTimeoutError) -> JSONResponse:
    logger.error(LogMessage.LOG_BRICKS_TIMEOUT, exc.detail)
    return _error_response(exc)


async def bricks_api_error_handler(_request: Request, exc: BricksApiError) -> JSONResponse:
    logger.error(LogMessage.LOG_BRICKS_API_ERROR, exc.detail)
    return _error_response(exc)


async def unsupported_format_handler(_request: Request, exc: UnsupportedFormatError) -> JSONResponse:
    logger.warning(LogMessage.LOG_DOCUMENT_ERROR, exc.detail)
    return _error_response(exc)


async def document_error_handler(_request: Request, exc: DocumentReadError) -> JSONResponse:
    logger.error(LogMessage.LOG_DOCUMENT_ERROR, exc.detail)
    return _error_response(exc)


async def vector_bad_request_handler(_request: Request, exc: VectorStoreConfigError) -> JSONResponse:
    logger.warning(LogMessage.LOG_VECTOR_STORE_ERROR, exc.detail)
    return _error_response(exc)


async def vector_store_error_handler(_request: Request, exc: VectorStoreError) -> JSONResponse:
    logger.error(LogMessage.LOG_VECTOR_STORE_ERROR, exc.detail)
    return _error_response(exc)


async def rag_config_error_handler(_request: Request, exc: RagConfigError) -> JSONResponse:
    logger.warning(LogMessage.LOG_RAG_ERROR, exc.detail)
    return _error_response(exc)


async def rag_unavailable_handler(_request: Request, exc: RagUnavailableError) -> JSONResponse:
    logger.error(LogMessage.LOG_RAG_UNAVAILABLE, exc.detail)
    return _error_response(exc)


async def rag_error_handler(_request: Request, exc: RagEngineError) -> JSONResponse:
    logger.error(LogMessage.LOG_RAG_ERROR, exc.detail)
    return _error_response(exc)


async def classical_error_handler(_request: Request, exc: ClassicalConfigError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CLASSICAL_ERROR, exc.detail)
    return _error_response(exc)


async def indexing_error_handler(_request: Request, exc: IndexingError) -> JSONResponse:
    logger.error(LogMessage.LOG_INDEXING_ERROR, exc.detail)
    return _error_response(exc)


async def file_validation_handler(_request: Request, exc: FileValidationError) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


async def file_too_large_handler(_request: Request, exc: FileTooLargeError) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


async def file_error_handler(_request: Request, exc: FileError) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


async def dependency_not_initialized_handler(
    _request: Request, exc: DependencyNotInitializedError
) -> JSONResponse:
    logger.error(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)


async def config_error_handler(_request: Request, exc: ConfigError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)


async def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
    logger.error(LogMessage.LOG_UNHANDLED_DOMAIN_ERROR, exc.detail)
    return _error_response(exc)
