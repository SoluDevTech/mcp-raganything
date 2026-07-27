"""FastAPI exception handlers for all domain errors.

Each handler reads the exception's own ``status_code`` and ``detail``,
logs at the appropriate level, and returns a uniform JSON response.
Imported and registered individually in ``main.py``.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from domain.errors.base import DomainError
from domain.errors.classical import ClassicalConfigError
from domain.errors.config import ConfigError, DependencyNotInitializedError
from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.file import FileError, FileTooLargeError, FileValidationError
from domain.errors.indexing import IndexingError
from domain.errors.llm import LlmNotConfiguredError
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from domain.errors.storage import StorageError, StorageNotFoundError
from domain.errors.vector_store import VectorStoreConfigError, VectorStoreError
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)


def _error_response(exc: DomainError) -> JSONResponse:
    """Build a JSON response from any domain error."""
    return JSONResponse(
        status_code=int(exc.status_code), content={"detail": exc.detail}
    )


def storage_not_found_handler(
    _request: Request, exc: StorageNotFoundError
) -> JSONResponse:
    logger.warning(LogMessage.LOG_STORAGE_ERROR, exc.detail)
    return _error_response(exc)


def storage_error_handler(_request: Request, exc: StorageError) -> JSONResponse:
    logger.error(LogMessage.LOG_STORAGE_ERROR, exc.detail)
    return _error_response(exc)


def unsupported_format_handler(
    _request: Request, exc: UnsupportedFormatError
) -> JSONResponse:
    logger.warning(LogMessage.LOG_DOCUMENT_ERROR, exc.detail)
    return _error_response(exc)


def document_error_handler(_request: Request, exc: DocumentReadError) -> JSONResponse:
    logger.error(LogMessage.LOG_DOCUMENT_ERROR, exc.detail)
    return _error_response(exc)


def vector_bad_request_handler(
    _request: Request, exc: VectorStoreConfigError
) -> JSONResponse:
    logger.warning(LogMessage.LOG_VECTOR_STORE_ERROR, exc.detail)
    return _error_response(exc)


def vector_store_error_handler(
    _request: Request, exc: VectorStoreError
) -> JSONResponse:
    logger.error(LogMessage.LOG_VECTOR_STORE_ERROR, exc.detail)
    return _error_response(exc)


def classical_error_handler(
    _request: Request, exc: ClassicalConfigError
) -> JSONResponse:
    logger.warning(LogMessage.LOG_CLASSICAL_ERROR, exc.detail)
    return _error_response(exc)


def indexing_error_handler(_request: Request, exc: IndexingError) -> JSONResponse:
    logger.error(LogMessage.LOG_INDEXING_ERROR, exc.detail)
    return _error_response(exc)


def file_validation_handler(
    _request: Request, exc: FileValidationError
) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


def file_too_large_handler(_request: Request, exc: FileTooLargeError) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


def file_error_handler(_request: Request, exc: FileError) -> JSONResponse:
    logger.warning(LogMessage.LOG_FILE_ERROR, exc.detail)
    return _error_response(exc)


def dependency_not_initialized_handler(
    _request: Request, exc: DependencyNotInitializedError
) -> JSONResponse:
    logger.error(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)


def config_error_handler(_request: Request, exc: ConfigError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)


def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
    logger.error(LogMessage.LOG_UNHANDLED_DOMAIN_ERROR, exc.detail)
    return _error_response(exc)


def invalid_api_key_handler(_request: Request, exc: InvalidApiKeyError) -> JSONResponse:
    logger.warning(LogMessage.LOG_INVALID_API_KEY, exc.detail)
    return _error_response(exc)


def authentication_error_handler(
    _request: Request, exc: AuthenticationError
) -> JSONResponse:
    """Map :class:`AuthenticationError` (raised by verify_credentials) to 401."""
    logger.warning(LogMessage.LOG_INVALID_API_KEY, exc.detail)
    return _error_response(exc)


def llm_not_configured_handler(
    _request: Request, exc: LlmNotConfiguredError
) -> JSONResponse:
    """Map :class:`LlmNotConfiguredError` to 422 (user has no LLM provider)."""
    logger.warning(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)
