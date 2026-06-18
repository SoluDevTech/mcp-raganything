"""RAG engine domain errors (LightRAG/RAGAnything)."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class RagEngineError(DomainError):
    """RAG engine runtime error."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class RagConfigError(RagEngineError):
    """RAG engine configuration error (bad request)."""

    status_code = ErrorCode.BAD_REQUEST


class RagUnavailableError(RagEngineError):
    """RAG engine or required dependency not initialized (service unavailable)."""

    status_code = ErrorCode.SERVICE_UNAVAILABLE
