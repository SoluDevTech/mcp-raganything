"""Vector store domain errors (pgvector/LangChain vector store)."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class VectorStoreError(DomainError):
    """Vector store runtime error."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class VectorStoreConfigError(VectorStoreError):
    """Vector store configuration/setup error (bad request)."""

    status_code = ErrorCode.BAD_REQUEST
