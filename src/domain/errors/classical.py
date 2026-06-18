"""Classical RAG pathway domain errors (BM25/vector helpers)."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class ClassicalConfigError(DomainError):
    """Classical RAG configuration error (invalid table prefix, bad input)."""

    status_code = ErrorCode.BAD_REQUEST
