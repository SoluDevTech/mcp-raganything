"""Indexing domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class IndexingError(DomainError):
    """Indexing/query pipeline runtime error."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR
