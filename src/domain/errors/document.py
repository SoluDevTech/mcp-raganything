"""Document reading/extraction domain errors (Kreuzberg)."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class DocumentReadError(DomainError):
    """Document reading/extraction runtime error."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class UnsupportedFormatError(DocumentReadError):
    """Unsupported file format or invalid file (bad request)."""

    status_code = ErrorCode.BAD_REQUEST
