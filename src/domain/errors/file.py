"""File upload/validation domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class FileError(DomainError):
    """Generic file handling error (bad request)."""

    status_code = ErrorCode.BAD_REQUEST


class FileValidationError(FileError):
    """File validation error (unprocessable entity)."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class FileTooLargeError(FileError):
    """Uploaded file exceeds the maximum allowed size."""

    status_code = 413
