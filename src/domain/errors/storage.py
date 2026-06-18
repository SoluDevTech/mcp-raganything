"""Storage-related domain errors (MinIO/object storage)."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class StorageError(DomainError):
    """Storage layer error (service unavailable)."""

    status_code = ErrorCode.SERVICE_UNAVAILABLE


class StorageNotFoundError(StorageError):
    """Object or bucket not found in storage."""

    status_code = ErrorCode.NOT_FOUND
