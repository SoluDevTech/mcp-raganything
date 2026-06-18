"""Bricks API domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class BricksApiError(DomainError):
    """Generic Bricks API error."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class BricksNotFoundError(BricksApiError):
    """Bricks project or document not found."""

    status_code = ErrorCode.NOT_FOUND


class BricksPermissionError(BricksApiError):
    """Bricks API authentication/authorization failure."""

    status_code = ErrorCode.FORBIDDEN


class BricksConnectionError(BricksApiError):
    """Bricks API connection failure (bad gateway)."""

    status_code = ErrorCode.BAD_GATEWAY


class BricksTimeoutError(BricksApiError):
    """Bricks API request timed out (bad gateway)."""

    status_code = ErrorCode.BAD_GATEWAY
