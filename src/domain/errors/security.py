from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class SecurityError(DomainError):
    """Base error for security concerns."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class InvalidApiKeyError(SecurityError):
    """Error raised when the API key sent by the client is invalid or missing."""

    status_code = ErrorCode.UNAUTHORIZED

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)


class AuthenticationError(SecurityError):
    """Raised when no valid credentials (JWT or API key) could be resolved.

    Mapped to HTTP 401 by the generic domain-error handler.
    """

    status_code = ErrorCode.UNAUTHORIZED

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
