"""Base domain exception.

All application errors inherit from :class:`DomainError`. Each subclass
declares its own ``status_code`` (an :class:`~domain.errors.codes.ErrorCode`
constant) so that a single generic HTTP handler can translate any domain error
into a response without a separate mapping registry.

Attributes:
    status_code: HTTP status code mapped to this error (see ErrorCode).
    detail: Human-readable error description.
"""

from domain.errors.codes import ErrorCode


class DomainError(Exception):
    """Base domain exception. All application errors inherit from it."""

    status_code: int = ErrorCode.INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(self.detail)
