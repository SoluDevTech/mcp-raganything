"""Centralized HTTP status code constants for domain errors.

Each domain exception declares one of these codes as a ``status_code`` class
attribute, so each registered HTTP handler can read ``exc.status_code`` to build
the response without a separate mapping registry.
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    """HTTP status codes used by domain error handlers."""

    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    GONE = 410
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
