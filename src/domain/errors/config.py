"""Configuration-related domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class ConfigError(DomainError):
    """Configuration error (invalid input, malformed value, bad request body)."""

    status_code = ErrorCode.BAD_REQUEST


class DependencyNotInitializedError(ConfigError):
    """A required dependency was not initialized (service unavailable)."""

    status_code = ErrorCode.SERVICE_UNAVAILABLE
