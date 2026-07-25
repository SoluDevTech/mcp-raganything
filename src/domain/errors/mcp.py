"""MCP (Model Context Protocol) domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class McpError(DomainError):
    """Base error for MCP operations."""

    status_code = ErrorCode.BAD_GATEWAY


class McpConnectionError(McpError):
    """Error connecting to an MCP server."""

    status_code = ErrorCode.BAD_GATEWAY


class McpToolLoadError(McpError):
    """Error loading MCP tools."""

    status_code = ErrorCode.BAD_GATEWAY


class McpServerNotFoundError(McpError):
    """Raised when a registered MCP server is not found in the registry."""

    status_code = ErrorCode.NOT_FOUND


class McpServerAlreadyExistsError(McpError):
    """Raised when attempting to register an MCP server whose name already exists."""

    status_code = ErrorCode.CONFLICT


class McpServerUnreachableError(McpError):
    """Raised when an MCP server cannot be reached or its tools cannot be loaded."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class McpAlreadyMountedError(McpError):
    """Raised when attempting to mount a generated MCP server whose name is already mounted."""

    status_code = ErrorCode.CONFLICT


class OpenApiFetchError(McpError):
    """Raised when an OpenAPI spec cannot be fetched (network/HTTP failure)."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class OpenApiInvalidSpecError(McpError):
    """Raised when a fetched OpenAPI spec is invalid (missing key, wrong version, too large)."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY
