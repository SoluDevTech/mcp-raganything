"""Configuration for connecting to an MCP server (runtime connection config).

Distinct from :class:`RegisteredMcpServer` (the persisted registry entry): this
entity is the lightweight configuration handed to the MCP tool loader to open a
connection at runtime.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class McpTransportType(StrEnum):
    """Transport types supported by the MCP client."""

    STDIO = "stdio"
    HTTP = "http"


class McpServerConfig(BaseModel, frozen=True):
    """Configuration for connecting to an MCP server.

    Attributes:
        name: Unique server name (1+ chars).
        transport: Transport type (stdio or http).
        command: Command to run for stdio transport (required when stdio).
        args: Arguments for the stdio command.
        url: Server URL (required for http transport).
        headers: HTTP headers to send to the server.
        env: Environment variables for stdio transport.
        auth_token: Bearer auth token (None if unset).
    """

    name: str = Field(..., min_length=1)
    transport: McpTransportType
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None

    @model_validator(mode="after")
    def validate_transport_fields(self) -> Self:
        """Enforce per-transport required fields.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``command`` is missing for stdio or ``url`` for http.
        """
        if self.transport == McpTransportType.STDIO and not self.command:
            raise ValueError("'command' is required for stdio transport")
        if self.transport == McpTransportType.HTTP and not self.url:
            raise ValueError("'url' is required for http transport")
        return self
