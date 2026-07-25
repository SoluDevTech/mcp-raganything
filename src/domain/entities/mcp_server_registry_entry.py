"""Registered MCP server domain entity (registry entry).

A ``RegisteredMcpServer`` is the persisted representation of an MCP server
that has been registered in the registry. Unlike :class:`McpServerConfig`
(which is the connection configuration used to load tools at runtime), this
entity carries the registry metadata: name, url, encrypted-at-rest secrets
(headers/env/auth_token are decrypted by the store on read), the resolved
``tool_count``, the source type (``external`` vs ``openapi``), and audit
timestamps.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.entities.mcp_server_config import McpTransportType


class RegisteredMcpServer(BaseModel):
    """A registered MCP server entry in the registry.

    Attributes:
        name: Unique server name (1-100 chars). Acts as the primary key.
        transport: Transport type (stdio or http). Defaults to http.
        url: Server URL. Required and must be non-empty.
        headers: HTTP headers to send to the server (decrypted on read).
        env: Environment variables for stdio transport (decrypted on read).
        auth_token: Bearer auth token (decrypted on read, None if unset).
        tool_count: Number of tools discovered during the last validation.
        source_type: Origin of the server — ``"external"`` (plain MCP server
            reached over the network) or ``"openapi"`` (server built in-process
            from an OpenAPI spec via FastMCP and mounted under
            ``/generated/{name}/mcp``). Defaults to ``"external"``.
        openapi_url: URL the OpenAPI spec was fetched from (only set when
            ``source_type == "openapi"``).
        created_at: Creation timestamp (UTC).
        updated_at: Last update timestamp (UTC).
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=100)
    transport: McpTransportType = McpTransportType.HTTP
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    tool_count: int = 0
    source_type: str = "external"
    openapi_url: str | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_url_required(self) -> Self:
        """Ensure url is present and non-empty.

        Returns:
            The validated entity.

        Raises:
            ValueError: If url is empty.
        """
        if not self.url:
            raise ValueError("'url' is required and must be non-empty")
        return self
