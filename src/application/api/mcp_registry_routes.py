"""FastAPI routes for the MCP server registry.

Endpoints (all under ``/api/v1/mcp/servers``, protected by the global API key
dependency injected when the router is included in ``main.py``):

    POST   /                 — create a registered MCP server (201, masked)
    POST   /validate         — dry-run validation of a server config (200)
    GET    /                 — list all registered servers (200, masked)
    GET    /{name}           — get a single server (200, masked)
    GET    /{name}/reveal    — get a single server with plaintext secrets (200)
    PUT    /{name}           — update a server (200, masked)
    DELETE /{name}           — delete a server (204)

The openapi source_type extends the request/response bodies with two optional
fields:

- ``source_type`` (``"external"`` default | ``"openapi"``)
- ``openapi_url`` (``str | None``, required when ``source_type="openapi"``)

When ``source_type="openapi"``, the backend fetches the OpenAPI spec, builds an
in-process FastMCP server via the ``GeneratedMcpRunnerPort``, and mounts it
under ``/generated/{name}/mcp``; the ``url`` field of the request body is then
omitted (the mounted URL is generated and returned in the response).
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Self

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, model_validator

from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from dependencies import (
    get_create_mcp_server_use_case,
    get_delete_mcp_server_use_case,
    get_get_mcp_server_use_case,
    get_list_mcp_servers_use_case,
    get_update_mcp_server_use_case,
    get_validate_mcp_server_use_case,
)
from domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer

logger = logging.getLogger(__name__)

mcp_registry_router = APIRouter(prefix="/api/v1/mcp/servers", tags=["mcp-registry"])


class _McpServerBaseRequest(BaseModel):
    """Shared fields for create/update/validate request bodies.

    Per-source-type required fields are enforced by
    :meth:`validate_source_type_fields`: ``url`` is required for
    ``source_type="external"`` (a configured remote URL) and ``openapi_url`` is
    required for ``source_type="openapi"`` (the mounted URL is generated, so
    ``url`` is omitted on the request).
    """

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    source_type: str = "external"
    openapi_url: str | None = None

    @model_validator(mode="after")
    def validate_source_type_fields(self) -> Self:
        """Enforce per-source-type required fields.

        Returns:
            The validated request.

        Raises:
            ValueError: If ``source_type="openapi"`` without ``openapi_url``,
                or ``source_type="external"`` without ``url``.
        """
        if self.source_type == "openapi":
            if not self.openapi_url:
                raise ValueError("'openapi_url' is required when source_type='openapi'")
        else:  # external (default)
            if not self.url:
                raise ValueError("'url' is required when source_type='external'")
        return self


class McpServerCreateRequest(_McpServerBaseRequest):
    """Request body for creating a registered MCP server.

    Attributes:
        name: Unique server name (1-100 chars).
    """

    name: str = Field(..., min_length=1, max_length=100)


class McpServerValidateRequest(_McpServerBaseRequest):
    """Request body for dry-run validation of an MCP server config."""


class McpServerUpdateRequest(_McpServerBaseRequest):
    """Request body for updating a registered MCP server.

    The server ``name`` is taken from the URL path, not the body.
    """


class McpServerValidateResponse(BaseModel):
    """Response body for the validate endpoint."""

    tool_count: int


def _mask(entry: RegisteredMcpServer) -> RegisteredMcpServer:
    """Return a copy of ``entry`` with secrets stripped (auth_token, headers, env).

    ``auth_token`` is set to ``None`` in masked responses. This is ambiguous
    with an entry that genuinely has no token; use the ``/reveal`` endpoint to
    fetch the plaintext value and distinguish "no token" from "masked token".

    Args:
        entry: The registered server entry with plaintext secrets.

    Returns:
        A frozen copy with ``auth_token=None``, ``headers={}``, ``env={}``.
    """
    return entry.model_copy(update={"auth_token": None, "headers": {}, "env": {}})


@mcp_registry_router.post(
    "", response_model=RegisteredMcpServer, status_code=status.HTTP_201_CREATED
)
async def create_mcp_server(
    body: McpServerCreateRequest,
    use_case: Annotated[CreateMcpServerUseCase, Depends(get_create_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Create a new registered MCP server.

    Args:
        body: The request body with server configuration.
        use_case: Injected create use case.

    Returns:
        The created registered server entry (secrets masked).
    """
    now = datetime.now(UTC)
    entry = RegisteredMcpServer(
        name=body.name,
        transport=McpTransportType.HTTP,
        url=body.url,
        headers=body.headers,
        env=body.env,
        auth_token=body.auth_token,
        source_type=body.source_type,
        openapi_url=body.openapi_url,
        created_at=now,
        updated_at=now,
    )
    created = await use_case.execute(entry)
    return _mask(created)


@mcp_registry_router.post(
    "/validate",
    response_model=McpServerValidateResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_mcp_server(
    body: McpServerValidateRequest,
    use_case: Annotated[ValidateMcpServerUseCase, Depends(get_validate_mcp_server_use_case)],
) -> McpServerValidateResponse:
    """Dry-run validation of an MCP server connection.

    For ``source_type="external"`` (default), validates a remote MCP server URL.
    For ``source_type="openapi"``, fetches the OpenAPI spec, builds an in-process
    FastMCP server, and counts its tools (no mount, no persistence).

    Args:
        body: The request body with server configuration (no persistence).
        use_case: Injected validate use case.

    Returns:
        The number of tools discovered on the server.
    """
    if body.source_type == "openapi":
        tool_count = await use_case.execute_openapi(
            openapi_url=body.openapi_url or "",
            headers=body.headers,
        )
    else:
        config = McpServerConfig(
            name="validate",
            transport=McpTransportType.HTTP,
            url=body.url or "",
            headers=body.headers,
            env=body.env,
            auth_token=body.auth_token,
        )
        tool_count = await use_case.execute(config)
    return McpServerValidateResponse(tool_count=tool_count)


@mcp_registry_router.get(
    "", response_model=list[RegisteredMcpServer], status_code=status.HTTP_200_OK
)
async def list_mcp_servers(
    use_case: Annotated[ListMcpServersUseCase, Depends(get_list_mcp_servers_use_case)],
) -> list[RegisteredMcpServer]:
    """List all registered MCP servers (secrets masked).

    Args:
        use_case: Injected list use case.

    Returns:
        A list of registered server entries with secrets stripped.
    """
    entries = await use_case.execute()
    return [_mask(e) for e in entries]


@mcp_registry_router.get(
    "/{name}", response_model=RegisteredMcpServer, status_code=status.HTTP_200_OK
)
async def get_mcp_server(
    name: str,
    use_case: Annotated[GetMcpServerUseCase, Depends(get_get_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Get a single registered MCP server by name (secrets masked).

    Args:
        name: The unique server name.
        use_case: Injected get use case.

    Returns:
        The registered server entry with secrets stripped.
    """
    entry = await use_case.execute(name)
    return _mask(entry)


@mcp_registry_router.get(
    "/{name}/reveal", response_model=RegisteredMcpServer, status_code=status.HTTP_200_OK
)
async def reveal_mcp_server(
    name: str,
    use_case: Annotated[GetMcpServerUseCase, Depends(get_get_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Get a single registered MCP server with plaintext secrets revealed.

    Args:
        name: The unique server name.
        use_case: Injected get use case.

    Returns:
        The registered server entry with plaintext secrets.
    """
    return await use_case.execute(name)


@mcp_registry_router.put(
    "/{name}", response_model=RegisteredMcpServer, status_code=status.HTTP_200_OK
)
async def update_mcp_server(
    name: str,
    body: McpServerUpdateRequest,
    use_case: Annotated[UpdateMcpServerUseCase, Depends(get_update_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Update an existing registered MCP server (secrets masked in response).

    Args:
        name: The unique server name (from the URL path).
        body: The request body with updated server configuration.
        use_case: Injected update use case.

    Returns:
        The updated registered server entry with secrets stripped.
    """
    now = datetime.now(UTC)
    entry = RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url=body.url,
        headers=body.headers,
        env=body.env,
        auth_token=body.auth_token,
        source_type=body.source_type,
        openapi_url=body.openapi_url,
        created_at=now,
        updated_at=now,
    )
    updated = await use_case.execute(name, entry)
    return _mask(updated)


@mcp_registry_router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    name: str,
    use_case: Annotated[DeleteMcpServerUseCase, Depends(get_delete_mcp_server_use_case)],
) -> None:
    """Delete a registered MCP server by name.

    Args:
        name: The unique server name.
        use_case: Injected delete use case.
    """
    await use_case.execute(name)
