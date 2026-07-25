"""Update an existing registered MCP server.

For ``source_type="external"`` (default): re-validates the server connection
(re-loads tools) only when the URL has changed, then persists the updated entry
with the fresh tool count.

For ``source_type="openapi"``: unmounts the old in-process server, re-fetches
the spec, mounts the new server via the runner, and counts tools. On success
the entry is persisted with the refreshed tool count and mounted URL.
"""

import logging
from datetime import UTC, datetime

from domain.entities.mcp_server_config import McpServerConfig
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import (
    McpConnectionError,
    McpServerUnreachableError,
    McpToolLoadError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory

logger = logging.getLogger(__name__)


class UpdateMcpServerUseCase:
    """Update an existing registered MCP server.

    Re-validates reachability when the URL changes (external) or always
    (openapi, since the spec may have changed), then persists the updated entry
    with the discovered tool count.
    """

    def __init__(
        self,
        store: McpServerRegistryStore,
        tool_loader: McpToolLoader | None = None,
        openapi_factory: OpenApiMcpFactory | None = None,
        runner: GeneratedMcpRunnerPort | None = None,
    ) -> None:
        self._store = store
        self._tool_loader = tool_loader
        self._openapi_factory = openapi_factory
        self._runner = runner

    async def execute(
        self, name: str, updated: RegisteredMcpServer
    ) -> RegisteredMcpServer:
        """Update a registered MCP server.

        Args:
            name: The unique server name (must match ``updated.name``).
            updated: The new entry values.

        Returns:
            The persisted entry with the (possibly refreshed) tool count.

        Raises:
            McpServerNotFoundError: If no server exists for this name.
            McpServerUnreachableError: If re-validation fails after a URL change.
            OpenApiFetchError: If the OpenAPI spec cannot be fetched (openapi).
            OpenApiInvalidSpecError: If the OpenAPI spec is invalid (openapi).
        """
        # Confirm the entry exists (raises McpServerNotFoundError if absent).
        existing = await self._store.get(name)

        # Preserve the original creation timestamp; only ``updated_at`` moves
        # forward. The DB upsert does not touch ``created_at`` either, but we
        # keep the entity truthful so the API response and any future write
        # path reflect reality.
        now = datetime.now(UTC)
        updated = updated.model_copy(
            update={"created_at": existing.created_at, "updated_at": now}
        )

        if updated.source_type == "openapi":
            return await self._update_openapi(name, updated)

        to_save = updated
        # Re-validate reachability only when the URL changed.
        if existing.url != updated.url and self._tool_loader is not None:
            config = McpServerConfig(
                name=updated.name,
                transport=updated.transport,
                url=updated.url,
                headers=updated.headers,
                env=updated.env,
                auth_token=updated.auth_token,
            )
            try:
                tools = await self._tool_loader.load_tools([config])
                to_save = updated.model_copy(update={"tool_count": len(tools)})
            except (McpConnectionError, McpToolLoadError) as e:
                raise McpServerUnreachableError(
                    ErrorMessage.MCP_SERVER_UNREACHABLE.format(error=e)
                ) from e

        await self._store.save(to_save)
        logger.info(
            LogMessage.MCP_SERVER_UPDATED_UC,
            to_save.name,
            to_save.tool_count,
        )
        return to_save

    async def _update_openapi(
        self, name: str, updated: RegisteredMcpServer
    ) -> RegisteredMcpServer:
        """Re-fetch + remount + count tools for an openapi server.

        Args:
            name: The unique server name.
            updated: The new entry values (``source_type="openapi"``).

        Returns:
            The persisted entry with the refreshed tool count and mounted URL.

        Raises:
            OpenApiFetchError: If the spec cannot be fetched.
            OpenApiInvalidSpecError: If the spec is invalid.
        """
        assert self._openapi_factory is not None  # noqa: S101
        assert self._runner is not None  # noqa: S101

        spec = await self._openapi_factory.fetch_spec(
            updated.openapi_url or "", headers=updated.headers
        )

        # Unmount the old in-process server before mounting the new one.
        await self._runner.unmount(name)
        mounted_url = await self._runner.mount(updated.name, spec, headers=updated.headers)
        tool_count = await self._runner.count_tools(updated.name)

        to_save = updated.model_copy(
            update={
                "tool_count": tool_count,
                "url": mounted_url,
                "source_type": "openapi",
                "openapi_url": updated.openapi_url,
            }
        )
        await self._store.save(to_save)
        logger.info(LogMessage.MCP_SERVER_UPDATED_UC, to_save.name, tool_count)
        return to_save
