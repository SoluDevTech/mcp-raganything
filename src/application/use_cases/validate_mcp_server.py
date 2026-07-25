"""Validate an MCP server connection (dry-run, no persistence).

Two dry-run modes:

- External: calls the MCP tool loader with the given config and returns the
  number of tools discovered.
- OpenAPI: fetches the OpenAPI spec, builds an in-process FastMCP server, and
  counts its tools WITHOUT mounting or persisting anything.

Both raise :class:`McpServerUnreachableError` if the validation fails.
"""

import logging

from domain.entities.mcp_server_config import McpServerConfig
from domain.errors.mcp import (
    McpConnectionError,
    McpServerUnreachableError,
    McpToolLoadError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory

logger = logging.getLogger(__name__)


class ValidateMcpServerUseCase:
    """Dry-run validation of an MCP server connection.

    Loads tools from the configured server without persisting anything, and
    returns the discovered tool count. For openapi servers, fetches the spec,
    builds the server and counts its tools (no mount, no save).
    """

    def __init__(
        self,
        tool_loader: McpToolLoader | None = None,
        openapi_factory: OpenApiMcpFactory | None = None,
    ) -> None:
        self._tool_loader = tool_loader
        self._openapi_factory = openapi_factory

    async def execute(self, config: McpServerConfig) -> int:
        """Validate an external MCP server connection and return the tool count.

        Args:
            config: The MCP server connection configuration to validate.

        Returns:
            The number of tools discovered on the server.

        Raises:
            McpServerUnreachableError: If the server cannot be reached or
                its tools cannot be loaded.
        """
        if self._tool_loader is None:
            raise McpServerUnreachableError(
                ErrorMessage.MCP_SERVER_UNREACHABLE.format(
                    error="external server validation is not available (no tool loader configured)"
                )
            )
        try:
            tools = await self._tool_loader.load_tools([config])
        except (McpConnectionError, McpToolLoadError) as e:
            raise McpServerUnreachableError(
                ErrorMessage.MCP_SERVER_UNREACHABLE.format(error=e)
            ) from e

        count = len(tools)
        logger.info(LogMessage.MCP_SERVER_VALIDATED_UC, config.name, count)
        return count

    async def execute_openapi(self, openapi_url: str, headers: dict[str, str]) -> int:
        """Dry-run validation of an OpenAPI spec (no mount, no persistence).

        Fetches the spec, builds an in-process FastMCP server, and counts its
        tools. Nothing is mounted or saved.

        Args:
            openapi_url: The URL of the OpenAPI document to validate.
            headers: Upstream auth headers to send when fetching the spec.

        Returns:
            The number of tools the built server exposes.

        Raises:
            OpenApiFetchError: If the spec cannot be fetched.
            OpenApiInvalidSpecError: If the spec is invalid.
        """
        assert self._openapi_factory is not None  # noqa: S101
        spec = await self._openapi_factory.fetch_spec(openapi_url, headers=headers)
        mcp = await self._openapi_factory.build_mcp(spec, base_url="", headers=headers)
        count = await self._openapi_factory.count_tools(mcp)

        logger.info(LogMessage.MCP_SERVER_VALIDATED_UC, openapi_url, count)
        return count
