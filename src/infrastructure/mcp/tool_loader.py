"""Infrastructure adapter: load tools from external MCP servers via fastmcp.

Implements :class:`McpToolLoader` using ``fastmcp.Client`` over either a
``StreamableHttpTransport`` (HTTP, with Bearer auth from ``auth_token``) or a
``StdioTransport`` (STDIO, with command/args/env). ``${VAR}`` placeholders in
headers (HTTP) and env (STDIO) are resolved against ``os.environ`` before the
transport is constructed.

Error mapping:
- ``ConnectionError`` raised while entering the client context ->
  :class:`McpConnectionError`.
- Any other exception raised during tool loading -> :class:`McpToolLoadError`.
- ``McpConnectionError`` / ``McpToolLoadError`` raised by an inner call are
  re-raised unchanged (no double wrapping).
"""

import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport
from fastmcp.client.transports.stdio import StdioTransport

from domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from domain.errors.mcp import McpConnectionError, McpToolLoadError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.mcp_tool_loader import McpToolLoader
from infrastructure.env_utils import resolve_env_vars_in_dict

logger = logging.getLogger(__name__)


class FastMcpToolLoader(McpToolLoader):
    """Concrete :class:`McpToolLoader` backed by ``fastmcp.Client``.

    Attributes:
        _tool_timeout: Per-client timeout (seconds) passed to ``fastmcp.Client``.
        _clients: Clients returned by ``Client.__aenter__`` and closed by
            ``close()``. Reset to an empty list on ``close()`` so a second
            ``close()`` is a no-op.
    """

    def __init__(self, tool_timeout: float = 60.0) -> None:
        self._tool_timeout = tool_timeout
        self._clients: list[Any] = []

    async def load_tools(self, configs: list[McpServerConfig]) -> list[Any]:
        """Load tools from the configured MCP servers.

        Args:
            configs: The MCP server connection configurations to load from.

        Returns:
            A list of discovered tool objects aggregated across all configs.

        Raises:
            McpConnectionError: If a server cannot be reached.
            McpToolLoadError: If the tools cannot be loaded.
        """
        all_tools: list[Any] = []
        for config in configs:
            if config.transport == McpTransportType.HTTP:
                assert config.url is not None  # noqa: S101 — entity validator guarantees url for HTTP
                headers = resolve_env_vars_in_dict(config.headers)
                transport = StreamableHttpTransport(
                    url=config.url,
                    headers=headers,
                    auth=config.auth_token or None,
                )
            elif config.transport == McpTransportType.STDIO:
                assert config.command is not None  # noqa: S101 — entity validator guarantees command for STDIO
                env = resolve_env_vars_in_dict(config.env)
                transport = StdioTransport(
                    command=config.command,
                    args=config.args,
                    env=env,
                )
            else:
                raise McpToolLoadError(
                    ErrorMessage.MCP_TOOL_LOAD_ERROR.format(
                        error=f"unsupported transport: {config.transport}"
                    )
                )
            try:
                async with Client(transport, timeout=self._tool_timeout) as client:
                    self._clients.append(client)
                    tools = await client.list_tools()
                    all_tools.extend(tools)
            except ConnectionError as e:
                logger.exception(LogMessage.MCP_CONNECT_FAILED, config.name, str(e))
                raise McpConnectionError(
                    ErrorMessage.MCP_CONNECTION_ERROR.format(error=e)
                ) from e
            except (McpConnectionError, McpToolLoadError):
                raise
            except Exception as e:
                logger.exception(LogMessage.MCP_TOOLS_LOAD_FAILED, config.name, str(e))
                raise McpToolLoadError(
                    ErrorMessage.MCP_TOOL_LOAD_ERROR.format(error=e)
                ) from e
        return all_tools

    async def close(self) -> None:
        """Close all open MCP clients.

        Idempotent: clears ``self._clients`` before iterating so a second call
        is a no-op. Tolerates clients without a ``close`` attribute and sync
        ``close()`` return values (awaits coroutine results only).
        """
        clients = self._clients
        self._clients = []
        for client in clients:
            close_fn = getattr(client, "close", None)
            if close_fn is None:
                continue
            result = close_fn()
            if hasattr(result, "__await__"):
                await result
