"""Port for loading tools from external MCP servers (runtime connection).

The external-path use cases (create/update/validate) use this port to validate
that a remote MCP server is reachable and to discover its tool count.
"""

from abc import ABC, abstractmethod
from typing import Any

from domain.entities.mcp_server_config import McpServerConfig


class McpToolLoader(ABC):
    """Outbound port: load tools from configured MCP servers."""

    @abstractmethod
    async def load_tools(self, configs: list[McpServerConfig]) -> list[Any]:
        """Load tools from the configured MCP servers.

        Args:
            configs: The MCP server connection configurations to load from.

        Returns:
            A list of discovered tool objects.

        Raises:
            McpConnectionError: If a server cannot be reached.
            McpToolLoadError: If the tools cannot be loaded.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close all open MCP connections."""
        ...
