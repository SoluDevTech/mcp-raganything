"""Get a single registered MCP server by name (secrets decrypted)."""

import logging

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.logging.messages import LogMessage
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

logger = logging.getLogger(__name__)


class GetMcpServerUseCase:
    """Retrieve a registered MCP server by name.

    Delegates to the store; the returned entry has secrets decrypted.
    """

    def __init__(self, store: McpServerRegistryStore) -> None:
        self._store = store

    async def execute(self, name: str) -> RegisteredMcpServer:
        """Retrieve a registered MCP server by name.

        Args:
            name: The unique server name.

        Returns:
            The registered server entry (secrets decrypted).

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        entry = await self._store.get(name)
        logger.info(LogMessage.MCP_SERVER_GET_UC, name)
        return entry
