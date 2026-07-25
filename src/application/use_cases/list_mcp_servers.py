"""List all registered MCP servers (secrets decrypted)."""

import logging

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.logging.messages import LogMessage
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

logger = logging.getLogger(__name__)


class ListMcpServersUseCase:
    """List all registered MCP servers.

    Delegates to the store; returned entries have secrets decrypted.
    """

    def __init__(self, store: McpServerRegistryStore) -> None:
        self._store = store

    async def execute(self) -> list[RegisteredMcpServer]:
        """List all registered MCP servers.

        Returns:
            A list of all registered server entries (secrets decrypted).
        """
        result = await self._store.list_all()
        logger.info(LogMessage.MCP_SERVER_LISTED_UC, len(result))
        return result
