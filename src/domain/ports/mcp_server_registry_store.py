"""Port for the MCP server registry store.

The registry stores registered MCP servers keyed by name. Secrets
(``headers``, ``env``, ``auth_token``) are encrypted at rest by the adapter
implementation using a :class:`SecretCipher`, and decrypted transparently on
read so the domain layer only ever sees plaintext.
"""

from abc import ABC, abstractmethod

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer


class McpServerRegistryStore(ABC):
    """Outbound port: persist registered MCP servers.

    The ``mcp_servers`` table schema is owned by this service's Alembic
    migrations (``001_create_mcp_servers_table``), run at startup.
    """

    @abstractmethod
    async def save(self, entry: RegisteredMcpServer) -> None:
        """Insert or update (upsert) a registered MCP server by name.

        Args:
            entry: The registered server entry to persist.
        """
        ...

    @abstractmethod
    async def get(self, name: str) -> RegisteredMcpServer:
        """Retrieve a registered MCP server by name.

        Args:
            name: The unique server name.

        Returns:
            The registered server entry (secrets decrypted).

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[RegisteredMcpServer]:
        """List all registered MCP servers.

        Returns:
            A list of all registered server entries (secrets decrypted).
        """
        ...

    @abstractmethod
    async def delete(self, name: str) -> None:
        """Delete a registered MCP server by name.

        Args:
            name: The unique server name to delete.

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        ...

    @abstractmethod
    async def exists(self, name: str) -> bool:
        """Check whether a registered MCP server exists by name.

        Args:
            name: The unique server name to check.

        Returns:
            True if a server exists, False otherwise.
        """
        ...
