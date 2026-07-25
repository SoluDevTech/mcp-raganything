"""Delete a registered MCP server by name.

For ``source_type="external"`` (default): delegates to the store; raises
:class:`McpServerNotFoundError` if the server does not exist.

For ``source_type="openapi"``: unmounts the in-process server (when a runner is
injected) before deleting the registry entry, so the ASGI app is no longer
reachable.

The runner is optional so the external path stays backward-compatible with
callers that only inject the store.
"""

import logging

from domain.logging.messages import LogMessage
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

logger = logging.getLogger(__name__)


class DeleteMcpServerUseCase:
    """Delete a registered MCP server.

    Delegates to the store; raises :class:`McpServerNotFoundError` if the
    server does not exist. When a runner is injected and the server is an
    in-process openapi server, it is unmounted first.
    """

    def __init__(
        self,
        store: McpServerRegistryStore,
        runner: GeneratedMcpRunnerPort | None = None,
    ) -> None:
        self._store = store
        self._runner = runner

    async def execute(self, name: str) -> None:
        """Delete a registered MCP server by name.

        Args:
            name: The unique server name to delete.

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        if self._runner is not None:
            entry = await self._store.get(name)
            if entry.source_type == "openapi":
                await self._runner.unmount(name)

        await self._store.delete(name)
        logger.info(LogMessage.MCP_SERVER_DELETED_UC, name)
