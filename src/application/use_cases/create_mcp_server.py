"""Create a new registered MCP server.

Validates the server is not already registered, then either:

- ``source_type="external"`` (default): loads tools from the remote MCP server
  via the ``McpToolLoader`` to confirm reachability, recording the tool count.
- ``source_type="openapi"``: fetches the OpenAPI spec via the factory, mounts
  an in-process FastMCP server via the ``GeneratedMcpRunnerPort`` (which builds
  the server internally and returns the mounted URL), then counts tools via the
  runner.

In both cases the entry is persisted with the discovered tool count. The
openapi ports (``OpenApiMcpFactory``, ``GeneratedMcpRunnerPort``) are optional
so the external path stays backward-compatible with callers that only inject
the original dependencies.
"""

import logging

from domain.entities.mcp_server_config import McpServerConfig
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import (
    McpConnectionError,
    McpServerAlreadyExistsError,
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


class CreateMcpServerUseCase:
    """Register a new MCP server after validating its reachability.

    Orchestrates: existence check -> (external: tool loading | openapi:
    fetch spec + runner.mount + runner.count_tools) -> persistence.
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

    async def execute(self, entry: RegisteredMcpServer) -> RegisteredMcpServer:
        """Register a new MCP server.

        Args:
            entry: The registered server entry to create.

        Returns:
            The persisted entry with the discovered tool count.

        Raises:
            McpServerAlreadyExistsError: If a server with this name already exists.
            McpServerUnreachableError: If the server cannot be reached or its
                tools cannot be loaded.
            OpenApiFetchError: If the OpenAPI spec cannot be fetched (openapi).
            OpenApiInvalidSpecError: If the OpenAPI spec is invalid (openapi).
        """
        if await self._store.exists(entry.name):
            raise McpServerAlreadyExistsError(
                ErrorMessage.MCP_SERVER_ALREADY_EXISTS.format(name=entry.name)
            )

        if entry.source_type == "openapi":
            return await self._create_openapi(entry)

        config = McpServerConfig(
            name=entry.name,
            transport=entry.transport,
            url=entry.url,
            headers=entry.headers,
            env=entry.env,
            auth_token=entry.auth_token,
        )

        if self._tool_loader is not None:
            try:
                tools = await self._tool_loader.load_tools([config])
                tool_count = len(tools)
            except (McpConnectionError, McpToolLoadError) as e:
                raise McpServerUnreachableError(
                    ErrorMessage.MCP_SERVER_UNREACHABLE.format(error=e)
                ) from e
        else:
            tool_count = 0

        saved = entry.model_copy(update={"tool_count": tool_count})
        await self._store.save(saved)

        logger.info(LogMessage.MCP_SERVER_CREATED_UC, saved.name, tool_count)
        return saved

    async def _create_openapi(self, entry: RegisteredMcpServer) -> RegisteredMcpServer:
        """Fetch spec + runner.mount + runner.count_tools, then persist.

        Args:
            entry: The registered server entry (``source_type="openapi"``).

        Returns:
            The persisted entry with the discovered tool count and mounted URL.

        Raises:
            OpenApiFetchError: If the spec cannot be fetched.
            OpenApiInvalidSpecError: If the spec is invalid.
        """
        assert self._openapi_factory is not None  # noqa: S101 — openapi path requires the factory
        assert self._runner is not None  # noqa: S101 — openapi path requires the runner

        spec = await self._openapi_factory.fetch_spec(
            entry.openapi_url or "", headers=entry.headers
        )
        mounted_url = await self._runner.mount(entry.name, spec, headers=entry.headers)
        tool_count = await self._runner.count_tools(entry.name)

        saved = entry.model_copy(
            update={
                "tool_count": tool_count,
                "url": mounted_url,
                "source_type": "openapi",
                "openapi_url": entry.openapi_url,
            }
        )
        await self._store.save(saved)

        logger.info(LogMessage.MCP_SERVER_CREATED_UC, saved.name, tool_count)
        return saved
