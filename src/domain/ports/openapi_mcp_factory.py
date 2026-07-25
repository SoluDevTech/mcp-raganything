"""Port for building an MCP server from an OpenAPI spec.

The :class:`OpenApiMcpFactory` abstracts the two-step transformation that turns
a remote OpenAPI document into an in-process FastMCP server:

1. ``fetch_spec`` — download the OpenAPI document from a URL via httpx
   (timeout, size cap, OpenAPI 3.x validation; Swagger 2.0 docs are converted
   to OpenAPI 3.0 in-process before being returned).
2. ``build_mcp`` — instantiate a FastMCP server from the spec using an
   ``httpx.AsyncClient`` configured with the upstream base URL + auth headers.

A dry-run helper ``count_tools`` lets the validate use case inspect a built
server without mounting or persisting it.
"""

from abc import ABC, abstractmethod
from typing import Any


class OpenApiMcpFactory(ABC):
    """Outbound port: turn an OpenAPI document into an in-process FastMCP server.

    Implementations live in the infrastructure layer (e.g.
    :class:`FastMcpOpenApiFactory`) and depend on httpx + fastmcp. The domain
    layer only sees this abstraction, so it stays free of those imports.
    """

    @abstractmethod
    async def fetch_spec(self, spec_url: str, headers: dict[str, str]) -> dict:
        """Fetch and validate an OpenAPI document from ``spec_url``.

        Args:
            spec_url: The URL of the OpenAPI document (JSON).
            headers: Upstream auth headers to send with the request.

        Returns:
            The parsed OpenAPI document as a ``dict``.

        Raises:
            OpenApiFetchError: If the document cannot be downloaded.
            OpenApiInvalidSpecError: If the document is missing the
                ``openapi`` key, is not OpenAPI 3.x, or exceeds the size cap.
        """
        ...

    @abstractmethod
    async def build_mcp(self, spec: dict, base_url: str, headers: dict[str, str]) -> Any:
        """Build an in-process FastMCP server from an OpenAPI document.

        The upstream base URL is derived from ``spec["servers"][0]["url"]``
        unless ``base_url`` is provided; an ``httpx.AsyncClient`` is created
        with that base URL and the given ``headers`` so FastMCP can proxy
        upstream calls with the right auth.

        Args:
            spec: The parsed OpenAPI document.
            base_url: Fallback upstream base URL (used to fill an empty server
                URL; does NOT replace a missing ``servers`` entry).
            headers: Upstream auth headers forwarded to the AsyncClient.

        Returns:
            A FastMCP server instance exposing ``http_app(path="/")``.

        Raises:
            OpenApiInvalidSpecError: If the spec has no ``servers`` entry.
        """
        ...

    @abstractmethod
    async def count_tools(self, mcp: Any) -> int:
        """Count the tools exposed by a built FastMCP server (dry-run helper).

        Args:
            mcp: A FastMCP server instance returned by ``build_mcp``.

        Returns:
            The number of tools the server exposes.
        """
        ...
