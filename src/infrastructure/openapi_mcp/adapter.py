"""Infrastructure adapter: build an in-process FastMCP server from an OpenAPI spec.

Implements :class:`OpenApiMcpFactory` using ``httpx`` (fetch + AsyncClient) and
``fastmcp.FastMCP.from_openapi`` (server build). Kept in the infrastructure
layer so the domain never imports httpx or fastmcp.

Validation rules (``fetch_spec``):
- HTTP errors (connect/timeout) -> :class:`OpenApiFetchError`.
- OpenAPI 3.x docs are passed through unchanged.
- Swagger 2.0 docs are converted to OpenAPI 3.0 in-process (offline, pure
  Python) before being returned — see :mod:`swagger2_converter`.
- Anything else (no version key, ``openapi: "2.x"``, unsupported version)
  -> :class:`OpenApiInvalidSpecError`.
- ``content`` larger than 5 MB -> :class:`OpenApiInvalidSpecError`.
"""

import logging
from typing import Any

import httpx
from fastmcp import FastMCP

from domain.errors.mcp import OpenApiFetchError, OpenApiInvalidSpecError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory
from infrastructure.openapi_mcp.swagger2_converter import convert_swagger2_to_openapi3

logger = logging.getLogger(__name__)

#: Maximum accepted spec size in bytes (5 MB).
_MAX_SPEC_BYTES = 5 * 1024 * 1024

#: httpx timeout for fetching an OpenAPI document (seconds).
_FETCH_TIMEOUT = 15.0


class FastMcpOpenApiFactory(OpenApiMcpFactory):
    """Build in-process FastMCP servers from remote OpenAPI documents.

    Uses ``httpx.get`` to fetch the spec and ``httpx.AsyncClient`` +
    ``FastMCP.from_openapi`` to build the server. All HTTP errors are wrapped
    into domain errors (:class:`OpenApiFetchError` /
    :class:`OpenApiInvalidSpecError`) so callers never see httpx exceptions.
    """

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
                ``openapi`` key, is not OpenAPI 3.x, or exceeds 5 MB.
        """
        try:
            response = httpx.get(spec_url, headers=headers, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        except httpx.HTTPError as e:
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_FETCH_FAILED, spec_url, e)
            raise OpenApiFetchError(
                ErrorMessage.OPENAPI_FETCH_FAILED.format(url=spec_url, error=e)
            ) from e

        # HTTP status check — non-2xx means the spec is not available.
        if response.status_code >= 400:
            reason = f"HTTP {response.status_code} from {spec_url}"
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_FETCH_FAILED, spec_url, reason)
            raise OpenApiFetchError(
                ErrorMessage.OPENAPI_FETCH_FAILED.format(url=spec_url, error=reason)
            )

        # Size guard before parsing.
        content = getattr(response, "content", b"")
        if len(content) > _MAX_SPEC_BYTES:
            reason = f"spec exceeds {_MAX_SPEC_BYTES} bytes"
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_INVALID, spec_url, reason)
            raise OpenApiInvalidSpecError(
                ErrorMessage.OPENAPI_INVALID_SPEC.format(url=spec_url, reason=reason)
            )

        try:
            spec = response.json()
        except Exception as e:
            reason = f"response is not valid JSON: {e}"
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_INVALID, spec_url, reason)
            raise OpenApiInvalidSpecError(
                ErrorMessage.OPENAPI_INVALID_SPEC.format(url=spec_url, reason=reason)
            ) from e

        return _normalise_spec_version(spec, spec_url)

    async def build_mcp(self, spec: dict, base_url: str, headers: dict[str, str]) -> Any:
        """Build an in-process FastMCP server from an OpenAPI document.

        The upstream base URL is taken from ``spec["servers"][0]["url"]``.
        ``base_url`` is only used as a fallback when a server entry exists but
        its ``url`` is empty. The spec MUST declare at least one server — an
        absent ``servers`` entry raises :class:`OpenApiInvalidSpecError` even
        when ``base_url`` is provided (the upstream base URL is part of the
        contract advertised by the spec, not guessed from a fallback).

        Args:
            spec: The parsed OpenAPI document.
            base_url: Fallback upstream base URL (only used to fill an empty
                server URL; does NOT replace a missing ``servers`` entry).
            headers: Upstream auth headers forwarded to the AsyncClient.

        Returns:
            A FastMCP server instance exposing ``http_app(path="/")``.

        Raises:
            OpenApiInvalidSpecError: If the spec has no ``servers`` entry.
        """
        servers = spec.get("servers") or []
        if not servers:
            reason = "spec has no 'servers' entry"
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_INVALID, "<build>", reason)
            raise OpenApiInvalidSpecError(
                ErrorMessage.OPENAPI_INVALID_SPEC.format(url="<build>", reason=reason)
            )

        upstream_base_url = servers[0].get("url") or base_url
        client = httpx.AsyncClient(base_url=upstream_base_url, headers=headers)

        # Derive a readable server name from the spec title (best-effort).
        info = spec.get("info") or {}
        name = info.get("title") or "openapi-server"

        mcp = FastMCP.from_openapi(openapi_spec=spec, client=client, name=name)
        logger.info(LogMessage.MCP_OPENAPI_MCP_BUILT, name, -1)
        return mcp

    async def count_tools(self, mcp: Any) -> int:
        """Count the tools exposed by a built FastMCP server (dry-run helper).

        Uses the public ``mcp.list_tools()`` API (NOT the private
        ``_tool_manager``) so the contract survives FastMCP internals churn.

        Args:
            mcp: A FastMCP server instance returned by ``build_mcp``.

        Returns:
            The number of tools the server exposes.
        """
        tools = await mcp.list_tools()  # type: ignore[attr-defined]
        return len(tools)


def _normalise_spec_version(spec: dict, spec_url: str) -> dict:
    """Return an OpenAPI 3.x document, converting Swagger 2.0 if needed.

    Args:
        spec: The parsed JSON document.
        spec_url: The URL the document was fetched from (for error messages).

    Returns:
        An OpenAPI 3.x dict (pass-through or converted from 2.0).

    Raises:
        OpenApiInvalidSpecError: If the document is neither OpenAPI 3.x nor
            Swagger 2.0, or if the 2.0 -> 3.0 conversion fails.
    """
    openapi_version = spec.get("openapi", "")
    if isinstance(openapi_version, str) and openapi_version.startswith("3."):
        logger.info(LogMessage.MCP_OPENAPI_SPEC_FETCHED, spec_url, -1)
        return spec

    if spec.get("swagger") == "2.0":
        try:
            converted = convert_swagger2_to_openapi3(spec)
        except ValueError as e:
            reason = f"Swagger 2.0 -> 3.0 conversion failed: {e}"
            logger.warning(LogMessage.MCP_OPENAPI_SPEC_INVALID, spec_url, reason)
            raise OpenApiInvalidSpecError(
                ErrorMessage.OPENAPI_INVALID_SPEC.format(url=spec_url, reason=reason)
            ) from e
        logger.info(LogMessage.MCP_OPENAPI_SPEC_FETCHED, spec_url, -1)
        return converted

    reason = (
        f"unsupported spec version (got openapi={openapi_version!r}, "
        f"swagger={spec.get('swagger')!r}; only OpenAPI 3.x and Swagger 2.0 are supported)"
    )
    logger.warning(LogMessage.MCP_OPENAPI_SPEC_INVALID, spec_url, reason)
    raise OpenApiInvalidSpecError(
        ErrorMessage.OPENAPI_INVALID_SPEC.format(url=spec_url, reason=reason)
    )
