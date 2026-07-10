"""API key security for REST and MCP layers.

Provides two components:

- ``ComposableAgentsSecurity`` — FastAPI dependency that validates the
  ``X-API-Key`` header against a configured master key. When the master key
  is empty, authentication is disabled (useful for local dev/test).

- ``McpApiKeyMiddleware`` — FastMCP middleware that guards ``tools/call``
  and ``tools/list`` by checking the same ``X-API-Key`` header extracted
  from the underlying HTTP request via ``get_http_headers``.
"""

import logging
import secrets

from fastapi import Depends
from fastapi.security import APIKeyHeader
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext

from domain.errors.messages import ErrorMessage
from domain.errors.security import InvalidApiKeyError

logger = logging.getLogger(__name__)


class ComposableAgentsSecurity:
    """Validates incoming API keys against the configured master key.

    The class is instantiated once at the composition root with the master
    key and its ``verify_api_key`` method is injected as a FastAPI dependency
    on protected routers. When the master key is empty the check is bypassed
    (useful for local dev/test).
    """

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def __init__(self, master_key: str) -> None:
        self.master_key = master_key

    def verify_api_key(
        self,
        api_key_header: str | None = Depends(api_key_header),
    ) -> str:
        """Validate the ``X-API-Key`` header against ``master_key``.

        Args:
            api_key_header: The value of the ``X-API-Key`` request header.

        Returns:
            The validated API key string, or an empty string when auth is
            disabled (no master key configured).

        Raises:
            InvalidApiKeyError: If the header is missing or does not match
                the master key.
        """
        if not self.master_key:
            logger.warning(ErrorMessage.API_KEY_DISABLED)
            return ""
        if not api_key_header:
            raise InvalidApiKeyError(ErrorMessage.API_KEY_EMPTY)
        if not secrets.compare_digest(api_key_header, self.master_key):
            raise InvalidApiKeyError(ErrorMessage.API_KEY_UNAUTHORIZED)
        return api_key_header


class McpApiKeyMiddleware(Middleware):
    """FastMCP middleware that enforces the same ``X-API-Key`` on MCP tools.

    Guards both ``tools/call`` and ``tools/list`` so that unauthenticated
    clients can neither execute nor discover the available tools. When the
    master key is empty, authentication is bypassed (dev/test mode).
    """

    def __init__(self, master_key: str) -> None:
        self._master_key = master_key

    async def _check_api_key(self) -> None:
        """Raise ``ToolError`` if the request lacks a valid API key."""
        if not self._master_key:
            return
        headers = get_http_headers() or {}
        api_key = headers.get("x-api-key")
        if not api_key or not secrets.compare_digest(api_key, self._master_key):
            raise ToolError("Invalid or missing API key")

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Guard tool execution with the API key check."""
        await self._check_api_key()
        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Guard tool discovery with the API key check."""
        await self._check_api_key()
        return await call_next(context)
