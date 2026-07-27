"""API key + JWT security for REST and MCP layers.

Provides two components:

- ``ComposableAgentsSecurity`` — FastAPI dependency that historically validated
  the ``X-API-Key`` header against a configured master key (``verify_api_key``,
  kept for backward-compat). The new dual-auth dependency ``verify_credentials``
  delegates to an injected :class:`~domain.services.auth.auth_service.AuthService`
  (JWT bearer token + per-user API key lookup), sets the RLS contextvars on
  success, and raises :class:`AuthenticationError` (401) on failure.

- ``McpApiKeyMiddleware`` — FastMCP middleware that guards ``tools/call`` and
  ``tools/list``. Historically checked only the ``X-API-Key`` master key; now
  extended to ALSO accept ``Authorization: Bearer`` (JWT) and per-user API keys
  via the same injected ``AuthService``. The RLS contextvars are set so that
  downstream asyncpg usage in the tool handler is RLS-scoped (contextvars
  propagate within the same asyncio task).

When the master key is empty, the legacy ``verify_api_key`` / ``_check_api_key``
paths disable auth (useful for local dev/test). The dual-auth paths are
engaged only when ``set_auth_service`` has been called from the composition
root (``main.py`` lifespan).
"""

import logging
import secrets

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext

from domain.entities.auth.auth_context import AuthContext
from domain.errors.messages import ErrorMessage
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from infrastructure.database.rls_context import (
    current_auth_method,
    current_credential,
    current_user_id,
)

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "Bearer "


class ComposableAgentsSecurity:
    """Validates incoming credentials (legacy master key or dual JWT/API key).

    The class is instantiated once at the composition root with the master key
    (``AppConfig.API_KEY``). Its ``verify_api_key`` method (legacy, kept for
    backward-compat) validates the ``X-API-Key`` header against the master key.
    When the master key is empty the check is bypassed (dev/test mode).

    The dual-auth ``verify_credentials`` dependency delegates to an injected
    :class:`~domain.services.auth.auth_service.AuthService` (JWT bearer token +
    per-user API key) and sets the RLS contextvars on success. It is engaged
    only after ``set_auth_service`` has been called from the composition root.
    """

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def __init__(self, master_key: str = "") -> None:
        self.master_key = master_key
        # Dual-auth service (JWT + per-user API key). Injected via
        # ``set_auth_service`` from the composition root; ``verify_credentials``
        # raises a clear RuntimeError if it is called before wiring.
        self._auth_service = None

    def set_auth_service(self, auth_service) -> None:
        """Inject the dual-auth :class:`AuthService`.

        Args:
            auth_service: The wired ``AuthService`` (JWT port + API key repo).
        """
        self._auth_service = auth_service

    def has_auth_service(self) -> bool:
        """Return whether a dual-auth :class:`AuthService` is wired."""
        return self._auth_service is not None

    # ------------------------------------------------------------------
    # Legacy: master-key X-API-Key check (backward-compat)
    # ------------------------------------------------------------------

    def verify_api_key(
        self,
        api_key_header: str | None = Depends(api_key_header),
    ) -> str:
        """Validate the ``X-API-Key`` header against ``master_key``.

        Kept for backward-compat with existing tests and the legacy wiring.
        New protected routers should use :meth:`verify_credentials` instead.

        Args:
            api_key_header: The value of the ``X-API-Key`` request header.

        Returns:
            The validated API key string, or an empty string when auth is
            disabled (no master key configured).

        Raises:
            InvalidApiKeyError: If the header is missing or does not match the
                master key.
        """
        if not self.master_key:
            logger.warning(ErrorMessage.API_KEY_DISABLED)
            return ""
        if not api_key_header:
            raise InvalidApiKeyError(ErrorMessage.API_KEY_EMPTY)
        if not secrets.compare_digest(api_key_header, self.master_key):
            raise InvalidApiKeyError(ErrorMessage.API_KEY_UNAUTHORIZED)
        return api_key_header

    # ------------------------------------------------------------------
    # Dual auth: JWT bearer token OR per-user API key
    # ------------------------------------------------------------------

    async def verify_credentials(self, request: Request) -> AuthContext | str:
        """Dual-auth FastAPI dependency: JWT bearer token OR per-user API key.

        Reads the ``Authorization`` and ``X-API-Key`` headers, delegates to the
        injected :class:`AuthService`, raises :class:`AuthenticationError` (401)
        when no credential validates, and otherwise sets the RLS contextvars
        (``current_user_id`` / ``current_credential`` / ``current_auth_method``)
        and returns the resolved :class:`AuthContext`.

        **Fallback when no AuthService is wired** (dev/test, or production
        before the lifespan runs): falls back to the legacy master-key
        ``verify_api_key`` check on the ``X-API-Key`` header. This keeps the
        REST routers working when dual auth is not configured (``LOGTO_URL``
        unset or asyncpg pool not available) and avoids a RuntimeError at
        request time.

        Args:
            request: The incoming FastAPI request (headers are read from it).

        Returns:
            The authenticated :class:`AuthContext` (dual-auth path) or the
            validated API key string / empty string (legacy fallback).

        Raises:
            AuthenticationError: If no credential could be validated (dual-auth
                path, 401).
            InvalidApiKeyError: If the legacy fallback rejects the key (401).
        """
        if self._auth_service is None:
            # Legacy fallback: no dual-auth wired. Use the master-key check.
            return self.verify_api_key(request.headers.get("x-api-key"))

        authorization = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")

        ctx = await self._auth_service.authenticate(
            authorization=authorization,
            api_key=api_key,
        )
        if ctx is None:
            raise AuthenticationError(ErrorMessage.AUTH_INVALID_CREDENTIALS)

        # Wire the RLS contextvars for downstream asyncpg usage (McpRegistryStore).
        current_user_id.set(ctx.user_id)
        current_credential.set(ctx.raw_credential)
        current_auth_method.set(ctx.method)
        return ctx


class McpApiKeyMiddleware(Middleware):
    """FastMCP middleware that enforces auth on MCP tools (master key or dual).

    Historically guarded ``tools/call`` and ``tools/list`` by checking the
    ``X-API-Key`` header against the configured master key. Now extended to
    ALSO accept ``Authorization: Bearer`` (JWT) and per-user API keys when an
    :class:`AuthService` is wired via :meth:`set_auth_service`.

    The RLS contextvars (``current_user_id`` / ``current_credential`` /
    ``current_auth_method``) are set on success so that downstream asyncpg
    usage in the tool handler is RLS-scoped. Contextvars propagate within
    the same asyncio task — the FastMCP request handler runs the middleware
    and the tool in the same task, so the tool sees the values set here.

    When the master key is empty AND no auth service is wired, authentication
    is bypassed (dev/test mode).
    """

    def __init__(self, master_key: str = "") -> None:
        self._master_key = master_key
        self._auth_service = None

    def set_auth_service(self, auth_service) -> None:
        """Inject the dual-auth :class:`AuthService`.

        Args:
            auth_service: The wired ``AuthService`` (JWT port + API key repo).
        """
        self._auth_service = auth_service

    def has_auth_service(self) -> bool:
        """Return whether a dual-auth :class:`AuthService` is wired."""
        return self._auth_service is not None

    async def _check_api_key(self) -> None:
        """Raise ``ToolError`` if the request lacks valid credentials.

        Resolution order:
        1. If a dual-auth :class:`AuthService` is wired → delegate to it
           (Bearer JWT precedence, else X-API-Key lookup). Set the RLS
           contextvars on success.
        2. Else (legacy master-key mode) → if master key empty, bypass; else
           require the ``X-API-Key`` header to match the master key.
        """
        headers = get_http_headers() or {}
        authorization = headers.get("authorization")
        api_key = headers.get("x-api-key")

        # 1. Dual-auth path (JWT + per-user API key).
        if self._auth_service is not None:
            ctx = await self._auth_service.authenticate(
                authorization=authorization,
                api_key=api_key,
            )
            if ctx is None:
                raise ToolError("Invalid or missing credentials")
            # Set the RLS contextvars for downstream asyncpg usage in the tool.
            current_user_id.set(ctx.user_id)
            current_credential.set(ctx.raw_credential)
            current_auth_method.set(ctx.method)
            return

        # 2. Legacy master-key path.
        if not self._master_key:
            return
        if not api_key or not secrets.compare_digest(api_key, self._master_key):
            raise ToolError("Invalid or missing API key")

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Guard tool execution with the credential check."""
        await self._check_api_key()
        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Guard tool discovery with the credential check."""
        await self._check_api_key()
        return await call_next(context)
