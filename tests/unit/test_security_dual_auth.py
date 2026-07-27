"""Tests for dual-auth (JWT + per-user API key) in ``security.py``.

Covers:
- ``ComposableAgentsSecurity.verify_credentials`` HTTP dependency: Bearer valid
  → 200, X-API-Key valid → 200, no credential → 401, wrong credential → 401,
  and the RLS contextvars are set on success.
- ``McpApiKeyMiddleware`` extended to ALSO accept ``Authorization: Bearer``
  (JWT) — set contextvars so downstream asyncpg usage is RLS-scoped.
- ``set_auth_service`` / ``has_auth_service`` lifecycle.
- ``AuthenticationError`` mapped to HTTP 401.

The in-memory FastMCP transport does not carry HTTP headers, so
``get_http_headers`` is patched where needed (same pattern as
``test_security.py``).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports.memory import FastMCPTransport
from fastmcp.exceptions import ToolError
from httpx import ASGITransport, AsyncClient

from domain.entities.auth.auth_context import AuthContext
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from infrastructure.database.rls_context import (
    current_auth_method,
    current_credential,
    current_user_id,
)
from security import ComposableAgentsSecurity, McpApiKeyMiddleware

_MASTER_KEY = "super-secret-key"


def _make_auth_service_mock(
    *,
    jwt_user_id: str | None = None,
    api_key_user_id: str | None = None,
) -> AsyncMock:
    """Return a mock AuthService that resolves JWT/api-key to AuthContext or None."""
    svc = AsyncMock()

    async def _authenticate(authorization: str | None, api_key: str | None):
        if authorization and authorization.startswith("Bearer "):
            if jwt_user_id is not None:
                return AuthContext(
                    user_id=jwt_user_id, method="jwt", raw_credential=authorization[7:]
                )
            return None
        if api_key is not None:
            if api_key_user_id is not None:
                return AuthContext(
                    user_id=api_key_user_id, method="api_key", raw_credential=api_key
                )
            return None
        return None

    svc.authenticate.side_effect = _authenticate
    return svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app_with_auth(auth_service: AsyncMock) -> FastAPI:
    """Build a minimal FastAPI app guarded by ``verify_credentials``."""
    security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
    security.set_auth_service(auth_service)  # type: ignore[arg-type]
    app = FastAPI()

    protected = APIRouter()

    @protected.get("/protected")
    async def protected_route(_ctx: AuthContext = Depends(security.verify_credentials)):
        # Read the contextvars here, INSIDE the request task, so the test can
        # assert they were set (they don't propagate back to the caller context).
        return {
            "user_id": _ctx.user_id,
            "method": _ctx.method,
            "ctxvar_user_id": current_user_id.get(),
            "ctxvar_method": current_auth_method.get(),
            "ctxvar_credential": current_credential.get(),
        }

    app.include_router(protected)

    @app.exception_handler(AuthenticationError)
    async def _auth_err(_req: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code), content={"detail": exc.detail}
        )

    @app.exception_handler(InvalidApiKeyError)
    async def _api_key_err(_req: Request, exc: InvalidApiKeyError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code), content={"detail": exc.detail}
        )

    return app


# ---------------------------------------------------------------------------
# 1. set_auth_service / has_auth_service lifecycle
# ---------------------------------------------------------------------------


class TestSetAuthService:
    """Tests for ``set_auth_service`` / ``has_auth_service``."""

    def test_has_auth_service_returns_false_before_set(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)

        # Act & Assert
        assert security.has_auth_service() is False

    def test_has_auth_service_returns_true_after_set(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        svc = _make_auth_service_mock(jwt_user_id="u1")

        # Act
        security.set_auth_service(svc)  # type: ignore[arg-type]

        # Assert
        assert security.has_auth_service() is True


# ---------------------------------------------------------------------------
# 2. verify_credentials (HTTP)
# ---------------------------------------------------------------------------


class TestVerifyCredentialsHttp:
    """Integration tests for ``verify_credentials`` over HTTP."""

    async def test_bearer_token_valid_returns_200_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(jwt_user_id="user-jwt-123")
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/protected", headers={"Authorization": "Bearer my-jwt"}
            )

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "user-jwt-123"
        assert body["method"] == "jwt"
        # Contextvars are read INSIDE the request task (see the route body) so
        # they don't need to propagate back to the test context.
        assert body["ctxvar_user_id"] == "user-jwt-123"
        assert body["ctxvar_credential"] == "my-jwt"
        assert body["ctxvar_method"] == "jwt"

    async def test_x_api_key_valid_returns_200_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(api_key_user_id="user-api-456")
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": "cpk_xxx"})

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "user-api-456"
        assert body["method"] == "api_key"
        assert body["ctxvar_user_id"] == "user-api-456"
        assert body["ctxvar_method"] == "api_key"

    async def test_no_credentials_returns_401(self):
        # Arrange — svc returns None for any credential
        svc = _make_auth_service_mock()
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected")

        # Assert
        assert resp.status_code == 401

    async def test_invalid_credentials_returns_401(self):
        # Arrange — svc resolves neither JWT nor API key
        svc = _make_auth_service_mock()
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/protected", headers={"Authorization": "Bearer bad-jwt"}
            )

        # Assert
        assert resp.status_code == 401

    async def test_authentication_error_status_code_is_401(self):
        # Act & Assert
        assert AuthenticationError().status_code == 401

    async def test_verify_credentials_falls_back_to_master_key_when_no_auth_service(
        self,
    ):
        """When no AuthService is wired, verify_credentials uses the master key.

        This covers the dev/test path and the startup window before the
        lifespan wires the AuthService: ``verify_credentials`` delegates to
        the legacy ``verify_api_key`` so the REST routers keep working.
        """
        # Arrange — no set_auth_service call
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        app = FastAPI()
        protected = APIRouter()

        @protected.get("/protected")
        async def protected_route(_=Depends(security.verify_credentials)):
            return {"ok": True}

        app.include_router(protected)

        @app.exception_handler(InvalidApiKeyError)
        async def _err(_req: Request, exc: InvalidApiKeyError) -> JSONResponse:
            return JSONResponse(
                status_code=int(exc.status_code), content={"detail": exc.detail}
            )

        # Act & Assert — valid master key → 200
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": _MASTER_KEY})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Act & Assert — wrong key → 401
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 3. McpApiKeyMiddleware dual auth (FastMCP in-memory transport)
# ---------------------------------------------------------------------------


def _build_mcp_server_with_auth(auth_service: AsyncMock) -> FastMCP:
    """Build a FastMCP server guarded by ``McpApiKeyMiddleware`` with dual auth.

    The echo tool returns the contextvar value seen INSIDE the tool body so the
    test can verify that contextvars set by the middleware propagate to the
    tool handler (same asyncio task, same context).
    """
    server = FastMCP("TestServer")
    middleware = McpApiKeyMiddleware(master_key=_MASTER_KEY)
    middleware.set_auth_service(auth_service)  # type: ignore[arg-type]
    server.add_middleware(middleware)

    @server.tool()
    async def echo(msg: str = "hello") -> str:
        """Echo its argument + the RLS contextvar user_id seen by the tool."""
        return (
            f"{msg}|user_id={current_user_id.get()}|method={current_auth_method.get()}"
        )

    return server


def _build_mcp_client(server: FastMCP) -> Client:
    return Client(transport=FastMCPTransport(mcp=server))


class TestMcpApiKeyMiddlewareDualAuth:
    """Tests for ``McpApiKeyMiddleware`` accepting Bearer + X-API-Key."""

    async def test_bearer_token_valid_succeeds_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(jwt_user_id="user-jwt-1")
        server = _build_mcp_server_with_auth(svc)

        # Act
        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer my-jwt"},
        ):
            async with _build_mcp_client(server) as client:
                result = await client.call_tool("echo", {"msg": "test"})

        # Assert — the tool reads the contextvar INSIDE its own body (same
        # asyncio task as the middleware), so the value set by the middleware
        # is visible there. We assert it via the tool's return value, NOT by
        # reading the contextvar in the test (which runs in a different context).
        assert result.data == "test|user_id=user-jwt-1|method=jwt"

    async def test_x_api_key_valid_succeeds_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(api_key_user_id="user-api-2")
        server = _build_mcp_server_with_auth(svc)

        # Act
        with patch(
            "security.get_http_headers",
            return_value={"x-api-key": "cpk_yyy"},
        ):
            async with _build_mcp_client(server) as client:
                result = await client.call_tool("echo", {"msg": "test"})

        # Assert
        assert result.data == "test|user_id=user-api-2|method=api_key"

    async def test_no_credentials_raises_tool_error(self):
        # Arrange
        svc = _make_auth_service_mock()
        server = _build_mcp_server_with_auth(svc)

        # Act & Assert
        with patch("security.get_http_headers", return_value={}):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("echo", {"msg": "test"})

    async def test_invalid_credentials_raises_tool_error(self):
        # Arrange
        svc = _make_auth_service_mock()
        server = _build_mcp_server_with_auth(svc)

        # Act & Assert
        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer bad-jwt"},
        ):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("echo", {"msg": "test"})


# ---------------------------------------------------------------------------
# 4. Backward-compat: verify_api_key still works without auth service
# ---------------------------------------------------------------------------


class TestVerifyApiKeyBackwardCompat:
    """``verify_api_key`` is preserved for backward-compat (no auth service wired)."""

    async def test_verify_api_key_still_works_without_auth_service(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        # no set_auth_service call — verify_api_key must still validate the master key

        # Act & Assert
        assert security.verify_api_key(api_key_header=_MASTER_KEY) == _MASTER_KEY
        with pytest.raises(InvalidApiKeyError):
            security.verify_api_key(api_key_header="wrong")
