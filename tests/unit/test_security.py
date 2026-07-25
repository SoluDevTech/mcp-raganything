"""Tests for the API key security feature.

Covers three layers:
- ``ComposableAgentsSecurity.verify_api_key`` (FastAPI dependency, unit tests)
- REST integration through a minimal FastAPI app over httpx ``ASGITransport``
- ``McpApiKeyMiddleware`` guarding FastMCP tools through the in-memory transport

Note on the MCP layer: when middleware raises ``ToolError`` from
``on_call_tool`` the FastMCP client re-raises ``fastmcp.exceptions.ToolError``.
However, errors raised from ``on_list_tools`` are converted by the MCP SDK's
``_handle_request`` into a JSON-RPC error response, so the client surfaces
``mcp.shared.exceptions.McpError`` (the message is still preserved) for the
``list_tools`` path.
"""

import logging
from unittest.mock import patch

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports.memory import FastMCPTransport
from fastmcp.exceptions import ToolError
from httpx import ASGITransport, AsyncClient
from mcp.shared.exceptions import McpError

from domain.errors.security import InvalidApiKeyError
from security import ComposableAgentsSecurity, McpApiKeyMiddleware

# Shared master key used by the REST and MCP integration tests.
_MASTER_KEY = "super-secret-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app(master_key: str) -> FastAPI:
    """Build a minimal FastAPI app with a public and a protected route.

    The protected route is guarded by ``ComposableAgentsSecurity.verify_api_key``
    and a registered ``InvalidApiKeyError`` handler that mirrors the real
    application's response shape (``{"detail": ...}`` with the error's
    ``status_code``).
    """
    security = ComposableAgentsSecurity(master_key=master_key)
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Public unauthenticated health check."""
        return {"status": "ok"}

    protected = APIRouter(dependencies=[Depends(security.verify_api_key)])

    @protected.get("/protected")
    async def protected_route() -> list:
        """Route reachable only with a valid X-API-Key header."""
        return []

    app.include_router(protected)

    @app.exception_handler(InvalidApiKeyError)
    async def _invalid_api_key_handler(
        _request: Request, exc: InvalidApiKeyError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code), content={"detail": exc.detail}
        )

    return app


def _build_mcp_server(master_key: str) -> FastMCP:
    """Build a FastMCP server guarded by ``McpApiKeyMiddleware`` with one tool."""
    server = FastMCP("TestServer")
    server.add_middleware(McpApiKeyMiddleware(master_key=master_key))

    @server.tool()
    async def echo(msg: str = "hello") -> str:
        """Echo its argument back to the caller."""
        return msg

    return server


def _build_mcp_client(server: FastMCP) -> Client:
    """Build an in-memory FastMCP client connected to ``server``."""
    return Client(transport=FastMCPTransport(mcp=server))


# ---------------------------------------------------------------------------
# 1. verify_api_key unit tests
# ---------------------------------------------------------------------------


class TestVerifyApiKey:
    """Direct unit tests for ``ComposableAgentsSecurity.verify_api_key``."""

    def test_verify_api_key_disabled_when_master_key_empty(self, caplog) -> None:
        """Should return an empty string and warn when the master key is disabled."""
        # Arrange
        caplog.set_level(logging.WARNING, logger="security")
        security = ComposableAgentsSecurity(master_key="")

        # Act
        result = security.verify_api_key(api_key_header="irrelevant")

        # Assert
        assert result == ""
        assert any(
            record.name == "security"
            and "Auth by Api key is disabled" in record.message
            for record in caplog.records
        )

    def test_verify_api_key_returns_key_when_valid(self) -> None:
        """Should return the provided key when it matches the master key."""
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)

        # Act
        result = security.verify_api_key(api_key_header=_MASTER_KEY)

        # Assert
        assert result == _MASTER_KEY

    def test_verify_api_key_raises_when_missing(self) -> None:
        """Should raise InvalidApiKeyError when no API key is provided."""
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)

        # Act & Assert
        with pytest.raises(InvalidApiKeyError, match="The Api Key is empty"):
            security.verify_api_key(api_key_header=None)

    def test_verify_api_key_raises_when_invalid(self) -> None:
        """Should raise InvalidApiKeyError when the key does not match."""
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)

        # Act & Assert
        with pytest.raises(
            InvalidApiKeyError, match="The Api Key you provided is unauthorized"
        ):
            security.verify_api_key(api_key_header="wrong")

    def test_invalid_api_key_error_status_code_is_401(self) -> None:
        """InvalidApiKeyError should resolve to HTTP 401."""
        # Assert
        assert InvalidApiKeyError().status_code == 401


# ---------------------------------------------------------------------------
# 2. REST integration tests (httpx + ASGITransport)
# ---------------------------------------------------------------------------


class TestApiKeyRestIntegration:
    """Integration tests for the FastAPI API key dependency over HTTP."""

    async def test_health_endpoint_public_without_key(self) -> None:
        """GET /health should be reachable without an API key."""
        # Arrange
        app = _build_app(master_key=_MASTER_KEY)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_protected_route_returns_401_without_key(self) -> None:
        """GET /protected without a key should return 401 with the empty-key detail."""
        # Arrange
        app = _build_app(master_key=_MASTER_KEY)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/protected")

        # Assert
        assert response.status_code == 401
        assert response.json()["detail"] == "The Api Key is empty"

    async def test_protected_route_returns_401_with_invalid_key(self) -> None:
        """GET /protected with a wrong key should return 401 with the unauthorized detail."""
        # Arrange
        app = _build_app(master_key=_MASTER_KEY)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/protected", headers={"X-API-Key": "wrong"})

        # Assert
        assert response.status_code == 401
        assert response.json()["detail"] == "The Api Key you provided is unauthorized"

    async def test_protected_route_accessible_with_valid_key(self) -> None:
        """GET /protected with the master key should return the protected payload."""
        # Arrange
        app = _build_app(master_key=_MASTER_KEY)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/protected", headers={"X-API-Key": _MASTER_KEY}
            )

        # Assert
        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# 3. McpApiKeyMiddleware unit tests (FastMCP in-memory transport)
# ---------------------------------------------------------------------------


class TestMcpApiKeyMiddleware:
    """Tests for ``McpApiKeyMiddleware`` guarding FastMCP tools.

    The in-memory transport does not carry HTTP headers, so ``get_http_headers``
    returns ``{}`` naturally and is mocked where an ``X-API-Key`` header must be
    simulated. It is patched at ``security.get_http_headers`` (the name under
    which the middleware imported the symbol).
    """

    async def test_mcp_call_tool_succeeds_when_auth_disabled(self) -> None:
        """A tool call should succeed when the master key is empty (auth off)."""
        # Arrange — auth disabled short-circuits before reading headers.
        server = _build_mcp_server(master_key="")

        # Act
        async with _build_mcp_client(server) as client:
            result = await client.call_tool("echo", {"msg": "hello"})

        # Assert
        assert result.data == "hello"

    async def test_mcp_call_tool_raises_tool_error_without_key(self) -> None:
        """A tool call without a key should raise ToolError."""
        # Arrange
        server = _build_mcp_server(master_key=_MASTER_KEY)

        # Act & Assert
        with patch("security.get_http_headers", return_value={}):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError, match="Invalid or missing API key"):
                    await client.call_tool("echo", {"msg": "test"})

    async def test_mcp_call_tool_raises_tool_error_with_invalid_key(self) -> None:
        """A tool call with a wrong key should raise ToolError."""
        # Arrange
        server = _build_mcp_server(master_key=_MASTER_KEY)

        # Act & Assert
        with patch("security.get_http_headers", return_value={"x-api-key": "wrong"}):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError, match="Invalid or missing API key"):
                    await client.call_tool("echo", {"msg": "test"})

    async def test_mcp_call_tool_succeeds_with_valid_key(self) -> None:
        """A tool call with the matching key should succeed."""
        # Arrange
        server = _build_mcp_server(master_key=_MASTER_KEY)

        # Act
        with patch(
            "security.get_http_headers", return_value={"x-api-key": _MASTER_KEY}
        ):
            async with _build_mcp_client(server) as client:
                result = await client.call_tool("echo", {"msg": "test"})

        # Assert
        assert result.data == "test"

    async def test_mcp_list_tools_raises_tool_error_without_key(self) -> None:
        """Listing tools without a key should fail with the API key message.

        The middleware raises ``ToolError`` server-side; the MCP SDK converts
        ``list_tools`` exceptions into a JSON-RPC error, so the client surfaces
        ``McpError`` (its message still carries the original ToolError text).
        """
        # Arrange
        server = _build_mcp_server(master_key=_MASTER_KEY)

        # Act & Assert
        with patch("security.get_http_headers", return_value={}):
            async with _build_mcp_client(server) as client:
                with pytest.raises(McpError, match="Invalid or missing API key"):
                    await client.list_tools()
