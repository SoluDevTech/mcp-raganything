"""Tests for FastMcpOpenApiFactory (httpx fetch + FastMCP.from_openapi build).

This adapter:
1. fetches an OpenAPI spec from a URL via httpx (timeout, 5MB max, OpenAPI 3.x);
2. builds a FastMCP server from the spec via ``FastMCP.from_openapi`` using an
   ``httpx.AsyncClient`` configured with the upstream base URL + headers;
3. exposes ``count_tools`` to validate the built server (used by the dry-run
   validate use case). ``count_tools`` MUST use ``mcp.list_tools()`` (the
   public API), NOT ``_tool_manager``.

Only httpx + FastMCP (external libraries) are mocked; the factory adapter
itself is exercised for real.

These tests are written TDD-Red: ``domain.ports.openapi_mcp_factory`` and
``infrastructure.openapi_mcp.adapter`` do not exist yet, so importing them
raises ``ImportError`` and every test fails until the implementation lands.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from domain.errors.mcp import OpenApiFetchError, OpenApiInvalidSpecError
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory
from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory


def _valid_spec() -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Petstore", "version": "1.0.0"},
        "paths": {"/pets": {"get": {"operationId": "listPets"}}},
        "servers": [{"url": "https://petstore.example.com"}],
    }


# -- fetch_spec ----------------------------------------------------------------


class TestFetchSpec:
    """Tests for OpenApiMcpFactory.fetch_spec (httpx GET)."""

    @pytest.fixture
    def factory(self) -> FastMcpOpenApiFactory:
        return FastMcpOpenApiFactory()

    async def test_returns_spec_dict_on_successful_get(self, factory: FastMcpOpenApiFactory):
        # Arrange
        spec = _valid_spec()
        mock_response = MagicMock()
        mock_response.json.return_value = spec
        mock_response.content = b"x" * 10
        mock_response.status_code = 2000
        mock_response.status_code = 200

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)) as mock_get:
            # Act
            result = await factory.fetch_spec("https://petstore.example.com/openapi.json", headers={})

        # Assert
        assert result == spec
        mock_get.assert_called_once()

    async def test_raises_openapi_fetch_error_on_connection_error(self, factory: FastMcpOpenApiFactory):
        # Arrange
        with patch.object(httpx, "get", new=MagicMock(side_effect=httpx.ConnectError("refused"))):
            # Act & Assert
            with pytest.raises(OpenApiFetchError):
                await factory.fetch_spec("https://unreachable.example.com/openapi.json", headers={})

    async def test_raises_openapi_fetch_error_on_timeout(self, factory: FastMcpOpenApiFactory):
        # Arrange
        with patch.object(httpx, "get", new=MagicMock(side_effect=httpx.TimeoutException("timeout"))):
            # Act & Assert
            with pytest.raises(OpenApiFetchError):
                await factory.fetch_spec("https://slow.example.com/openapi.json", headers={})

    async def test_raises_openapi_invalid_spec_error_when_openapi_key_missing(
        self, factory: FastMcpOpenApiFactory
    ):
        # Arrange — a doc with neither 'openapi' nor 'swagger: 2.0' version key
        bad_spec = {"info": {"title": "x", "version": "1"}, "paths": {}}
        mock_response = MagicMock()
        mock_response.json.return_value = bad_spec
        mock_response.content = b"x" * 10
        mock_response.status_code = 200

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)):
            # Act & Assert
            with pytest.raises(OpenApiInvalidSpecError):
                await factory.fetch_spec("https://example.com/openapi.json", headers={})

    async def test_converts_swagger2_spec_to_openapi3(self, factory: FastMcpOpenApiFactory):
        # Arrange — a Swagger 2.0 doc that fetch_spec should convert to 3.0
        swagger2_spec = {
            "swagger": "2.0",
            "info": {"title": "Petstore", "version": "1.0.0"},
            "host": "petstore.example.com",
            "basePath": "/v2",
            "schemes": ["https"],
            "paths": {
                "/pet": {
                    "post": {
                        "operationId": "addPet",
                        "parameters": [{"in": "body", "name": "body", "schema": {"$ref": "#/definitions/Pet"}}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
            "definitions": {"Pet": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = swagger2_spec
        mock_response.content = b"x" * 10
        mock_response.status_code = 200

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)):
            # Act
            result = await factory.fetch_spec("https://petstore.example.com/v2/swagger.json", headers={})

        # Assert — converted to OpenAPI 3.0
        assert result["openapi"].startswith("3.")
        assert result["servers"] == [{"url": "https://petstore.example.com/v2"}]
        assert "components" in result and "schemas" in result["components"]
        assert "Pet" in result["components"]["schemas"]
        # $ref rewritten
        body_schema = result["paths"]["/pet"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert body_schema == {"$ref": "#/components/schemas/Pet"}

    async def test_raises_openapi_invalid_spec_error_when_version_is_2x(self, factory: FastMcpOpenApiFactory):
        # Arrange — fake "openapi" key but a 2.x version
        bad_spec = {"openapi": "2.0.0", "info": {"title": "x", "version": "1"}, "paths": {}}
        mock_response = MagicMock()
        mock_response.json.return_value = bad_spec
        mock_response.content = b"x" * 10
        mock_response.status_code = 200

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)):
            # Act & Assert
            with pytest.raises(OpenApiInvalidSpecError):
                await factory.fetch_spec("https://example.com/openapi.json", headers={})

    async def test_raises_openapi_invalid_spec_error_when_spec_exceeds_5mb(
        self, factory: FastMcpOpenApiFactory
    ):
        # Arrange — content > 5MB
        mock_response = MagicMock()
        mock_response.json.return_value = _valid_spec()
        mock_response.status_code = 200
        mock_response.content = b"x" * (5 * 1024 * 1024 + 1)

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)):
            # Act & Assert
            with pytest.raises(OpenApiInvalidSpecError):
                await factory.fetch_spec("https://huge.example.com/openapi.json", headers={})

    async def test_passes_headers_to_httpx_get(self, factory: FastMcpOpenApiFactory):
        # Arrange
        spec = _valid_spec()
        mock_response = MagicMock()
        mock_response.json.return_value = spec
        mock_response.content = b"x" * 10
        mock_response.status_code = 200
        upstream_headers = {"Authorization": "Bearer upstream-token"}

        with patch.object(httpx, "get", new=MagicMock(return_value=mock_response)) as mock_get:
            # Act
            await factory.fetch_spec("https://petstore.example.com/openapi.json", headers=upstream_headers)

        # Assert
        _, kwargs = mock_get.call_args
        assert kwargs.get("headers") == upstream_headers


# -- build_mcp -----------------------------------------------------------------


class TestBuildMcp:
    """Tests for OpenApiMcpFactory.build_mcp (FastMCP.from_openapi)."""

    @pytest.fixture
    def factory(self) -> FastMcpOpenApiFactory:
        return FastMcpOpenApiFactory()

    async def test_returns_object_with_http_app_method(self, factory: FastMcpOpenApiFactory):
        # Arrange
        spec = _valid_spec()
        base_url = "https://petstore.example.com"
        upstream_headers = {"Authorization": "Bearer upstream-token"}
        fake_mcp = MagicMock()
        fake_mcp.http_app = MagicMock(return_value=MagicMock())

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)) as mock_from_openapi,
            patch.object(httpx, "AsyncClient", new=MagicMock()) as mock_async_client_cls,
        ):
            # Act
            result = await factory.build_mcp(spec, base_url, upstream_headers)

        # Assert
        assert result is fake_mcp
        assert hasattr(result, "http_app")
        mock_from_openapi.assert_called_once()
        mock_async_client_cls.assert_called_once()

    async def test_calls_from_openapi_with_spec_and_client(self, factory: FastMcpOpenApiFactory):
        # Arrange
        spec = _valid_spec()
        base_url = "https://petstore.example.com"
        upstream_headers = {"Authorization": "Bearer upstream-token"}
        fake_client = MagicMock()
        fake_mcp = MagicMock()

        def _from_openapi(*, openapi_spec, client, name):
            return fake_mcp

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(side_effect=_from_openapi)) as mock_from_openapi,
            patch.object(httpx, "AsyncClient", new=MagicMock(return_value=fake_client)) as mock_async_client_cls,
        ):
            # Act
            await factory.build_mcp(spec, base_url, upstream_headers)

        # Assert — httpx.AsyncClient built with base_url + headers
        _, kwargs = mock_async_client_cls.call_args
        assert kwargs["base_url"] == base_url
        assert kwargs["headers"] == upstream_headers

        # Assert — FastMCP.from_openapi called with openapi_spec + client
        call_kwargs = mock_from_openapi.call_args.kwargs
        assert call_kwargs["openapi_spec"] is spec
        assert call_kwargs["client"] is fake_client

    async def test_raises_openapi_invalid_spec_error_when_no_servers(self, factory: FastMcpOpenApiFactory):
        # Arrange — spec without servers
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "x", "version": "1"},
            "paths": {},
        }
        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock()),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act & Assert
            with pytest.raises(OpenApiInvalidSpecError):
                await factory.build_mcp(spec, "https://fallback.example.com", {})


# -- count_tools ----------------------------------------------------------------


class TestCountTools:
    """Tests for OpenApiMcpFactory.count_tools (dry-run validation helper).

    count_tools MUST use ``mcp.list_tools()`` (the public API), NOT
    ``_tool_manager``.
    """

    @pytest.fixture
    def factory(self) -> FastMcpOpenApiFactory:
        return FastMcpOpenApiFactory()

    async def test_returns_number_of_tools_from_mcp_server(self, factory: FastMcpOpenApiFactory):
        # Arrange
        fake_mcp = MagicMock()
        fake_mcp.list_tools = AsyncMock(return_value=["a", "b", "c"])

        # Act
        count = await factory.count_tools(fake_mcp)

        # Assert
        assert count == 3
        fake_mcp.list_tools.assert_awaited_once()

    async def test_returns_zero_when_no_tools(self, factory: FastMcpOpenApiFactory):
        # Arrange
        fake_mcp = MagicMock()
        fake_mcp.list_tools = AsyncMock(return_value=[])

        # Act
        count = await factory.count_tools(fake_mcp)

        # Assert
        assert count == 0


# -- port contract -------------------------------------------------------------


class TestOpenApiMcpFactoryPortContract:
    """FastMcpOpenApiFactory must implement the OpenApiMcpFactory port."""

    def test_is_subclass_of_openapi_mcp_factory_port(self):
        # Arrange / Act / Assert
        assert issubclass(FastMcpOpenApiFactory, OpenApiMcpFactory)
