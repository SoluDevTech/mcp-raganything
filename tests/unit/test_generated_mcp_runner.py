"""Tests for the GeneratedMcpRunner adapter.

The runner (``infrastructure.openapi_mcp.runner``) mounts in-process FastMCP
servers built from OpenAPI specs into the running FastAPI app under
``/generated/{name}/mcp``. It owns the ASGI lifecycle via an
``AsyncExitStack`` (lifespan startup on mount, shutdown on unmount) and applies
middleware to each mounted server.

Mocked external boundaries:
- ``fastmcp.FastMCP.from_openapi`` (server build)
- ``httpx.AsyncClient`` (upstream HTTP client)
- The FastAPI/Starlette ``app.mount`` call (ASGI mount side-effect)

Internal component (the runner adapter) is exercised for real.

These tests are written TDD-Red: ``infrastructure.openapi_mcp.runner`` and
``domain.ports.generated_mcp_runner`` do not exist yet, so importing them
raises ``ImportError`` and every test fails until the implementation lands.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from domain.errors.mcp import McpAlreadyMountedError, OpenApiInvalidSpecError
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from infrastructure.openapi_mcp.runner import GeneratedMcpRunner


def _valid_spec() -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Petstore", "version": "1.0.0"},
        "paths": {"/pets": {"get": {"operationId": "listPets"}}},
        "servers": [{"url": "https://petstore.example.com"}],
    }


def _fake_mcp() -> MagicMock:
    """A fake FastMCP instance with http_app, lifespan, list_tools, add_middleware."""
    mcp = MagicMock()
    mcp.http_app = MagicMock(return_value=MagicMock())
    # lifespan must be an async context manager
    lifespan_cm = AsyncMock()
    lifespan_cm.__aenter__ = AsyncMock(return_value=None)
    lifespan_cm.__aexit__ = AsyncMock(return_value=None)
    mcp.http_app.return_value.lifespan = MagicMock(return_value=lifespan_cm)
    mcp.list_tools = AsyncMock(return_value=["tool-a", "tool-b"])
    mcp.add_middleware = MagicMock()
    return mcp


@pytest.fixture
def mock_app() -> MagicMock:
    app = MagicMock()
    app.mount = MagicMock()
    return app


@pytest.fixture
def runner(mock_app: MagicMock) -> GeneratedMcpRunner:
    return GeneratedMcpRunner(
        app=mock_app,
        base_url="http://raganything-api:8000",
    )


# -- mount ---------------------------------------------------------------------


class TestMount:
    """Tests for GeneratedMcpRunner.mount."""

    async def test_builds_mcp_via_fastmcp_from_openapi(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)) as mock_from_openapi,
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act
            await runner.mount("petstore", spec, headers={})

        # Assert
        mock_from_openapi.assert_called_once()
        call_kwargs = mock_from_openapi.call_args.kwargs
        assert call_kwargs["openapi_spec"] is spec

    async def test_creates_httpx_async_client_with_upstream_url_and_headers(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        upstream_headers = {"Authorization": "Bearer upstream-token"}

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()) as mock_async_client_cls,
        ):
            # Act
            await runner.mount("petstore", spec, headers=upstream_headers)

        # Assert
        _, kwargs = mock_async_client_cls.call_args
        assert kwargs["base_url"] == "https://petstore.example.com"
        assert kwargs["headers"] == upstream_headers

    async def test_mounts_http_app_at_generated_path(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        fake_http_app = fake_mcp.http_app.return_value

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act
            await runner.mount("petstore", spec, headers={})

        # Assert — app.mount called with /generated/petstore/mcp
        mock_app.mount.assert_called_once()
        mount_args = mock_app.mount.call_args.args
        assert mount_args[0] == "/generated/petstore/mcp"
        assert mount_args[1] is fake_http_app

    async def test_returns_absolute_mounted_url(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act
            url = await runner.mount("petstore", spec, headers={})

        # Assert
        assert url == "http://raganything-api:8000/generated/petstore/mcp"

    async def test_runs_lifespan_via_async_exit_stack(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        lifespan_cm = fake_mcp.http_app.return_value.lifespan.return_value

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act
            await runner.mount("petstore", spec, headers={})

        # Assert — lifespan was entered
        lifespan_cm.__aenter__.assert_awaited_once()

    async def test_raises_409_when_already_mounted(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act — mount once, then try again
            await runner.mount("petstore", spec, headers={})
            with pytest.raises(McpAlreadyMountedError):
                await runner.mount("petstore", spec, headers={})

    async def test_raises_invalid_spec_when_no_servers(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange — spec without servers
        spec = {"openapi": "3.0.3", "info": {"title": "x", "version": "1"}, "paths": {}}

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=_fake_mcp())),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act & Assert
            with pytest.raises(OpenApiInvalidSpecError):
                await runner.mount("petstore", spec, headers={})

    async def test_applies_middleware_to_built_mcp(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        middleware = MagicMock()

        runner_with_mw = GeneratedMcpRunner(
            app=mock_app,
            base_url="http://raganything-api:8000",
            middleware=[middleware],
        )

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act
            await runner_with_mw.mount("petstore", spec, headers={})

        # Assert — add_middleware was called with the middleware
        fake_mcp.add_middleware.assert_called_once_with(middleware)


# -- unmount -------------------------------------------------------------------


class TestUnmount:
    """Tests for GeneratedMcpRunner.unmount."""

    async def test_closes_lifespan_stack_on_unmount(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        lifespan_cm = fake_mcp.http_app.return_value.lifespan.return_value

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            await runner.mount("petstore", spec, headers={})

            # Act
            await runner.unmount("petstore")

        # Assert — lifespan was exited
        lifespan_cm.__aexit__.assert_awaited_once()

    async def test_is_noop_when_not_mounted(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Act — unmount something that was never mounted; should not raise
        await runner.unmount("never-mounted")

        # Assert — no exception, nothing to verify beyond no raise

    async def test_allows_remounting_after_unmount(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=_fake_mcp())),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            # Act — mount, unmount, then mount again (should not raise 409)
            await runner.mount("petstore", spec, headers={})
            await runner.unmount("petstore")
            url = await runner.mount("petstore", spec, headers={})

        # Assert
        assert url == "http://raganything-api:8000/generated/petstore/mcp"


# -- count_tools ---------------------------------------------------------------


class TestCountTools:
    """Tests for GeneratedMcpRunner.count_tools (must use mcp.list_tools())."""

    async def test_returns_tool_count_for_mounted_server(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        fake_mcp.list_tools = AsyncMock(return_value=["a", "b", "c"])

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            await runner.mount("petstore", spec, headers={})

            # Act
            count = await runner.count_tools("petstore")

        # Assert
        assert count == 3
        fake_mcp.list_tools.assert_awaited_once()

    async def test_returns_zero_when_no_tools(
        self, runner: GeneratedMcpRunner, mock_app: MagicMock
    ):
        # Arrange
        spec = _valid_spec()
        fake_mcp = _fake_mcp()
        fake_mcp.list_tools = AsyncMock(return_value=[])

        with (
            patch("fastmcp.FastMCP.from_openapi", new=MagicMock(return_value=fake_mcp)),
            patch.object(httpx, "AsyncClient", new=MagicMock()),
        ):
            await runner.mount("petstore", spec, headers={})

            # Act
            count = await runner.count_tools("petstore")

        # Assert
        assert count == 0


# -- port contract -------------------------------------------------------------


class TestGeneratedMcpRunnerPortContract:
    """GeneratedMcpRunner must implement the GeneratedMcpRunnerPort."""

    def test_is_subclass_of_generated_mcp_runner_port(self):
        assert issubclass(GeneratedMcpRunner, GeneratedMcpRunnerPort)
