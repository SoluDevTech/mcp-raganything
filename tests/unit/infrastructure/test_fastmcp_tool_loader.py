"""Tests for FastMcpToolLoader (fastmcp.Client + StreamableHttp/Stdio transports).

This adapter:
1. For each ``McpServerConfig``, builds a ``fastmcp.Client`` over either a
   ``StreamableHttpTransport`` (HTTP, with Bearer auth from ``auth_token``) or
   a ``StdioTransport`` (STDIO, with command/args/env).
2. Resolves ``${VAR}`` placeholders in headers (HTTP) and env (STDIO) via
   ``infrastructure.env_utils.resolve_env_vars_in_dict``.
3. Uses ``async with Client(transport, timeout=...) as client: tools = await
   client.list_tools()`` and aggregates tools across all configs.
4. ``close()`` closes every held client (no-op when none).

Only fastmcp classes (``Client``, ``StreamableHttpTransport``,
``StdioTransport``) — external libraries — are mocked at the module import
site; the adapter itself is exercised for real.

Error mapping:
- ``ConnectionError`` raised by ``Client.__aenter__`` -> ``McpConnectionError``.
- Any other exception raised during tool loading -> ``McpToolLoadError``.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from domain.errors.mcp import McpConnectionError, McpToolLoadError
from domain.ports.mcp_tool_loader import McpToolLoader
from infrastructure.mcp.tool_loader import FastMcpToolLoader

# -- Helpers -------------------------------------------------------------------


class _FakeClientAsyncCM:
    """A fake ``async with Client(...) as client`` context manager.

    The patched ``Client`` callable returns one of these instances. The fake
    client exposes ``list_tools`` (AsyncMock) and ``close`` (AsyncMock) so
    tests can both configure return values and assert on close calls.
    """

    def __init__(self, tools: list[Any], aenter_side_effect: BaseException | None = None) -> None:
        self._tools = tools
        self._aenter_side_effect = aenter_side_effect
        self.client = MagicMock()
        self.client.list_tools = AsyncMock(return_value=tools)
        self.client.close = AsyncMock()
        self.transport: Any = None
        self.timeout: float | None = None

    async def __aenter__(self) -> Any:
        if self._aenter_side_effect is not None:
            raise self._aenter_side_effect
        return self.client

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


def _http_config(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    headers: dict[str, str] | None = None,
    auth_token: str | None = "secret-token",
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers=headers if headers is not None else {"Authorization": "Bearer tok"},
        auth_token=auth_token,
    )


def _stdio_config(
    name: str = "fs-server",
    command: str = "npx",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        transport=McpTransportType.STDIO,
        command=command,
        args=args if args is not None else ["-y", "@mcp/fs-server"],
        env=env if env is not None else {"PATH": "/usr/bin"},
    )


def _patched_loader(
    fake_cms: list[_FakeClientAsyncCM],
):
    """Patch Client + transports in ``infrastructure.mcp.tool_loader``.

    ``Client`` is replaced by a callable that returns the next fake CM. The
    transport classes are replaced by plain ``MagicMock`` factories so their
    construction call args can be inspected.
    """

    cm_iter = iter(fake_cms)

    def _client_factory(transport: Any, timeout: float | None = None) -> _FakeClientAsyncCM:
        cm = next(cm_iter)
        cm.transport = transport
        cm.timeout = timeout
        return cm

    return (
        patch("infrastructure.mcp.tool_loader.Client", side_effect=_client_factory),
        patch("infrastructure.mcp.tool_loader.StreamableHttpTransport"),
        patch("infrastructure.mcp.tool_loader.StdioTransport"),
    )


# -- HTTP path -----------------------------------------------------------------


class TestLoadToolsHttp:
    """Tests for FastMcpToolLoader.load_tools (HTTP configs)."""

    @pytest.fixture
    def loader(self) -> FastMcpToolLoader:
        return FastMcpToolLoader(tool_timeout=60.0)

    async def test_load_tools_http_returns_aggregated_tools(self, loader: FastMcpToolLoader) -> None:
        # Arrange — two HTTP configs; mocked clients return 2 and 3 tools.
        cms = [_FakeClientAsyncCM(tools=["t1", "t2"]), _FakeClientAsyncCM(tools=["t3", "t4", "t5"])]
        client_p, http_p, stdio_p = _patched_loader(cms)

        # Act
        with client_p as mock_client, http_p as mock_http, stdio_p as _mock_stdio:
            result = await loader.load_tools([_http_config("a"), _http_config("b")])

        # Assert — tools aggregated, HTTP transport constructed twice, no STDIO.
        assert result == ["t1", "t2", "t3", "t4", "t5"]
        assert mock_http.call_count == 2
        assert _mock_stdio.call_count == 0
        assert mock_client.call_count == 2

    async def test_load_tools_http_no_auth_token_passes_none(self, loader: FastMcpToolLoader) -> None:
        # Arrange — config without auth_token
        cms = [_FakeClientAsyncCM(tools=["t1"])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _http_config(auth_token=None)

        # Act
        with client_p, http_p as mock_http, stdio_p:
            await loader.load_tools([config])

        # Assert — StreamableHttpTransport called with auth=None
        _, kwargs = mock_http.call_args
        assert kwargs.get("auth") is None

    async def test_load_tools_http_empty_string_auth_token_coerced_to_none(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange — config with auth_token="" (empty string from DB / request body).
        # An empty string MUST be coerced to None, otherwise fastmcp creates a
        # BearerAuth with an empty token -> "Bearer " header -> httpx.LocalProtocolError.
        cms = [_FakeClientAsyncCM(tools=["t1"])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _http_config(auth_token="")

        # Act
        with client_p, http_p as mock_http, stdio_p:
            await loader.load_tools([config])

        # Assert — StreamableHttpTransport called with auth=None, not ""
        _, kwargs = mock_http.call_args
        assert kwargs.get("auth") is None

    async def test_load_tools_http_passes_auth_token_as_auth(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange — config with an auth_token; adapter should pass it as ``auth``.
        cms = [_FakeClientAsyncCM(tools=["t1"])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _http_config(auth_token="my-bearer")

        # Act
        with client_p, http_p as mock_http, stdio_p:
            await loader.load_tools([config])

        # Assert — StreamableHttpTransport called with auth="my-bearer".
        _, kwargs = mock_http.call_args
        assert kwargs.get("auth") == "my-bearer"

    async def test_load_tools_http_passes_url_and_resolved_headers(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange
        cms = [_FakeClientAsyncCM(tools=[])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _http_config(url="http://srv:1234/mcp", headers={"X-Custom": "abc"})

        # Act
        with client_p, http_p as mock_http, stdio_p:
            await loader.load_tools([config])

        # Assert
        args, kwargs = mock_http.call_args
        # Accept either positional url or keyword url.
        url = kwargs.get("url") if "url" in kwargs else (args[0] if args else None)
        assert url == "http://srv:1234/mcp"
        assert kwargs.get("headers") == {"X-Custom": "abc"}

    async def test_load_tools_resolves_env_var_placeholders_in_headers(
        self, loader: FastMcpToolLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — header value references ${TOK}; adapter must resolve it.
        monkeypatch.setenv("TOK", "resolved-token")
        cms = [_FakeClientAsyncCM(tools=["t1"])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _http_config(headers={"Authorization": "Bearer ${TOK}"}, auth_token=None)

        # Act
        with client_p, http_p as mock_http, stdio_p:
            await loader.load_tools([config])

        # Assert — placeholder resolved before being passed to transport.
        _, kwargs = mock_http.call_args
        assert kwargs.get("headers") == {"Authorization": "Bearer resolved-token"}


# -- STDIO path ----------------------------------------------------------------


class TestLoadToolsStdio:
    """Tests for FastMcpToolLoader.load_tools (STDIO configs)."""

    @pytest.fixture
    def loader(self) -> FastMcpToolLoader:
        return FastMcpToolLoader(tool_timeout=60.0)

    async def test_load_tools_stdio_constructs_stdio_transport(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange
        cms = [_FakeClientAsyncCM(tools=["s1", "s2"])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _stdio_config(command="npx", args=["-y", "pkg"])

        # Act
        with client_p, http_p as _mock_http, stdio_p as mock_stdio:
            result = await loader.load_tools([config])

        # Assert — StdioTransport used, tools returned, no HTTP transport.
        assert result == ["s1", "s2"]
        assert mock_stdio.call_count == 1
        assert _mock_http.call_count == 0

    async def test_load_tools_stdio_passes_command_and_args(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange
        cms = [_FakeClientAsyncCM(tools=[])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _stdio_config(command="python", args=["-m", "server"])

        # Act
        with client_p, http_p, stdio_p as mock_stdio:
            await loader.load_tools([config])

        # Assert
        args, kwargs = mock_stdio.call_args
        command = kwargs.get("command") if "command" in kwargs else (args[0] if args else None)
        assert command == "python"
        passed_args = kwargs.get("args")
        assert passed_args == ["-m", "server"]

    async def test_load_tools_resolves_env_var_placeholders_in_env_for_stdio(
        self, loader: FastMcpToolLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — env value references ${VAL}; adapter must resolve it.
        monkeypatch.setenv("VAL", "resolved-value")
        cms = [_FakeClientAsyncCM(tools=[])]
        client_p, http_p, stdio_p = _patched_loader(cms)
        config = _stdio_config(env={"X": "${VAL}", "STATIC": "keep"})

        # Act
        with client_p, http_p, stdio_p as mock_stdio:
            await loader.load_tools([config])

        # Assert
        _, kwargs = mock_stdio.call_args
        assert kwargs.get("env") == {"X": "resolved-value", "STATIC": "keep"}


# -- Mixed / edge cases --------------------------------------------------------


class TestLoadToolsMixedAndEdgeCases:
    """Tests for mixed configs and edge cases."""

    @pytest.fixture
    def loader(self) -> FastMcpToolLoader:
        return FastMcpToolLoader(tool_timeout=60.0)

    async def test_load_tools_mixed_http_and_stdio_configs(self, loader: FastMcpToolLoader) -> None:
        # Arrange — first config HTTP, second STDIO
        cms = [_FakeClientAsyncCM(tools=["h1"]), _FakeClientAsyncCM(tools=["s1", "s2"])]
        client_p, http_p, stdio_p = _patched_loader(cms)

        # Act
        with client_p, http_p as mock_http, stdio_p as mock_stdio:
            result = await loader.load_tools([_http_config(), _stdio_config()])

        # Assert — both transports constructed, tools aggregated.
        assert mock_http.call_count == 1
        assert mock_stdio.call_count == 1
        assert result == ["h1", "s1", "s2"]

    async def test_load_tools_empty_configs_returns_empty_list(self, loader: FastMcpToolLoader) -> None:
        # Arrange / Act
        result = await loader.load_tools([])

        # Assert
        assert result == []

    async def test_load_tools_connection_error_raises_mcp_connection_error(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange — fake CM raises ConnectionError on __aenter__.
        cm = _FakeClientAsyncCM(tools=[], aenter_side_effect=ConnectionError("refused"))
        client_p, http_p, stdio_p = _patched_loader([cm])

        # Act / Assert
        with client_p, http_p, stdio_p, pytest.raises(McpConnectionError):
            await loader.load_tools([_http_config()])

    async def test_load_tools_generic_exception_raises_mcp_tool_load_error(
        self, loader: FastMcpToolLoader
    ) -> None:
        # Arrange — fake client.list_tools raises a generic RuntimeError.
        cm = _FakeClientAsyncCM(tools=[])
        cm.client.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
        client_p, http_p, stdio_p = _patched_loader([cm])

        # Act / Assert
        with client_p, http_p, stdio_p, pytest.raises(McpToolLoadError):
            await loader.load_tools([_http_config()])

    async def test_tool_timeout_passed_to_client(self) -> None:
        # Arrange — loader with a non-default timeout.
        loader = FastMcpToolLoader(tool_timeout=12.5)
        cms = [_FakeClientAsyncCM(tools=[])]
        client_p, http_p, stdio_p = _patched_loader(cms)

        # Act
        with client_p as mock_client, http_p, stdio_p:
            await loader.load_tools([_http_config()])

        # Assert — Client constructed with timeout=12.5.
        args, kwargs = mock_client.call_args
        timeout = kwargs.get("timeout") if "timeout" in kwargs else (args[1] if len(args) > 1 else None)
        assert timeout == 12.5


# -- close() -------------------------------------------------------------------


class TestClose:
    """Tests for FastMcpToolLoader.close()."""

    async def test_close_closes_all_clients(self) -> None:
        # Arrange — load tools from 2 configs, then close.
        loader = FastMcpToolLoader(tool_timeout=60.0)
        cms = [_FakeClientAsyncCM(tools=["a"]), _FakeClientAsyncCM(tools=["b"])]
        client_p, http_p, stdio_p = _patched_loader(cms)

        with client_p, http_p, stdio_p:
            await loader.load_tools([_http_config("x"), _http_config("y")])

        # Sanity — two clients were created.
        assert len(cms) == 2

        # Act
        await loader.close()

        # Assert — every client.close awaited exactly once.
        for cm in cms:
            cm.client.close.assert_awaited_once()

        # Second close call is safe (no error, no further close calls).
        await loader.close()
        for cm in cms:
            cm.client.close.assert_awaited_once()

    async def test_close_no_clients_is_noop(self) -> None:
        # Arrange — loader that has never loaded any tools.
        loader = FastMcpToolLoader(tool_timeout=60.0)

        # Act / Assert — close is a no-op, must not raise.
        await loader.close()


# -- port contract -------------------------------------------------------------


class TestFastMcpToolLoaderPortContract:
    """FastMcpToolLoader must implement the McpToolLoader port."""

    def test_is_subclass_of_mcp_tool_loader_port(self) -> None:
        # Arrange / Act / Assert
        assert issubclass(FastMcpToolLoader, McpToolLoader)
