"""Tests for MCP server rehydration at application startup.

At startup, the application reads all openapi entries from the store and mounts
each via the ``GeneratedMcpRunner``, gracefully skipping failures so a single
broken spec never blocks application startup.

Mocked external boundaries:
- McpServerRegistryStore (store.list_all) -> AsyncMock
- GeneratedMcpRunnerPort (runner.mount) -> AsyncMock

Internal component (the rehydration orchestrator) is exercised for real.
"""

import logging
from unittest.mock import AsyncMock

import pytest

from domain.entities.mcp_server_config import McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from infrastructure.openapi_mcp.rehydration import rehydrate_openapi_servers


def _openapi_entry(name: str = "petstore") -> RegisteredMcpServer:
    from datetime import UTC, datetime

    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url="http://raganything-api:8000/generated/petstore/mcp",
        headers={"Authorization": "Bearer tok"},
        env={},
        auth_token=None,
        tool_count=3,
        source_type="openapi",
        openapi_url="https://petstore.example.com/openapi.json",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _external_entry(name: str = "web-search") -> RegisteredMcpServer:
    from datetime import UTC, datetime

    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url="http://localhost:3001/mcp",
        headers={},
        env={},
        auth_token=None,
        tool_count=2,
        source_type="external",
        openapi_url=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_store() -> AsyncMock:
    return AsyncMock(spec=McpServerRegistryStore)


@pytest.fixture
def mock_runner() -> AsyncMock:
    return AsyncMock(spec=GeneratedMcpRunnerPort)


class TestRehydrateOpenapiServers:
    """Tests for rehydrate_openapi_servers (startup rehydration)."""

    async def test_mounts_each_openapi_entry_via_runner(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        # Arrange — two openapi entries in the store
        mock_store.list_all.return_value = [
            _openapi_entry("server-a"),
            _openapi_entry("server-b"),
        ]
        mock_runner.mount.return_value = "http://raganything-api:8000/generated/server-a/mcp"

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert — runner.mount called for each openapi entry
        assert mock_runner.mount.await_count == 2

    async def test_skips_external_entries(self, mock_store: AsyncMock, mock_runner: AsyncMock):
        # Arrange — mix of external and openapi entries
        mock_store.list_all.return_value = [
            _external_entry("web-search"),
            _openapi_entry("petstore"),
        ]
        mock_runner.mount.return_value = "http://raganything-api:8000/generated/petstore/mcp"

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert — only the openapi entry is mounted
        assert mock_runner.mount.await_count == 1
        mount_args = mock_runner.mount.await_args_list[0]
        assert mount_args.args[0] == "petstore"

    async def test_skips_entries_that_fail_to_mount(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        """A single broken spec should not block startup — failures are skipped."""
        # Arrange — first mount fails, second succeeds
        mock_store.list_all.return_value = [
            _openapi_entry("broken"),
            _openapi_entry("good"),
        ]
        mock_runner.mount.side_effect = [
            Exception("spec fetch failed"),
            "http://raganything-api:8000/generated/good/mcp",
        ]

        # Act — should not raise
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert — both were attempted, the failure was swallowed
        assert mock_runner.mount.await_count == 2

    async def test_logs_error_summary_when_entries_skipped(
        self, mock_store: AsyncMock, mock_runner: AsyncMock, caplog
    ):
        """When at least one entry is skipped, an ERROR-level summary is emitted."""
        # Arrange — one broken entry, one good entry
        mock_store.list_all.return_value = [
            _openapi_entry("broken"),
            _openapi_entry("good"),
        ]
        mock_runner.mount.side_effect = [
            Exception("spec fetch failed"),
            "http://raganything-api:8000/generated/good/mcp",
        ]

        # Act
        with caplog.at_level("ERROR", logger="infrastructure.openapi_mcp.rehydration"):
            await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert — a single ERROR summary line referencing the skipped count
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("skipped 1" in r.getMessage() for r in error_records)

    async def test_does_not_log_error_summary_when_none_skipped(
        self, mock_store: AsyncMock, mock_runner: AsyncMock, caplog
    ):
        """No ERROR summary when rehydration succeeds fully."""
        mock_store.list_all.return_value = [_openapi_entry("good")]
        mock_runner.mount.return_value = "http://raganything-api:8000/generated/good/mcp"

        with caplog.at_level("ERROR", logger="infrastructure.openapi_mcp.rehydration"):
            await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records == []

    async def test_does_not_call_mount_when_no_openapi_entries(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        # Arrange — only external entries
        mock_store.list_all.return_value = [
            _external_entry("web-search"),
            _external_entry("another"),
        ]

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert
        mock_runner.mount.assert_not_awaited()

    async def test_does_not_call_mount_when_store_is_empty(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        # Arrange
        mock_store.list_all.return_value = []

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert
        mock_runner.mount.assert_not_awaited()

    async def test_calls_store_list_all_once(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        # Arrange
        mock_store.list_all.return_value = []

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert
        mock_store.list_all.assert_awaited_once()

    async def test_passes_entry_headers_to_mount(
        self, mock_store: AsyncMock, mock_runner: AsyncMock
    ):
        # Arrange
        entry = _openapi_entry("petstore")
        entry_headers = {"Authorization": "Bearer upstream-token"}

        # Override headers on the entry
        from datetime import UTC, datetime

        entry = RegisteredMcpServer(
            name="petstore",
            transport=McpTransportType.HTTP,
            url="http://raganything-api:8000/generated/petstore/mcp",
            headers=entry_headers,
            env={},
            auth_token=None,
            tool_count=3,
            source_type="openapi",
            openapi_url="https://petstore.example.com/openapi.json",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        mock_store.list_all.return_value = [entry]
        mock_runner.mount.return_value = "http://raganything-api:8000/generated/petstore/mcp"

        # Act
        await rehydrate_openapi_servers(store=mock_store, runner=mock_runner)

        # Assert — mount receives the entry's headers
        mount_kwargs = mock_runner.mount.await_args.kwargs
        # headers can be passed as kwarg or positional arg[2]; check both
        if "headers" in mount_kwargs:
            assert mount_kwargs["headers"] == entry_headers
        else:
            mount_args = mock_runner.mount.await_args.args
            assert len(mount_args) >= 3 and mount_args[2] == entry_headers
