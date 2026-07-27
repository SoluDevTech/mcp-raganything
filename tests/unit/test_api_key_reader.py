"""Tests for the ``AsyncpgApiKeyReader`` adapter.

The reader is the READ-ONLY mcp-raganything side of the shared ``api_keys``
table (owned by composable-agents). It exposes a single method,
``find_active_by_hash(key_hash) -> tuple[str, str] | None``, returning the
``(user_id, key_id)`` tuple for the active (``revoked_at IS NULL``) key
matching the SHA-256 hash, or ``None`` when no active key matches.

The ``api_keys`` table is FORCE RLS-protected by composable-agents, but at
auth time the GUC ``app.user_id`` is NOT set yet (we are reading the table TO
determine the user). The reader therefore emits ``SET LOCAL row_security =
off`` on its connection before the SELECT — a privileged auth operation.

The asyncpg pool / connection is mocked so no real database is needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from infrastructure.database.api_key_reader import AsyncpgApiKeyReader


def _row(user_id: str = "user-1", key_id: str = "key-id-1") -> dict:
    """Return a fake asyncpg record row."""
    return {"user_id": user_id, "id": key_id}


@pytest.fixture
def mock_conn() -> AsyncMock:
    """A mock asyncpg.Connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@pytest.fixture
def mock_pool(mock_conn: AsyncMock) -> MagicMock:
    """A mock asyncpg.Pool whose acquire() yields a mock connection."""
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_conn
    cm.__aexit__.return_value = None
    pool.acquire.return_value = cm
    return pool


@pytest.fixture
def reader(mock_pool: MagicMock) -> AsyncpgApiKeyReader:
    return AsyncpgApiKeyReader(pool=mock_pool)


class TestFindActiveByHash:
    """Tests for ``AsyncpgApiKeyReader.find_active_by_hash``."""

    async def test_returns_user_id_and_key_id_when_active_match(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = _row(user_id="user-abc", key_id="key-123")

        # Act
        result = await reader.find_active_by_hash("a" * 64)

        # Assert
        assert result is not None
        assert result == ("user-abc", "key-123")

    async def test_returns_none_when_no_row(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        result = await reader.find_active_by_hash("0" * 64)

        # Assert
        assert result is None

    async def test_emits_set_local_row_security_off_before_select(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.find_active_by_hash("a" * 64)

        # Assert — at least one execute with row_security = off
        execute_calls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert any(
            "row_security" in sql.lower() and "off" in sql.lower()
            for sql in execute_calls
        ), f"row_security=off not emitted; calls={execute_calls}"

    async def test_select_filters_on_revoked_at_is_null(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.find_active_by_hash("a" * 64)

        # Assert — the SELECT query filters revoked_at IS NULL (case-insensitive)
        fetchrow_call = mock_conn.fetchrow.await_args
        sql = fetchrow_call.args[0].lower()
        assert "revoked_at is null" in sql, f"revoked_at filter missing; sql={sql}"

    async def test_select_filters_on_key_hash(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = None
        key_hash = "deadbeef" * 8  # 64-char hex

        # Act
        await reader.find_active_by_hash(key_hash)

        # Assert — the SELECT query passes the hash as a parameter
        fetchrow_call = mock_conn.fetchrow.await_args
        params = fetchrow_call.args[1:]
        assert key_hash in params, f"hash not passed as parameter; params={params}"

    async def test_row_security_off_emitted_before_select(
        self, reader: AsyncpgApiKeyReader, mock_conn: AsyncMock
    ):
        """The row_security=off must be issued BEFORE the SELECT (privileged auth read)."""
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.find_active_by_hash("a" * 64)

        # Assert — order of awaited calls: execute (row_security) then fetchrow
        all_calls = mock_conn.method_calls
        # ``mock_calls`` entries are (name, args, kwargs) tuples; the name for
        # ``await conn.execute(...)`` is just ``"execute"`` (the await is recorded
        # separately on the await_calls list, not in the method name).
        execute_idx = next(
            (i for i, c in enumerate(all_calls) if c[0] == "execute"), None
        )
        fetchrow_idx = next(
            (i for i, c in enumerate(all_calls) if c[0] == "fetchrow"), None
        )
        assert execute_idx is not None and fetchrow_idx is not None
        assert execute_idx < fetchrow_idx, (
            f"row_security=off must be set before the SELECT; "
            f"execute_idx={execute_idx} fetchrow_idx={fetchrow_idx}"
        )
