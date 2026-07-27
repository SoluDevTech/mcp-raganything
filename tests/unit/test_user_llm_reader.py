"""Tests for the ``AsyncpgUserLlmReader`` adapter (TDD Red phase).

The reader is the READ-ONLY mcp-raganything side of the shared
``user_llm_settings`` table (owned and created by composable-agents' Alembic).
It exposes two methods:

- ``get(user_id) -> UserLlmSettings | None`` — returns the user's settings with
  a **masked** API key (decrypt via ``FernetSecretCipher`` then mask
  ``first3...last3``; never the full plaintext).
- ``get_decrypted(user_id) -> tuple[str, str] | None`` — returns
  ``(base_url, api_key_plaintext)`` for the per-user LLM factory.

The ``user_llm_settings`` table is FORCE RLS-protected by composable-agents,
but the reader resolves credentials AFTER auth (the ``current_user_id`` is
already known). The reader still emits ``SET LOCAL row_security = off`` before
the SELECT — a privileged read so the FORCE RLS policy does not hide the row
(the RLS policy on the table filters by ``user_id``, and we are filtering by
``user_id`` ourselves, but the GUC ``app.user_id`` may not be set on this
connection; bypassing RLS is simpler and safe for a read-only PK lookup).

The asyncpg pool / connection is mocked so no real database is needed. The
``FernetSecretCipher`` is real (with a test-generated key) so the mask logic
is exercised end-to-end.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

from domain.entities.user_llm_settings import UserLlmSettings
from infrastructure.database.user_llm_reader import AsyncpgUserLlmReader
from infrastructure.security.crypto import FernetSecretCipher


@pytest.fixture
def fernet_key() -> str:
    """A valid Fernet key for the test cipher."""
    return Fernet.generate_key().decode()


@pytest.fixture
def cipher(fernet_key: str) -> FernetSecretCipher:
    return FernetSecretCipher(fernet_key)


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)


async def _encrypt(cipher: FernetSecretCipher, plaintext: str) -> str:
    return await cipher.encrypt(plaintext)


def _row(
    user_id: str = "user-1",
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_encrypted: str = "ENC",
    now: datetime | None = None,
) -> dict:
    """Return a fake asyncpg record row for user_llm_settings."""
    ts = now or datetime.now(UTC)
    return {
        "user_id": user_id,
        "provider": provider,
        "base_url": base_url,
        "api_key_encrypted": api_key_encrypted,
        "created_at": ts,
        "updated_at": ts,
    }


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
def reader(mock_pool: MagicMock, cipher: FernetSecretCipher) -> AsyncpgUserLlmReader:
    return AsyncpgUserLlmReader(pool=mock_pool, cipher=cipher)


class TestGet:
    """Tests for ``AsyncpgUserLlmReader.get`` (masked API key)."""

    async def test_returns_masked_user_llm_settings_when_row_exists(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
        cipher: FernetSecretCipher,
        now: datetime,
    ) -> None:
        # Arrange — encrypt a real plaintext so the mask logic is exercised
        api_key = "sk-or-v1-abcdefghijklmnop"
        encrypted = await _encrypt(cipher, api_key)
        mock_conn.fetchrow.return_value = _row(
            user_id="user-A",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_encrypted=encrypted,
            now=now,
        )

        # Act
        result = await reader.get("user-A")

        # Assert
        assert result is not None
        assert isinstance(result, UserLlmSettings)
        assert result.user_id == "user-A"
        assert result.provider == "openrouter"
        assert result.base_url == "https://openrouter.ai/api/v1"
        # Masked key: first 3 + ... + last 3, never the full plaintext
        assert result.api_key_masked is not None
        assert result.api_key_masked == "sk-...nop"
        assert api_key not in result.api_key_masked

    async def test_returns_none_when_no_row(
        self, reader: AsyncpgUserLlmReader, mock_conn: AsyncMock
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        result = await reader.get("user-missing")

        # Assert
        assert result is None

    async def test_emits_set_local_row_security_off_before_select(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.get("user-A")

        # Assert — at least one execute with row_security = off
        execute_calls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert any(
            "row_security" in sql.lower() and "off" in sql.lower()
            for sql in execute_calls
        ), f"row_security=off not emitted; calls={execute_calls}"

    async def test_row_security_off_emitted_before_select(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
    ) -> None:
        """The row_security=off must be issued BEFORE the SELECT."""
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.get("user-A")

        # Assert — order of awaited calls: execute (row_security) then fetchrow
        all_calls = mock_conn.method_calls
        execute_idx = next(
            (i for i, c in enumerate(all_calls) if c[0] == "execute"), None
        )
        fetchrow_idx = next(
            (i for i, c in enumerate(all_calls) if c[0] == "fetchrow"), None
        )
        assert execute_idx is not None and fetchrow_idx is not None
        assert execute_idx < fetchrow_idx

    async def test_select_filters_by_user_id(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.get("user-A")

        # Assert — the SELECT query filters by user_id
        fetchrow_call = mock_conn.fetchrow.await_args
        sql = fetchrow_call.args[0].lower()
        assert "user_id" in sql, f"user_id filter missing; sql={sql}"
        params = fetchrow_call.args[1:]
        assert "user-A" in params, f"user_id not passed as param; params={params}"

    async def test_select_targets_user_llm_settings_table(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.get("user-A")

        # Assert — the FROM clause targets user_llm_settings
        fetchrow_call = mock_conn.fetchrow.await_args
        sql = fetchrow_call.args[0].lower()
        assert "from user_llm_settings" in sql, f"wrong table; sql={sql}"


class TestGetDecrypted:
    """Tests for ``AsyncpgUserLlmReader.get_decrypted``."""

    async def test_returns_base_url_and_plaintext_when_row_exists(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
        cipher: FernetSecretCipher,
        now: datetime,
    ) -> None:
        # Arrange
        api_key = "sk-or-v1-secret-key-123456"
        encrypted = await _encrypt(cipher, api_key)
        mock_conn.fetchrow.return_value = _row(
            user_id="user-A",
            base_url="https://openrouter.ai/api/v1",
            api_key_encrypted=encrypted,
            now=now,
        )

        # Act
        result = await reader.get_decrypted("user-A")

        # Assert
        assert result is not None
        base_url, plaintext = result
        assert base_url == "https://openrouter.ai/api/v1"
        assert plaintext == api_key

    async def test_returns_none_when_no_row(
        self, reader: AsyncpgUserLlmReader, mock_conn: AsyncMock
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        result = await reader.get_decrypted("user-missing")

        # Assert
        assert result is None

    async def test_emits_set_local_row_security_off_before_select(
        self,
        reader: AsyncpgUserLlmReader,
        mock_conn: AsyncMock,
    ) -> None:
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act
        await reader.get_decrypted("user-A")

        # Assert
        execute_calls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert any(
            "row_security" in sql.lower() and "off" in sql.lower()
            for sql in execute_calls
        ), f"row_security=off not emitted; calls={execute_calls}"
