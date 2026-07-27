"""Tests for per-user isolation in ``McpRegistryStore``.

The store now filters its queries by ``current_user_id`` when the RLS
contextvar is set (and emits ``set_config('app.user_id', ...)`` on the
acquired asyncpg connection so PostgreSQL RLS policies also enforce
isolation server-side). When no user is set (legacy/tests), the store
behaves as before — no user_id filter, no GUC.

Cases:
- save under user A → list under user A includes the entry
- list under user B excludes the entry saved under user A
- get under user B raises McpServerNotFoundError (or returns None equivalent)
- RLS GUC (app.user_id) emitted on the connection when current_user_id is set
- no GUC emitted when current_user_id is None (legacy path)
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.entities.mcp_server_config import McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerAlreadyExistsError, McpServerNotFoundError
from domain.ports.secret_cipher import SecretCipher
from infrastructure.database.mcp_registry_store import McpRegistryStore
from infrastructure.database.rls_context import current_user_id


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    auth_token: str | None = "secret-token",
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers={"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token=auth_token,
        tool_count=3,
        source_type="external",
        openapi_url=None,
        created_at=_now(),
        updated_at=_now(),
    )


def _row(
    name: str = "web-search",
    transport: str = "http",
    url: str = "http://localhost:3001/mcp",
    user_id: str = "",
    headers_encrypted: str = "ENC({})",
    env_encrypted: str = "ENC({})",
    auth_token_encrypted: str | None = None,
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
) -> dict:
    return {
        "name": name,
        "transport": transport,
        "url": url,
        "user_id": user_id,
        "headers_encrypted": headers_encrypted,
        "env_encrypted": env_encrypted,
        "auth_token_encrypted": auth_token_encrypted,
        "tool_count": tool_count,
        "source_type": source_type,
        "openapi_url": openapi_url,
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def mock_cipher() -> AsyncMock:
    """Mock SecretCipher that encrypts by prefixing and decrypts by stripping."""
    cipher = AsyncMock(spec=SecretCipher)

    async def _encrypt(plaintext: str) -> str:
        return f"ENC({plaintext})"

    async def _decrypt(ciphertext: str) -> str:
        if ciphertext.startswith("ENC(") and ciphertext.endswith(")"):
            return ciphertext[4:-1]
        return ciphertext

    cipher.encrypt.side_effect = _encrypt
    cipher.decrypt.side_effect = _decrypt
    return cipher


@pytest.fixture
def mock_conn() -> AsyncMock:
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    return conn


@pytest.fixture
def mock_pool(mock_conn: AsyncMock) -> MagicMock:
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_conn
    cm.__aexit__.return_value = None
    pool.acquire.return_value = cm
    return pool


@pytest.fixture
def store(mock_pool: MagicMock, mock_cipher: AsyncMock) -> McpRegistryStore:
    return McpRegistryStore(pool=mock_pool, cipher=mock_cipher)


# -- RLS GUC emission ----------------------------------------------------------


class TestRlsGucEmission:
    """The store must emit the app.user_id GUC when current_user_id is set."""

    async def test_save_emits_app_user_id_guc_when_user_set(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        token = current_user_id.set("user-A")
        try:
            # Act
            await store.save(_entry(name="srv-A"))

            # Assert — at least one execute with set_config('app.user_id', ...)
            # and the user_id is passed as the bound parameter ($1).
            execute_calls = mock_conn.execute.await_args_list
            guc_call = next(
                (
                    c
                    for c in execute_calls
                    if "set_config" in c.args[0].lower()
                    and "app.user_id" in c.args[0].lower()
                ),
                None,
            )
            assert guc_call is not None, (
                f"app.user_id GUC missing; calls={[c.args[0] for c in execute_calls]}"
            )
            # The user_id is the second positional arg (after the SQL string).
            guc_params = guc_call.args[1:]
            assert "user-A" in guc_params, (
                f"user_id not passed as param to set_config; params={guc_params}"
            )
        finally:
            current_user_id.reset(token)

    async def test_save_no_guc_when_user_none(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — no user set (contextvar default None)
        # Act
        await store.save(_entry(name="srv-legacy"))

        # Assert — no app.user_id GUC
        execute_calls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert not any("app.user_id" in sql.lower() for sql in execute_calls), (
            f"unexpected app.user_id GUC; calls={execute_calls}"
        )


# -- save user_id persisted ----------------------------------------------------


class TestSaveUserIsolation:
    """save() must persist current_user_id on the row."""

    async def test_save_persists_current_user_id(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        token = current_user_id.set("user-A")
        try:
            # Act
            await store.save(_entry(name="srv-A"))

            # Assert — the INSERT/UPSERT was called with user_id in the params
            # The execute call's args[1:] are the bound parameters.
            execute_call = mock_conn.execute.await_args
            params = execute_call.args[1:]
            assert "user-A" in params, f"user_id not passed to INSERT; params={params}"
        finally:
            current_user_id.reset(token)


# -- save cross-user conflict guard -------------------------------------------


class TestSaveCrossUserConflict:
    """save() must refuse to overwrite another user's server (data-loss guard).

    Regression guard for a Critical bug: the ``INSERT ... ON CONFLICT (name)
    DO UPDATE`` upsert with FORCE RLS silently overwrote another user's row
    because PostgreSQL resolves the unique-constraint conflict BEFORE the
    RLS ``USING`` policy filters the row, so only the ``WITH CHECK`` policy
    (which passes for the new user's ``user_id``) was enforced. The fix scopes
    the UPDATE to the current user's row via ``WHERE mcp_servers.user_id =
    EXCLUDED.user_id``; when the conflict is cross-user, 0 rows are affected
    and the store raises :class:`McpServerAlreadyExistsError` (409).
    """

    async def test_save_raises_already_exists_on_cross_user_conflict(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — asyncpg returns "INSERT 0 0" when the ON CONFLICT UPDATE's
        # WHERE filters out the conflicting (other user's) row.
        mock_conn.execute = AsyncMock(return_value="INSERT 0 0")
        token = current_user_id.set("user-B")
        try:
            # Act + Assert
            with pytest.raises(McpServerAlreadyExistsError):
                await store.save(_entry(name="taken-by-user-A"))
        finally:
            current_user_id.reset(token)

    async def test_save_update_zero_also_raises(self, store, mock_conn):
        # Some asyncpg versions return "UPDATE 0" instead of "INSERT 0 0".
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")
        token = current_user_id.set("user-B")
        try:
            with pytest.raises(McpServerAlreadyExistsError):
                await store.save(_entry(name="taken"))
        finally:
            current_user_id.reset(token)

    async def test_save_insert_one_does_not_raise(self, store, mock_conn):
        # A fresh insert (no conflict) returns "INSERT 0 1" — must not raise.
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        token = current_user_id.set("user-A")
        try:
            await store.save(_entry(name="brand-new"))  # should not raise
        finally:
            current_user_id.reset(token)

    async def test_save_update_one_does_not_raise(self, store, mock_conn):
        # Updating the user's own row returns "UPDATE 1" — must not raise.
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        token = current_user_id.set("user-A")
        try:
            await store.save(_entry(name="own-server"))  # should not raise
        finally:
            current_user_id.reset(token)

    async def test_save_legacy_mode_no_raise_on_ok(self, store, mock_conn):
        # Legacy mode (no user) + mocked "OK" return — must not raise (the
        # "0" check only runs when user_id is set).
        mock_conn.execute = AsyncMock(return_value="OK")
        # current_user_id is None (no token set) → legacy path
        await store.save(_entry(name="legacy"))  # should not raise

    async def test_save_per_user_sql_has_where_clause(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        """The per-user upsert SQL must include the ``WHERE
        mcp_servers.user_id = EXCLUDED.user_id`` scoping clause.
        """
        token = current_user_id.set("user-A")
        try:
            await store.save(_entry(name="srv-A"))
            sql_arg = mock_conn.execute.await_args.args[0]
            assert "mcp_servers.user_id = EXCLUDED.user_id" in sql_arg, (
                f"per-user WHERE clause missing; SQL={sql_arg}"
            )
        finally:
            current_user_id.reset(token)

    async def test_save_legacy_sql_has_no_where_clause(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        """Legacy mode (no user) keeps the original global upsert (no WHERE)."""
        # current_user_id is None
        await store.save(_entry(name="legacy-srv"))
        sql_arg = mock_conn.execute.await_args.args[0]
        assert "mcp_servers.user_id = EXCLUDED.user_id" not in sql_arg, (
            f"unexpected per-user WHERE in legacy SQL; SQL={sql_arg}"
        )


# -- list_all user isolation ---------------------------------------------------


class TestListAllUserIsolation:
    """list_all() must filter rows by current_user_id when set."""

    async def test_list_returns_only_rows_for_current_user(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
        mock_cipher: AsyncMock,
    ):
        # Arrange — DB returns a row owned by user A, the store asks for user A
        mock_conn.fetch.return_value = [
            _row(name="srv-A", user_id="user-A"),
        ]
        token = current_user_id.set("user-A")
        try:
            # Act
            result = await store.list_all()

            # Assert — the SELECT filters by user_id
            fetch_call = mock_conn.fetch.await_args
            sql = fetch_call.args[0]
            assert "user_id" in sql.lower(), (
                f"user_id filter missing from list_all; sql={sql}"
            )
            assert len(result) == 1
            assert result[0].name == "srv-A"
        finally:
            current_user_id.reset(token)


# -- get user isolation --------------------------------------------------------


class TestGetUserIsolation:
    """get() must filter by user_id when set; raise NotFound for other users."""

    async def test_get_returns_entry_for_current_user(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
    ):
        # Arrange
        mock_conn.fetchrow.return_value = _row(name="srv-A", user_id="user-A")
        token = current_user_id.set("user-A")
        try:
            # Act
            result = await store.get("srv-A")

            # Assert — SQL filters by user_id
            fetchrow_call = mock_conn.fetchrow.await_args
            sql = fetchrow_call.args[0]
            assert "user_id" in sql.lower(), f"user_id missing; sql={sql}"
            assert result.name == "srv-A"
        finally:
            current_user_id.reset(token)

    async def test_get_raises_not_found_for_other_user_when_filter_excludes(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
    ):
        # Arrange — DB returns no row (RLS filter excluded it)
        mock_conn.fetchrow.return_value = None
        token = current_user_id.set("user-B")
        try:
            # Act & Assert
            with pytest.raises(McpServerNotFoundError):
                await store.get("srv-A")
        finally:
            current_user_id.reset(token)

    async def test_get_no_user_filter_when_contextvar_none(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
    ):
        # Arrange — contextvar default None
        mock_conn.fetchrow.return_value = _row(name="srv-A", user_id="")
        # Act
        result = await store.get("srv-A")

        # Assert — the WHERE clause does NOT filter by user_id (legacy path).
        # ``user_id`` may appear in the SELECT column list (it's a real column)
        # but must NOT appear in the WHERE clause.
        fetchrow_call = mock_conn.fetchrow.await_args
        sql = fetchrow_call.args[0].lower()
        where_clause = sql.split("where", 1)[1] if "where" in sql else ""
        assert "user_id" not in where_clause, (
            f"unexpected user_id filter in WHERE; sql={sql}"
        )
        assert result.name == "srv-A"


# -- delete user isolation -----------------------------------------------------


class TestDeleteUserIsolation:
    """delete() must filter by user_id when set."""

    async def test_delete_filters_by_user_id_when_user_set(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
    ):
        # Arrange — fetchval returns None (RLS excluded the row)
        mock_conn.fetchval.return_value = None
        token = current_user_id.set("user-B")
        try:
            # Act & Assert
            with pytest.raises(McpServerNotFoundError):
                await store.delete("srv-A")

            # Assert — the existence check filters by user_id
            fetchval_call = mock_conn.fetchval.await_args
            sql = fetchval_call.args[0]
            assert "user_id" in sql.lower(), f"user_id missing; sql={sql}"
        finally:
            current_user_id.reset(token)
