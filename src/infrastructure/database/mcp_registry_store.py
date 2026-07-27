"""Raw asyncpg store for the MCP server registry.

Persists registered MCP servers in the ``mcp_servers`` table. Secrets
(``headers``, ``env``, ``auth_token``) are encrypted with the injected
:class:`SecretCipher` before being written and decrypted on read, so the domain
entity only ever holds plaintext.

**Schema ownership:** the ``mcp_servers`` table is created and evolved by this
service's Alembic migrations (``001_create_mcp_servers_table`` and
``002_add_user_id_to_mcp_servers``), run automatically at startup via
``_run_alembic_upgrade`` in ``main.py``. The migration state is tracked in the
``raganything_alembic_version`` table, separate from composable-agents'
``alembic_version`` table so both services can share the same database without
colliding.

**Per-user RLS isolation:** when the ``current_user_id`` RLS contextvar is set
(by ``ComposableAgentsSecurity.verify_credentials`` or the MCP middleware after
a successful dual-auth), the store:

1. Emits ``SELECT set_config('app.user_id', $1, true)`` on the acquired
   asyncpg connection so PostgreSQL RLS policies on ``mcp_servers``
   (``003_enable_rls_mcp_servers``) filter rows per authenticated user.
2. Adds a ``WHERE user_id = $X`` clause to its own SQL queries so isolation
   works even on databases where RLS is not yet enabled (defence in depth).
3. Persists the ``current_user_id`` on inserts/upserts.

When the contextvar is None (legacy/tests, no auth wired), the store behaves
as before — no GUC, no user_id filter — so existing tests and the dev path
keep working unchanged.
"""

import json
import logging
from typing import Any

from domain.entities.mcp_server_config import McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerAlreadyExistsError, McpServerNotFoundError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.secret_cipher import SecretCipher
from infrastructure.database.rls_context import current_user_id, set_rls_context

logger = logging.getLogger(__name__)


class McpRegistryStore(McpServerRegistryStore):
    """Asyncpg-backed MCP server registry store.

    Attributes:
        _pool: The asyncpg connection pool (``pool.acquire`` async cm).
        _cipher: The secret cipher for encrypting/decrypting secrets.
    """

    def __init__(self, pool: Any, cipher: SecretCipher) -> None:
        self._pool = pool
        self._cipher = cipher

    async def save(self, entry: RegisteredMcpServer) -> None:
        """Insert or update (upsert) a registered MCP server by name.

        Secrets (headers, env, auth_token) are encrypted before being written.
        The ``current_user_id`` RLS contextvar is persisted as ``user_id`` on
        the row (empty string when no user is set — legacy path).

        **Per-user conflict guard:** when ``current_user_id`` is set, the
        ``ON CONFLICT (name) DO UPDATE`` is scoped to the current user's row
        via a ``WHERE mcp_servers.user_id = EXCLUDED.user_id`` clause. This
        prevents a cross-user overwrite: if the conflicting row belongs to
        another user, the UPDATE affects 0 rows and the INSERT did not happen
        either, so the method detects the cross-user conflict (a row with the
        same name exists globally but not for this user) and raises
        :class:`McpServerAlreadyExistsError` (409). Without this guard, RLS
        hides the other user's row from the ``exists`` pre-check but the
        global PK ``name`` still triggers ``ON CONFLICT``, silently
        transferring ownership of the row to the new user (data loss for the
        original owner). See the PostgreSQL RLS pitfall: ``ON CONFLICT DO
        UPDATE`` resolves the conflict via the unique index BEFORE the
        ``USING`` policy filters the row, so the UPDATE's ``WITH CHECK`` is
        the only policy enforced — and it passes for the new user's
        ``user_id``.

        Args:
            entry: The registered server entry to persist.

        Raises:
            McpServerAlreadyExistsError: When ``current_user_id`` is set and a
                server with the same name exists for ANOTHER user (the upsert
                is refused to prevent cross-user data loss).
        """
        headers_encrypted = await self._cipher.encrypt(json.dumps(entry.headers))
        env_encrypted = await self._cipher.encrypt(json.dumps(entry.env))
        auth_token_encrypted = (
            await self._cipher.encrypt(entry.auth_token)
            if entry.auth_token is not None
            else None
        )
        user_id = current_user_id.get() or ""

        if user_id:
            # Per-user mode: scope the UPDATE to the current user's row so a
            # cross-user conflict (same name owned by another user) is NOT
            # silently overwritten. The ``WHERE`` clause on the UPDATE filters
            # out the other user's row, so 0 rows are updated when the
            # conflict is cross-user — detect that and raise.
            sql = """
            INSERT INTO mcp_servers
                (name, transport, url, headers_encrypted, env_encrypted,
                 auth_token_encrypted, tool_count, created_at, updated_at,
                 source_type, openapi_url, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (name) DO UPDATE SET
                transport            = EXCLUDED.transport,
                url                  = EXCLUDED.url,
                headers_encrypted    = EXCLUDED.headers_encrypted,
                env_encrypted        = EXCLUDED.env_encrypted,
                auth_token_encrypted = EXCLUDED.auth_token_encrypted,
                tool_count           = EXCLUDED.tool_count,
                updated_at           = now(),
                source_type          = EXCLUDED.source_type,
                openapi_url          = EXCLUDED.openapi_url,
                user_id              = EXCLUDED.user_id
            WHERE mcp_servers.user_id = EXCLUDED.user_id;
            """
        else:
            # Legacy mode (no user): original global upsert (backward-compat).
            sql = """
            INSERT INTO mcp_servers
                (name, transport, url, headers_encrypted, env_encrypted,
                 auth_token_encrypted, tool_count, created_at, updated_at,
                 source_type, openapi_url, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (name) DO UPDATE SET
                transport            = EXCLUDED.transport,
                url                  = EXCLUDED.url,
                headers_encrypted    = EXCLUDED.headers_encrypted,
                env_encrypted        = EXCLUDED.env_encrypted,
                auth_token_encrypted = EXCLUDED.auth_token_encrypted,
                tool_count           = EXCLUDED.tool_count,
                updated_at           = now(),
                source_type          = EXCLUDED.source_type,
                openapi_url          = EXCLUDED.openapi_url,
                user_id              = EXCLUDED.user_id;
            """
        async with self._pool.acquire() as conn:
            await set_rls_context(conn, current_user_id.get())
            result = await conn.execute(
                sql,
                entry.name,
                entry.transport.value,
                entry.url,
                headers_encrypted,
                env_encrypted,
                auth_token_encrypted,
                entry.tool_count,
                entry.created_at,
                entry.updated_at,
                entry.source_type,
                entry.openapi_url,
                user_id,
            )
            logger.info(LogMessage.MCP_SERVER_SAVE_REPOSITORY, entry.name)

        # Per-user cross-user conflict detection: when the upsert resulted in
        # 0 rows affected in per-user mode, the name is taken by ANOTHER user.
        # The ``WHERE mcp_servers.user_id = EXCLUDED.user_id`` filtered the
        # conflicting row out, so 0 rows were updated and no INSERT happened
        # (asyncpg returns ``"INSERT 0 0"`` for this path, or ``"UPDATE 0"``
        # for a same-user no-op). Raise a 409 so the caller reports the
        # conflict instead of silently no-op'ing (which the create use case
        # would mistake for success).
        # ``"INSERT 0 1"`` = a fresh insert (no conflict);
        # ``"UPDATE 1"`` = the user's own row was updated;
        # ``"INSERT 0 0"`` / ``"UPDATE 0"`` = cross-user conflict (or no-op).
        if user_id and result.split()[-1] == "0":
            raise McpServerAlreadyExistsError(
                ErrorMessage.MCP_SERVER_ALREADY_EXISTS.format(name=entry.name)
            )

    async def get(self, name: str) -> RegisteredMcpServer:
        """Retrieve a registered MCP server by name (secrets decrypted).

        When ``current_user_id`` is set, the query filters by ``user_id`` so a
        user cannot read another user's server.

        Args:
            name: The unique server name.

        Returns:
            The registered server entry with secrets decrypted.

        Raises:
            McpServerNotFoundError: If no server exists for this name (or it
                belongs to another user when RLS is engaged).
        """
        user_id = current_user_id.get()
        where = (
            "WHERE name = $1" if user_id is None else "WHERE name = $1 AND user_id = $2"
        )
        params: tuple[Any, ...] = (name, user_id) if user_id is not None else (name,)
        sql = f"""
        SELECT name, transport, url, headers_encrypted, env_encrypted,
               auth_token_encrypted, tool_count, created_at, updated_at,
               source_type, openapi_url, user_id
        FROM mcp_servers
        {where};
        """

        async with self._pool.acquire() as conn:
            await set_rls_context(conn, user_id)
            row = await conn.fetchrow(sql, *params)

        if row is None:
            raise McpServerNotFoundError(
                ErrorMessage.MCP_SERVER_NOT_FOUND.format(name=name)
            )
        return await self._row_to_entry(row)

    async def list_all(self) -> list[RegisteredMcpServer]:
        """List all registered MCP servers (secrets decrypted).

        When ``current_user_id`` is set, only servers owned by the current
        user are returned.

        Entries whose secrets cannot be decrypted (e.g. encrypted with a
        previous/rotated key) are logged and skipped so a single corrupted row
        never blocks the list or application startup.

        Returns:
            A list of all registered server entries with decryptable secrets.
        """
        user_id = current_user_id.get()
        if user_id is not None:
            where = "WHERE user_id = $1"
            params: tuple[Any, ...] = (user_id,)
        else:
            where = ""
            params = ()
        sql = f"""
        SELECT name, transport, url, headers_encrypted, env_encrypted,
               auth_token_encrypted, tool_count, created_at, updated_at,
               source_type, openapi_url, user_id
        FROM mcp_servers
        {where}
        ORDER BY name;
        """

        async with self._pool.acquire() as conn:
            await set_rls_context(conn, user_id)
            rows = await conn.fetch(sql, *params)

        entries: list[RegisteredMcpServer] = []
        for row in rows:
            try:
                entries.append(await self._row_to_entry(row))
            except Exception as e:  # noqa: BLE001 — decryption/json errors are per-row
                logger.warning(
                    "Skipping MCP server %s: secret decryption failed: %s",
                    row["name"],
                    e,
                )
        return entries

    async def delete(self, name: str) -> None:
        """Delete a registered MCP server by name.

        When ``current_user_id`` is set, the existence check and delete filter
        by ``user_id`` so a user cannot delete another user's server.

        Args:
            name: The unique server name to delete.

        Raises:
            McpServerNotFoundError: If no server exists for this name (or it
                belongs to another user when RLS is engaged).
        """
        user_id = current_user_id.get()
        async with self._pool.acquire() as conn:
            await set_rls_context(conn, user_id)
            if user_id is not None:
                existing = await conn.fetchval(
                    "SELECT name FROM mcp_servers WHERE name = $1 AND user_id = $2;",
                    name,
                    user_id,
                )
            else:
                existing = await conn.fetchval(
                    "SELECT name FROM mcp_servers WHERE name = $1;", name
                )
            if existing is None:
                raise McpServerNotFoundError(
                    ErrorMessage.MCP_SERVER_NOT_FOUND.format(name=name)
                )
            if user_id is not None:
                await conn.execute(
                    "DELETE FROM mcp_servers WHERE name = $1 AND user_id = $2;",
                    name,
                    user_id,
                )
            else:
                await conn.execute("DELETE FROM mcp_servers WHERE name = $1;", name)
            logger.info(LogMessage.MCP_SERVER_DELETE_REPOSITORY, name)

    async def exists(self, name: str) -> bool:
        """Check whether a registered MCP server exists by name.

        When ``current_user_id`` is set, only servers owned by the current
        user are considered.

        Args:
            name: The unique server name to check.

        Returns:
            True if a server exists (for the current user when RLS is engaged),
            False otherwise.
        """
        user_id = current_user_id.get()
        async with self._pool.acquire() as conn:
            await set_rls_context(conn, user_id)
            if user_id is not None:
                result = await conn.fetchval(
                    "SELECT 1 FROM mcp_servers WHERE name = $1 AND user_id = $2;",
                    name,
                    user_id,
                )
            else:
                result = await conn.fetchval(
                    "SELECT 1 FROM mcp_servers WHERE name = $1;", name
                )
        return result is not None

    async def _row_to_entry(self, row: Any) -> RegisteredMcpServer:
        """Convert an asyncpg record to a domain entity, decrypting secrets.

        Args:
            row: The asyncpg record (mapping by column name).

        Returns:
            The registered server entity with plaintext secrets.
        """
        headers_json = await self._cipher.decrypt(row["headers_encrypted"])
        env_json = await self._cipher.decrypt(row["env_encrypted"])
        auth_token_encrypted = row["auth_token_encrypted"]
        auth_token = (
            await self._cipher.decrypt(auth_token_encrypted)
            if auth_token_encrypted is not None
            else None
        )
        return RegisteredMcpServer(
            name=row["name"],
            transport=McpTransportType(row["transport"]),
            url=row["url"],
            headers=json.loads(headers_json),
            env=json.loads(env_json),
            auth_token=auth_token,
            tool_count=row["tool_count"],
            source_type=row["source_type"],
            openapi_url=row["openapi_url"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
