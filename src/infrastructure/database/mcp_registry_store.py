"""Raw asyncpg store for the MCP server registry.

Persists registered MCP servers in the ``mcp_servers`` table. Secrets
(``headers``, ``env``, ``auth_token``) are encrypted with the injected
:class:`SecretCipher` before being written and decrypted on read, so the domain
entity only ever holds plaintext.

**Schema ownership:** the ``mcp_servers`` table is created and evolved by this
service's Alembic migrations (``src/alembic/versions/001_create_mcp_servers_
table.py``), run automatically at startup via ``_run_alembic_upgrade`` in
``main.py``. The migration state is tracked in the ``raganything_alembic_version``
table, separate from composable-agents' ``alembic_version`` table so both
services can share the same database without colliding.
"""

import json
import logging
from typing import Any

from domain.entities.mcp_server_config import McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerNotFoundError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.secret_cipher import SecretCipher

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

        Args:
            entry: The registered server entry to persist.
        """
        headers_encrypted = await self._cipher.encrypt(json.dumps(entry.headers))
        env_encrypted = await self._cipher.encrypt(json.dumps(entry.env))
        auth_token_encrypted = (
            await self._cipher.encrypt(entry.auth_token)
            if entry.auth_token is not None
            else None
        )

        sql = """
        INSERT INTO mcp_servers
            (name, transport, url, headers_encrypted, env_encrypted,
             auth_token_encrypted, tool_count, created_at, updated_at,
             source_type, openapi_url)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (name) DO UPDATE SET
            transport            = EXCLUDED.transport,
            url                  = EXCLUDED.url,
            headers_encrypted    = EXCLUDED.headers_encrypted,
            env_encrypted        = EXCLUDED.env_encrypted,
            auth_token_encrypted = EXCLUDED.auth_token_encrypted,
            tool_count           = EXCLUDED.tool_count,
            updated_at           = now(),
            source_type          = EXCLUDED.source_type,
            openapi_url          = EXCLUDED.openapi_url;
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
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
            )
            logger.info(LogMessage.MCP_SERVER_SAVE_REPOSITORY, entry.name)

    async def get(self, name: str) -> RegisteredMcpServer:
        """Retrieve a registered MCP server by name (secrets decrypted).

        Args:
            name: The unique server name.

        Returns:
            The registered server entry with secrets decrypted.

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        sql = """
        SELECT name, transport, url, headers_encrypted, env_encrypted,
               auth_token_encrypted, tool_count, created_at, updated_at,
               source_type, openapi_url
        FROM mcp_servers
        WHERE name = $1;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, name)

        if row is None:
            raise McpServerNotFoundError(ErrorMessage.MCP_SERVER_NOT_FOUND.format(name=name))
        return await self._row_to_entry(row)

    async def list_all(self) -> list[RegisteredMcpServer]:
        """List all registered MCP servers (secrets decrypted).

        Entries whose secrets cannot be decrypted (e.g. encrypted with a
        previous/rotated key) are logged and skipped so a single corrupted row
        never blocks the list or application startup.

        Returns:
            A list of all registered server entries with decryptable secrets.
        """
        sql = """
        SELECT name, transport, url, headers_encrypted, env_encrypted,
               auth_token_encrypted, tool_count, created_at, updated_at,
               source_type, openapi_url
        FROM mcp_servers
        ORDER BY name;
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)

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

        Args:
            name: The unique server name to delete.

        Raises:
            McpServerNotFoundError: If no server exists for this name.
        """
        async with self._pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT name FROM mcp_servers WHERE name = $1;", name
            )
            if existing is None:
                raise McpServerNotFoundError(
                    ErrorMessage.MCP_SERVER_NOT_FOUND.format(name=name)
                )
            await conn.execute("DELETE FROM mcp_servers WHERE name = $1;", name)
            logger.info(LogMessage.MCP_SERVER_DELETE_REPOSITORY, name)

    async def exists(self, name: str) -> bool:
        """Check whether a registered MCP server exists by name.

        Args:
            name: The unique server name to check.

        Returns:
            True if a server exists, False otherwise.
        """
        async with self._pool.acquire() as conn:
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
