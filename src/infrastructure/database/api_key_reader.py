"""Read-only asyncpg adapter for the shared ``api_keys`` table.

mcp-raganything shares the SAME Postgres database (``raganything``) as
composable-agents, which owns the ``api_keys`` table (created by its Alembic
migrations 011/014). This adapter is the READ-ONLY mcp-raganything side: it
exposes a single method, ``find_active_by_hash(key_hash)``, returning the
``(user_id, key_id)`` tuple for the active (``revoked_at IS NULL``) key
matching the SHA-256 hash, or ``None`` when no active key matches.

**RLS bypass at auth time:** the ``api_keys`` table is FORCE RLS-protected by
composable-agents, but at auth time the GUC ``app.user_id`` is NOT set yet
(we are reading the table TO determine the user). The reader therefore emits
``SET LOCAL row_security = off`` on its connection before the SELECT — a
privileged auth operation. This is intentional and safe: the reader never
mutates the table and only resolves the user identity, exactly like a login
form reading a credentials table.
"""

import logging
from typing import Any

from domain.ports.auth.api_key_repository import ApiKeyRepositoryPort

logger = logging.getLogger(__name__)


class AsyncpgApiKeyReader(ApiKeyRepositoryPort):
    """Read-only asyncpg adapter for the shared ``api_keys`` table.

    Args:
        pool: The asyncpg connection pool (shared with ``McpRegistryStore``).
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def find_active_by_hash(self, key_hash: str) -> tuple[str, str] | None:
        """Return ``(user_id, key_id)`` for the active key matching ``key_hash``.

        Emits ``SET LOCAL row_security = off`` BEFORE the SELECT because the
        ``api_keys`` table is FORCE RLS-protected by composable-agents and the
        GUC ``app.user_id`` is not set yet at auth time.

        Args:
            key_hash: The SHA-256 hex digest of the API key plaintext.

        Returns:
            A ``(user_id, key_id)`` tuple if an active key matches, else ``None``.
        """
        sql = """
        SELECT user_id, id
        FROM api_keys
        WHERE key_hash = $1 AND revoked_at IS NULL;
        """
        async with self._pool.acquire() as conn:
            # Privileged auth read: bypass RLS to resolve the user identity.
            # Done BEFORE the SELECT so the FORCE RLS policy does not hide rows.
            await conn.execute("SET LOCAL row_security = off")
            row = await conn.fetchrow(sql, key_hash)

        if row is None:
            return None
        return row["user_id"], row["id"]
