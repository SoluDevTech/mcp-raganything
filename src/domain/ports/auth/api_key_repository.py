"""Port for the API-key repository (outbound boundary, READ-ONLY).

mcp-raganything shares the same Postgres ``raganything`` database as
composable-agents, which owns the ``api_keys`` table (created by its Alembic
migrations 011/014). mcp-raganything only READS this table to validate
per-user API keys — write operations (create/list/revoke) live in
composable-agents and are NOT exposed here.

Implemented by :class:`~infrastructure.database.api_key_reader.AsyncpgApiKeyReader`
against the ``api_keys`` table via the existing asyncpg pool.
"""

from abc import ABC, abstractmethod


class ApiKeyRepositoryPort(ABC):
    """Outbound port: look up active per-user API keys by hash (READ-ONLY).

    API keys are stored hashed (SHA-256 hex of the plaintext) so lookups are
    performed on the hash, never on the plaintext. An active key is one that
    is not revoked (``revoked_at IS NULL``).

    The ``api_keys`` table is FORCE RLS-protected by composable-agents, but at
    auth time the GUC ``app.user_id`` is NOT set yet (we are reading the table
    TO determine the user). The reader implementation therefore emits
    ``SET LOCAL row_security = off`` on its connection before the SELECT —
    a privileged auth operation.
    """

    @abstractmethod
    async def find_active_by_hash(self, key_hash: str) -> tuple[str, str] | None:
        """Return ``(user_id, key_id)`` for the active key matching ``key_hash``.

        Args:
            key_hash: The SHA-256 hex digest of the API key plaintext.

        Returns:
            A ``(user_id, key_id)`` tuple if an active key matches, else ``None``.
        """
        ...
