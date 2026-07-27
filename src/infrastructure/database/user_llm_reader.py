"""Read-only asyncpg adapter for the shared ``user_llm_settings`` table.

mcp-raganything shares the SAME Postgres database as composable-agents, which
owns the ``user_llm_settings`` table (created by composable-agents' Alembic).
This adapter is the READ-ONLY mcp-raganything side: it exposes ``get`` and
``get_decrypted`` only. Writes (upsert/delete) are composable-agents' concern.

**RLS bypass:** the ``user_llm_settings`` table is FORCE RLS-protected by
composable-agents. The reader resolves credentials AFTER auth (the
``current_user_id`` is already known), but the RLS GUC ``app.user_id`` may not
be set on the mcp-raganything asyncpg pool connection used here (the pool is
shared with the MCP registry store which sets the GUC per-connection, but this
reader is called from the per-user LLM factory, not from a request-scoped
store call). The reader therefore emits ``SET LOCAL row_security = off`` before
the SELECT — a privileged read. This is safe: the reader never mutates the
table and only resolves credentials by primary key (``user_id``), exactly
like the :class:`AsyncpgApiKeyReader` resolves the user identity at auth time.

**Masking:** ``get`` decrypts the API key (via the injected
:class:`FernetSecretCipher`) and returns a masked preview
(``first3...last3``; ``***`` for keys <= 6 chars). ``get_decrypted`` returns
the plaintext ``(base_url, api_key)`` tuple for the per-user LLM factory.
"""

import logging
from typing import Any

from domain.entities.user_llm_settings import UserLlmSettings
from domain.ports.secret_cipher import SecretCipher
from domain.ports.user_llm_settings_repository import UserLlmSettingsRepositoryPort

logger = logging.getLogger(__name__)


def _mask(api_key_plaintext: str) -> str:
    """Return a masked preview of an API key (first 3 + ``...`` + last 3).

    For very short keys (<=6 chars) the whole key is masked as ``***``.

    Args:
        api_key_plaintext: The decrypted API key.

    Returns:
        A masked string safe to expose in GET responses.
    """
    if len(api_key_plaintext) <= 6:
        return "***"
    return f"{api_key_plaintext[:3]}...{api_key_plaintext[-3:]}"


class AsyncpgUserLlmReader(UserLlmSettingsRepositoryPort):
    """Read-only asyncpg adapter for the shared ``user_llm_settings`` table.

    Args:
        pool: The asyncpg connection pool (shared with ``McpRegistryStore``).
        cipher: The :class:`FernetSecretCipher` used to decrypt the API key.
    """

    def __init__(self, pool: Any, cipher: SecretCipher) -> None:
        self._pool = pool
        self._cipher = cipher

    async def get(self, user_id: str) -> UserLlmSettings | None:
        """Return the user's settings with a masked API key, or ``None``.

        Emits ``SET LOCAL row_security = off`` BEFORE the SELECT because the
        ``user_llm_settings`` table is FORCE RLS-protected by composable-agents.

        Args:
            user_id: Owner identifier.

        Returns:
            A :class:`UserLlmSettings` with ``api_key_masked`` (decrypted then
            masked — never the full key), or ``None`` if absent.
        """
        sql = """
        SELECT user_id, provider, base_url, api_key_encrypted,
               created_at, updated_at
        FROM user_llm_settings
        WHERE user_id = $1;
        """
        async with self._pool.acquire() as conn:
            await conn.execute("SET LOCAL row_security = off")
            row = await conn.fetchrow(sql, user_id)

        if row is None:
            return None
        masked: str | None
        try:
            plaintext = await self._cipher.decrypt(row["api_key_encrypted"])
            masked = _mask(plaintext)
        except Exception as e:  # noqa: BLE001 — decryption errors are non-fatal
            logger.warning("Failed to decrypt API key for user_id=%s: %s", user_id, e)
            masked = None
        return UserLlmSettings(
            user_id=row["user_id"],
            provider=row["provider"],
            base_url=row["base_url"],
            api_key_masked=masked,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_decrypted(self, user_id: str) -> tuple[str, str] | None:
        """Return ``(base_url, api_key_plaintext)`` for the per-user LLM factory.

        Emits ``SET LOCAL row_security = off`` BEFORE the SELECT (see class
        docstring).

        Args:
            user_id: Owner identifier.

        Returns:
            A ``(base_url, api_key_plaintext)`` tuple, or ``None`` if absent.
        """
        sql = """
        SELECT base_url, api_key_encrypted
        FROM user_llm_settings
        WHERE user_id = $1;
        """
        async with self._pool.acquire() as conn:
            await conn.execute("SET LOCAL row_security = off")
            row = await conn.fetchrow(sql, user_id)

        if row is None:
            return None
        plaintext = await self._cipher.decrypt(row["api_key_encrypted"])
        return row["base_url"], plaintext
