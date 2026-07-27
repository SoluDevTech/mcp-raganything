"""Outbound port: resolve per-user LLM provider settings (READ-ONLY).

mcp-raganything shares the same Postgres database as composable-agents, which
owns the ``user_llm_settings`` table (created by composable-agents' Alembic).
This port is the READ-ONLY mcp-raganything side: it exposes only ``get`` and
``get_decrypted``. Writes (upsert/delete) are composable-agents' concern and
are NOT part of this port — the mcp-raganything adapter must never mutate the
table.

The per-user LLM/embeddings factories depend on ``get_decrypted`` to resolve
the ``(base_url, api_key)`` tuple needed to build a per-user
``OpenAIEmbeddings`` / ``ChatOpenAI`` instance.
"""

from abc import ABC, abstractmethod

from domain.entities.user_llm_settings import UserLlmSettings


class UserLlmSettingsRepositoryPort(ABC):
    """Outbound port: resolve per-user LLM provider settings (read-only)."""

    @abstractmethod
    async def get(self, user_id: str) -> UserLlmSettings | None:
        """Return the user's settings (masked key), or ``None`` if not configured.

        Args:
            user_id: Owner identifier.

        Returns:
            A :class:`UserLlmSettings` with ``api_key_masked`` (never the full
            key), or ``None`` when the user has no configured provider.
        """
        ...

    @abstractmethod
    async def get_decrypted(self, user_id: str) -> tuple[str, str] | None:
        """Return ``(base_url, api_key_plaintext)`` for the per-user LLM factory.

        The plaintext is decrypted on demand (per-request). Returns ``None``
        when the user has no configured provider.

        Args:
            user_id: Owner identifier.

        Returns:
            A ``(base_url, api_key_plaintext)`` tuple, or ``None``.
        """
        ...
