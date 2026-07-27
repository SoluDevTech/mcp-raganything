"""Domain service: resolve per-user LLM credentials.

A thin service that wraps :meth:`UserLlmSettingsRepositoryPort.get_decrypted`
and returns ``(base_url, api_key)`` or ``None``. It is the single point the
per-user LLM/embeddings factories depend on, keeping the factories decoupled
from the repository port (DIP): the factories depend on this service (a
domain abstraction), not on the port directly.

Keeping the resolution in a service (rather than calling the port from the
factory) lets us add caching, fallback, or audit logic here without touching
the infrastructure factories.
"""

import logging

from domain.ports.user_llm_settings_repository import UserLlmSettingsRepositoryPort

logger = logging.getLogger(__name__)


class UserLlmCredentialResolver:
    """Resolve ``(base_url, api_key)`` for a user from the settings repository.

    Attributes:
        _repo: The read-only user LLM settings repository port.
    """

    def __init__(self, repo: UserLlmSettingsRepositoryPort) -> None:
        self._repo = repo

    async def resolve(self, user_id: str) -> tuple[str, str] | None:
        """Return ``(base_url, api_key_plaintext)`` for the user, or ``None``.

        Args:
            user_id: The authenticated user identifier.

        Returns:
            A ``(base_url, api_key)`` tuple if the user has configured a
            provider, else ``None`` (the caller raises
            :class:`LlmNotConfiguredError`).
        """
        result = await self._repo.get_decrypted(user_id)
        if result is None:
            logger.debug("No LLM settings found for user_id=%s", user_id)
        return result
