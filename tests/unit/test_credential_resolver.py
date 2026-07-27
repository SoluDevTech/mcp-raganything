"""Tests for the ``UserLlmCredentialResolver`` domain service (TDD Red phase).

The resolver is a thin domain service that wraps the
``UserLlmSettingsRepositoryPort.get_decrypted`` method and returns
``(base_url, api_key)`` or ``None``. It is the single point the per-user LLM
factories depend on, keeping the factory decoupled from the repository port.

The repository port is mocked so no database is needed.
"""

from unittest.mock import AsyncMock

import pytest

from domain.ports.user_llm_settings_repository import UserLlmSettingsRepositoryPort
from domain.services.llm.credential_resolver import UserLlmCredentialResolver


@pytest.fixture
def mock_repo() -> AsyncMock:
    return AsyncMock(spec=UserLlmSettingsRepositoryPort)


@pytest.fixture
def resolver(mock_repo: AsyncMock) -> UserLlmCredentialResolver:
    return UserLlmCredentialResolver(repo=mock_repo)


class TestResolve:
    """Tests for ``UserLlmCredentialResolver.resolve``."""

    async def test_returns_tuple_when_user_has_settings(
        self, resolver: UserLlmCredentialResolver, mock_repo: AsyncMock
    ) -> None:
        # Arrange
        mock_repo.get_decrypted.return_value = (
            "https://openrouter.ai/api/v1",
            "sk-or-v1-secret",
        )

        # Act
        result = await resolver.resolve("user-A")

        # Assert
        assert result is not None
        assert result == ("https://openrouter.ai/api/v1", "sk-or-v1-secret")
        mock_repo.get_decrypted.assert_awaited_once_with("user-A")

    async def test_returns_none_when_user_has_no_settings(
        self, resolver: UserLlmCredentialResolver, mock_repo: AsyncMock
    ) -> None:
        # Arrange
        mock_repo.get_decrypted.return_value = None

        # Act
        result = await resolver.resolve("user-missing")

        # Assert
        assert result is None

    async def test_passes_user_id_to_repository(
        self, resolver: UserLlmCredentialResolver, mock_repo: AsyncMock
    ) -> None:
        # Arrange
        mock_repo.get_decrypted.return_value = None

        # Act
        await resolver.resolve("user-XYZ")

        # Assert
        mock_repo.get_decrypted.assert_awaited_once_with("user-XYZ")
