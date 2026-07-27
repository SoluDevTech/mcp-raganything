"""Tests for the per-user LLM/embeddings factories (TDD Red phase).

The factories replace the module-level singletons ``_embedding`` /
``classical_llm`` in ``dependencies.py``. They read ``current_user_id`` from
the RLS contextvar and resolve per-user credentials via
``UserLlmCredentialResolver``:

- when ``current_user_id`` is set AND the user has settings → build
  ``OpenAIEmbeddings`` / ``LangchainOpenAIAdapter`` with the user's
  ``api_key`` and ``base_url``;
- when ``current_user_id`` is set BUT the user has no settings → raise
  ``LlmNotConfiguredError`` (HTTP 422);
- when ``current_user_id`` is None (legacy/tests, no auth wired) → fall back
  to the ``LLMConfig`` env-based credentials (the original behaviour).

The credential resolver and ``FernetSecretCipher`` are mocked so no database
or real keys are needed.
"""

from unittest.mock import AsyncMock, patch

import pytest

from domain.errors.llm import LlmNotConfiguredError
from infrastructure.database.rls_context import current_user_id


@pytest.fixture
def mock_resolver() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def llm_config_env() -> dict:
    """Env-based LLMConfig values (legacy fallback path)."""
    return {
        "api_key": "env-key-legacy",
        "api_base_url": "https://env.example/api/v1",
        "CHAT_MODEL": "openai/gpt-4o-mini",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIM": 1536,
    }


class TestGetEmbeddingForUser:
    """Tests for ``get_embedding_for_user``."""

    async def test_uses_user_credentials_when_user_set_and_configured(
        self, mock_resolver: AsyncMock
    ) -> None:
        # Arrange
        mock_resolver.resolve.return_value = (
            "https://user.example/api/v1",
            "sk-user-secret-key",
        )
        token = current_user_id.set("user-A")
        try:
            with patch(
                "dependencies.get_user_llm_resolver", return_value=mock_resolver
            ):
                from dependencies import get_embedding_for_user

                # Act
                embedding = await get_embedding_for_user()

                # Assert — OpenAIEmbeddings built with user creds
                assert embedding is not None
                # OpenAIEmbeddings stores api_key / base_url; check via the
                # underlying client or openai_api_key attribute.
                assert (
                    embedding.openai_api_key.get_secret_value() == "sk-user-secret-key"
                )
                assert embedding.openai_api_base == "https://user.example/api/v1"
        finally:
            current_user_id.reset(token)

    async def test_raises_llm_not_configured_when_user_set_but_no_settings(
        self, mock_resolver: AsyncMock
    ) -> None:
        # Arrange
        mock_resolver.resolve.return_value = None
        token = current_user_id.set("user-B")
        try:
            with patch(
                "dependencies.get_user_llm_resolver", return_value=mock_resolver
            ):
                from dependencies import get_embedding_for_user

                # Act & Assert
                with pytest.raises(LlmNotConfiguredError):
                    await get_embedding_for_user()
        finally:
            current_user_id.reset(token)

    async def test_falls_back_to_env_when_contextvar_none(
        self, mock_resolver: AsyncMock, llm_config_env: dict
    ) -> None:
        # Arrange — contextvar default None (legacy/tests path)
        with patch("dependencies.get_user_llm_resolver", return_value=mock_resolver):
            from dependencies import _legacy_embedding, get_embedding_for_user

            # Act
            embedding = await get_embedding_for_user()

            # Assert — resolver NOT called (env fallback) and the legacy
            # singleton built at import time is returned directly.
            mock_resolver.resolve.assert_not_awaited()
            assert embedding is _legacy_embedding


class TestGetChatLlmForUser:
    """Tests for ``get_chat_llm_for_user``."""

    async def test_uses_user_credentials_when_user_set_and_configured(
        self, mock_resolver: AsyncMock
    ) -> None:
        # Arrange
        mock_resolver.resolve.return_value = (
            "https://user.example/api/v1",
            "sk-user-chat-key",
        )
        token = current_user_id.set("user-A")
        try:
            with (
                patch("dependencies.get_user_llm_resolver", return_value=mock_resolver),
                patch("dependencies.classical_rag_config") as mock_rag_cfg,
            ):
                mock_rag_cfg.CLASSICAL_LLM_TEMPERATURE = 0.0
                with patch("dependencies.llm_config") as mock_cfg:
                    mock_cfg.CHAT_MODEL = "openai/gpt-4o-mini"
                    from dependencies import get_chat_llm_for_user

                    # Act
                    llm = await get_chat_llm_for_user()

                    # Assert — LangchainOpenAIAdapter wrapping ChatOpenAI with user creds
                    assert llm is not None
                    inner = llm._llm
                    assert inner.openai_api_key.get_secret_value() == "sk-user-chat-key"
                    assert inner.openai_api_base == "https://user.example/api/v1"
        finally:
            current_user_id.reset(token)

    async def test_raises_llm_not_configured_when_user_set_but_no_settings(
        self, mock_resolver: AsyncMock
    ) -> None:
        # Arrange
        mock_resolver.resolve.return_value = None
        token = current_user_id.set("user-B")
        try:
            with patch(
                "dependencies.get_user_llm_resolver", return_value=mock_resolver
            ):
                from dependencies import get_chat_llm_for_user

                # Act & Assert
                with pytest.raises(LlmNotConfiguredError):
                    await get_chat_llm_for_user()
        finally:
            current_user_id.reset(token)

    async def test_falls_back_to_env_when_contextvar_none(
        self, mock_resolver: AsyncMock, llm_config_env: dict
    ) -> None:
        # Arrange — contextvar default None (legacy/tests path)
        with patch("dependencies.get_user_llm_resolver", return_value=mock_resolver):
            from dependencies import _legacy_llm, get_chat_llm_for_user

            # Act
            llm = await get_chat_llm_for_user()

            # Assert — resolver NOT called (env fallback) and the legacy
            # singleton built at import time is returned directly.
            mock_resolver.resolve.assert_not_awaited()
            assert llm is _legacy_llm
