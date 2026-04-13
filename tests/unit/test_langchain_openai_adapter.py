"""Tests for LangchainOpenAIAdapter — TDD Red phase.

These tests will FAIL until the production code is implemented.
This adapter implements LLMPort using langchain-openai ChatOpenAI.
The ChatOpenAI client is an external dependency and is fully mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.ports.llm_port import LLMPort


class TestLangchainOpenAIAdapter:
    """Tests for LangchainOpenAIAdapter."""

    @pytest.fixture
    def adapter(self):
        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        return LangchainOpenAIAdapter(
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.0,
        )

    @patch("infrastructure.llm.langchain_openai_adapter.ChatOpenAI")
    async def test_generate_calls_chat_openAI(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Should call ChatOpenAI with system and user messages."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response text"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_cls.return_value = mock_llm

        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        adapter = LangchainOpenAIAdapter(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.0,
        )

        result = await adapter.generate(
            system_prompt="You are a helpful assistant.",
            user_message="What is machine learning?",
        )

        assert result == "Generated response text"
        mock_llm.ainvoke.assert_called_once()

    @patch("infrastructure.llm.langchain_openai_adapter.ChatOpenAI")
    async def test_generate_returns_string(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Should return a string from generate."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_cls.return_value = mock_llm

        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        adapter = LangchainOpenAIAdapter(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.0,
        )

        result = await adapter.generate(
            system_prompt="Be concise.",
            user_message="Say hello",
        )

        assert isinstance(result, str)
        assert result == "Hello world"

    @patch("infrastructure.llm.langchain_openai_adapter.ChatOpenAI")
    async def test_generate_chat_calls_with_messages(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Should call ChatOpenAI with the full message list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Chat response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_cls.return_value = mock_llm

        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        adapter = LangchainOpenAIAdapter(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.0,
        )

        messages = [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "Score this chunk: 8/10"},
        ]

        result = await adapter.generate_chat(messages=messages)

        assert result == "Chat response"
        mock_llm.ainvoke.assert_called_once()

    @patch("infrastructure.llm.langchain_openai_adapter.ChatOpenAI")
    async def test_generate_chat_returns_string(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Should return a string from generate_chat."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Chat result"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_cls.return_value = mock_llm

        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        adapter = LangchainOpenAIAdapter(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.0,
        )

        result = await adapter.generate_chat(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, str)

    @patch("infrastructure.llm.langchain_openai_adapter.ChatOpenAI")
    async def test_adapter_configures_chat_openai_with_params(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Should configure ChatOpenAI with api_key, base_url, model, temperature."""
        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        LangchainOpenAIAdapter(
            api_key="my-key",
            base_url="https://custom.api.com/v1",
            model="gpt-4o",
            temperature=0.7,
        )

        mock_chat_cls.assert_called_once()
        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["api_key"] == "my-key"
        assert call_kwargs["base_url"] == "https://custom.api.com/v1"
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7

    def test_implements_llm_port(self) -> None:
        """LangchainOpenAIAdapter should implement LLMPort."""
        from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

        assert issubclass(LangchainOpenAIAdapter, LLMPort)
