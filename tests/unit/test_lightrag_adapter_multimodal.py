"""Tests for LightRAGAdapter.query_multimodal() and _build_vision_messages().

These methods exist in lightrag_adapter.py but have uncovered paths.
The RAG engine (RAGAnything) is an external dependency, so we mock it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.requests.query_request import MultimodalContentItem
from infrastructure.rag.lightrag_adapter import (
    LightRAGAdapter,
    _build_vision_messages,
)


@pytest.fixture
def mock_llm_config() -> MagicMock:
    config = MagicMock()
    config.CHAT_MODEL = "test-model"
    config.VISION_MODEL = "test-vision"
    config.EMBEDDING_MODEL = "test-embed"
    config.EMBEDDING_DIM = 1536
    config.MAX_TOKEN_SIZE = 8192
    config.api_key = "test-key"
    config.api_base_url = "http://test"
    return config


@pytest.fixture
def mock_rag_config() -> MagicMock:
    config = MagicMock()
    config.ENABLE_IMAGE_PROCESSING = True
    config.ENABLE_TABLE_PROCESSING = True
    config.ENABLE_EQUATION_PROCESSING = True
    config.MAX_CONCURRENT_FILES = 1
    config.COSINE_THRESHOLD = 0.2
    config.RAG_STORAGE_TYPE = "postgres"
    return config


@pytest.fixture
def adapter(mock_llm_config: MagicMock, mock_rag_config: MagicMock) -> LightRAGAdapter:
    return LightRAGAdapter(llm_config=mock_llm_config, rag_config=mock_rag_config)


class TestQueryMultimodal:
    """Tests for LightRAGAdapter.query_multimodal."""

    async def test_returns_multimodal_result(
        self, adapter: LightRAGAdapter
    ) -> None:
        """Should call rag.aquery_with_multimodal and return its string result."""
        mock_rag = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.aquery_with_multimodal.return_value = "Image shows a bar chart with sales data."

        # Pre-populate the rag dict so _ensure_initialized succeeds
        adapter.rag["/tmp/project"] = mock_rag

        content = [
            MultimodalContentItem(type="image", img_path="/tmp/chart.png"),
        ]

        result = await adapter.query_multimodal(
            query="What does this image show?",
            multimodal_content=content,
            mode="hybrid",
            top_k=5,
            working_dir="/tmp/project",
        )

        assert result == "Image shows a bar chart with sales data."
        mock_rag.aquery_with_multimodal.assert_called_once_with(
            query="What does this image show?",
            multimodal_content=[item.model_dump(exclude_none=True) for item in content],
            mode="hybrid",
            top_k=5,
        )

    async def test_raises_runtime_error_when_not_initialized(
        self, adapter: LightRAGAdapter
    ) -> None:
        """Should raise RuntimeError if working_dir was never initialized."""
        content = [
            MultimodalContentItem(type="image", img_path="/tmp/img.png"),
        ]

        with pytest.raises(
            RuntimeError,
            match="RAG engine not initialized.*Call init_project",
        ):
            await adapter.query_multimodal(
                query="test",
                multimodal_content=content,
                working_dir="/tmp/uninitialized",
            )

    async def test_passes_mode_and_top_k(
        self, adapter: LightRAGAdapter
    ) -> None:
        """Should forward mode and top_k to aquery_with_multimodal."""
        mock_rag = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.aquery_with_multimodal.return_value = "result"
        adapter.rag["/tmp/project"] = mock_rag

        content = [
            MultimodalContentItem(type="table", table_data="A,B\n1,2"),
        ]

        await adapter.query_multimodal(
            query="Analyze",
            multimodal_content=content,
            mode="local",
            top_k=20,
            working_dir="/tmp/project",
        )

        call_kwargs = mock_rag.aquery_with_multimodal.call_args[1]
        assert call_kwargs["mode"] == "local"
        assert call_kwargs["top_k"] == 20

    async def test_serializes_multimodal_items_excluding_none(
        self, adapter: LightRAGAdapter
    ) -> None:
        """Should serialize content items with exclude_none=True."""
        mock_rag = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.aquery_with_multimodal.return_value = "result"
        adapter.rag["/tmp/project"] = mock_rag

        content = [
            MultimodalContentItem(
                type="equation",
                latex="E = mc^2",
                equation_caption="Mass-energy equivalence",
            ),
        ]

        await adapter.query_multimodal(
            query="Explain",
            multimodal_content=content,
            working_dir="/tmp/project",
        )

        raw_content = mock_rag.aquery_with_multimodal.call_args[1]["multimodal_content"]
        # img_path and image_data should NOT be present since they are None
        assert "img_path" not in raw_content[0]
        assert "image_data" not in raw_content[0]
        assert raw_content[0]["type"] == "equation"
        assert raw_content[0]["latex"] == "E = mc^2"


class TestBuildVisionMessages:
    """Tests for the module-level _build_vision_messages helper."""

    def test_builds_messages_with_image_data_string(self) -> None:
        """Should wrap base64 image_data in a data URI."""
        messages = _build_vision_messages(
            system_prompt="You are a vision assistant.",
            history_messages=[],
            prompt="What is this?",
            image_data="iVBORw0KGgo=",
        )

        # system message + user message = 2
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a vision assistant."
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2  # text + image
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_builds_messages_with_image_url(self) -> None:
        """Should use http URLs directly without base64 prefix."""
        messages = _build_vision_messages(
            system_prompt=None,
            history_messages=[],
            prompt="Describe",
            image_data="https://example.com/image.png",
        )

        user_msg = messages[0]
        assert user_msg["content"][1]["image_url"]["url"] == "https://example.com/image.png"

    def test_builds_messages_with_multiple_images(self) -> None:
        """Should handle list of images in image_data."""
        messages = _build_vision_messages(
            system_prompt=None,
            history_messages=[],
            prompt="Compare these",
            image_data=["https://img1.png", "https://img2.png"],
        )

        user_msg = messages[0]
        # 1 text + 2 images = 3 content items
        assert len(user_msg["content"]) == 3
        assert user_msg["content"][1]["image_url"]["url"] == "https://img1.png"
        assert user_msg["content"][2]["image_url"]["url"] == "https://img2.png"

    def test_builds_messages_without_system_prompt(self) -> None:
        """Should skip system message when system_prompt is None."""
        messages = _build_vision_messages(
            system_prompt=None,
            history_messages=[],
            prompt="Hello",
            image_data=None,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_includes_history_messages(self) -> None:
        """Should include history messages before the user message."""
        history = [
            {"role": "assistant", "content": "Previous answer."},
            {"role": "user", "content": "Follow-up question."},
        ]

        messages = _build_vision_messages(
            system_prompt="Be helpful.",
            history_messages=history,
            prompt="Next question",
            image_data=None,
        )

        # system, history[0], history[1], user
        assert len(messages) == 4
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "user"

    def test_builds_text_only_when_no_image_data(self) -> None:
        """Should create a text-only user message when image_data is None."""
        messages = _build_vision_messages(
            system_prompt=None,
            history_messages=[],
            prompt="Just text",
            image_data=None,
        )

        user_msg = messages[0]
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0] == {"type": "text", "text": "Just text"}
