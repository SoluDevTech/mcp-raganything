"""Tests for mcp_query_tools.py — query MCP tools registered with FastMCP."""

from unittest.mock import AsyncMock, patch

import pytest

from application.api.mcp_query_tools import (
    mcp_query,
    query_knowledge_base,
    query_knowledge_base_multimodal,
)
from application.requests.query_request import MultimodalContentItem
from application.responses.query_response import ChunkResponse


class TestMCPQueryInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_query_has_correct_name(self) -> None:
        """mcp_query should be named 'RAGAnythingQuery'."""
        assert mcp_query.name == "RAGAnythingQuery"


class TestQueryKnowledgeBase:
    """Tests for the query_knowledge_base MCP tool."""

    @pytest.fixture
    def mock_query_result(self) -> dict:
        return {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": "Relevant chunk content",
                        "file_path": "/docs/report.pdf",
                    },
                ],
                "references": [],
            },
        }

    async def test_returns_chunks_from_use_case(self, mock_query_result: dict) -> None:
        """Should call use_case.execute and return the chunk list."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            result = await query_knowledge_base(
                working_dir="/tmp/rag/project_1",
                query="What is the summary?",
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ChunkResponse)
        assert result[0].content == "Relevant chunk content"

    async def test_calls_use_case_with_defaults(self, mock_query_result: dict) -> None:
        """Should pass default mode='hybrid' and top_k=5 to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="test query",
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="test query",
            mode="hybrid",
            top_k=5,
        )

    async def test_calls_use_case_with_custom_mode_and_top_k(
        self, mock_query_result: dict
    ) -> None:
        """Should forward custom mode and top_k to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/project_42",
                query="What are the findings?",
                mode="local",
                top_k=20,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_42",
            query="What are the findings?",
            mode="local",
            top_k=20,
        )

    async def test_handles_naive_mode(self, mock_query_result: dict) -> None:
        """Should work with naive mode (vector search only)."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            result = await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="search",
                mode="naive",
            )

        assert isinstance(result, list)
        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="search",
            mode="naive",
            top_k=5,
        )

    async def test_handles_global_mode(self, mock_query_result: dict) -> None:
        """Should work with global mode."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="search",
                mode="global",
                top_k=10,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="search",
            mode="global",
            top_k=10,
        )

    async def test_handles_mix_mode(self, mock_query_result: dict) -> None:
        """Should work with mix mode."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="search",
                mode="mix",
                top_k=15,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="search",
            mode="mix",
            top_k=15,
        )

    async def test_handles_hybrid_plus_mode(self, mock_query_result: dict) -> None:
        """Should work with hybrid+ mode (BM25 + vector)."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_query_result

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="search",
                mode="hybrid+",
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="search",
            mode="hybrid+",
            top_k=5,
        )

    async def test_returns_empty_chunks_when_no_results(self) -> None:
        """Should return empty list when query returns no chunks."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
        }

        with patch(
            "application.api.mcp_query_tools.get_query_use_case",
            return_value=mock_use_case,
        ):
            result = await query_knowledge_base(
                working_dir="/tmp/rag/empty",
                query="nothing matches",
            )

        assert result == []

    async def test_propagates_use_case_error(self) -> None:
        """Should let exceptions from the use case propagate."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = RuntimeError("RAG engine failure")

        with (
            patch(
                "application.api.mcp_query_tools.get_query_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(RuntimeError, match="RAG engine failure"),
        ):
            await query_knowledge_base(
                working_dir="/tmp/rag/test",
                query="will fail",
            )


class TestQueryKnowledgeBaseMultimodal:
    """Tests for the query_knowledge_base_multimodal MCP tool."""

    @pytest.fixture
    def multimodal_content(self) -> list[MultimodalContentItem]:
        return [
            MultimodalContentItem(type="image", img_path="/tmp/images/chart.png"),
        ]

    async def test_returns_result_from_use_case(
        self, multimodal_content: list[MultimodalContentItem]
    ) -> None:
        """Should call multimodal use case and return its result."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = {
            "status": "success",
            "data": "The chart shows increasing revenue.",
        }

        with patch(
            "application.api.mcp_query_tools.get_multimodal_query_use_case",
            return_value=mock_use_case,
        ):
            result = await query_knowledge_base_multimodal(
                working_dir="/tmp/rag/project_1",
                query="What does this image show?",
                multimodal_content=multimodal_content,
            )

        assert result["status"] == "success"
        assert result["data"] == "The chart shows increasing revenue."

    async def test_calls_use_case_with_defaults(
        self, multimodal_content: list[MultimodalContentItem]
    ) -> None:
        """Should pass default mode='hybrid' and top_k=5."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = {"status": "success", "data": ""}

        with patch(
            "application.api.mcp_query_tools.get_multimodal_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base_multimodal(
                working_dir="/tmp/rag/test",
                query="Describe",
                multimodal_content=multimodal_content,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="Describe",
            multimodal_content=multimodal_content,
            mode="hybrid",
            top_k=5,
        )

    async def test_calls_use_case_with_custom_params(
        self,
        multimodal_content: list[MultimodalContentItem],  # noqa: ARG002
    ) -> None:
        """Should forward custom mode and top_k to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = {"status": "success", "data": ""}

        with patch(
            "application.api.mcp_query_tools.get_multimodal_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base_multimodal(
                working_dir="/tmp/rag/project_42",
                query="Analyze this table",
                multimodal_content=[
                    MultimodalContentItem(
                        type="table",
                        table_data="A,B\n1,2",
                        table_caption="Test table",
                    ),
                ],
                mode="global",
                top_k=20,
            )

        call_kwargs = mock_use_case.execute.call_args[1]
        assert call_kwargs["working_dir"] == "/tmp/rag/project_42"
        assert call_kwargs["query"] == "Analyze this table"
        assert call_kwargs["mode"] == "global"
        assert call_kwargs["top_k"] == 20
        assert len(call_kwargs["multimodal_content"]) == 1

    async def test_handles_naive_mode(
        self, multimodal_content: list[MultimodalContentItem]
    ) -> None:
        """Should work with naive mode."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = {"status": "success", "data": ""}

        with patch(
            "application.api.mcp_query_tools.get_multimodal_query_use_case",
            return_value=mock_use_case,
        ):
            await query_knowledge_base_multimodal(
                working_dir="/tmp/rag/test",
                query="search",
                multimodal_content=multimodal_content,
                mode="naive",
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="search",
            multimodal_content=multimodal_content,
            mode="naive",
            top_k=5,
        )

    async def test_propagates_use_case_error(
        self, multimodal_content: list[MultimodalContentItem]
    ) -> None:
        """Should let exceptions from the use case propagate."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = RuntimeError("Vision model failed")

        with (
            patch(
                "application.api.mcp_query_tools.get_multimodal_query_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(RuntimeError, match="Vision model failed"),
        ):
            await query_knowledge_base_multimodal(
                working_dir="/tmp/rag/test",
                query="will fail",
                multimodal_content=multimodal_content,
            )
