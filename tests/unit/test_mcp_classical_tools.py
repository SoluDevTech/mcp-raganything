"""Tests for MCP classical tools.

Tests cover FastMCP tools: classical_query.
"""

from unittest.mock import AsyncMock, patch

import pytest

from application.responses.classical_query_response import ClassicalQueryResponse


class TestMCPClassicalToolsInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_classical_has_correct_name(self) -> None:
        """mcp_classical should be named 'RAGAnythingClassical'."""
        from application.api.mcp_classical_tools import mcp_classical

        assert mcp_classical.name == "RAGAnythingClassical"


class TestClassicalQueryTool:
    """Tests for the classical_query MCP tool."""

    async def test_calls_use_case_with_correct_params(self) -> None:
        """Should call use_case.execute with working_dir, query, and optional params."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = ClassicalQueryResponse(
            status="success",
            message="",
            queries=["What is ML?"],
            chunks=[],
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_query_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_query

            await classical_query(
                working_dir="/tmp/rag/project_1",
                query="What is machine learning?",
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            top_k=10,
            num_variations=3,
            relevance_threshold=5.0,
            vector_distance_threshold=0.5,
            enable_llm_judge=True,
            mode="vector",
        )

    async def test_passes_custom_params(self) -> None:
        """Should forward custom top_k, num_variations, relevance_threshold."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = ClassicalQueryResponse(
            status="success",
            queries=["test"],
            chunks=[],
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_query_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_query

            await classical_query(
                working_dir="/tmp/rag/project_42",
                query="Find relevant info",
                top_k=20,
                num_variations=5,
                relevance_threshold=7.0,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_42",
            query="Find relevant info",
            top_k=20,
            num_variations=5,
            relevance_threshold=7.0,
            vector_distance_threshold=0.5,
            enable_llm_judge=True,
            mode="vector",
        )

    async def test_returns_classical_query_response(self) -> None:
        """Should return a result from the use case (FastMCP serializes the response)."""
        expected = ClassicalQueryResponse(
            status="success",
            message="Found 3 relevant chunks",
            queries=["What is ML?", "Define machine learning", "Explain ML concepts"],
            chunks=[],
        )
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = expected

        with patch(
            "application.api.mcp_classical_tools.get_classical_query_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_query

            result = await classical_query(
                working_dir="/tmp/rag/project_1",
                query="What is ML?",
            )

        import application.responses.classical_query_response as cqr
        assert isinstance(result, cqr.McpClassicalRagResponse)
        assert result.rag_response == []

    async def test_propagates_use_case_error(self) -> None:
        """Should let exceptions from the use case propagate."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = RuntimeError("LLM unavailable")

        with (
            patch(
                "application.api.mcp_classical_tools.get_classical_query_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(RuntimeError, match="LLM unavailable"),
        ):
            from application.api.mcp_classical_tools import classical_query

            await classical_query(
                working_dir="/tmp/rag/test",
                query="test",
            )
