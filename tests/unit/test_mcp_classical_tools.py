"""Tests for MCP classical tools — TDD Red phase.

These tests will FAIL until the production code is implemented.
Tests cover FastMCP tools: classical_index_file, classical_index_folder, classical_query.
"""

from unittest.mock import AsyncMock, patch

import pytest

from application.responses.classical_query_response import ClassicalQueryResponse
from domain.entities.indexing_result import (
    FileIndexingResult,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)


class TestMCPClassicalToolsInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_classical_has_correct_name(self) -> None:
        """mcp_classical should be named 'RAGAnythingClassical'."""
        from application.api.mcp_classical_tools import mcp_classical

        assert mcp_classical.name == "RAGAnythingClassical"


class TestClassicalIndexFileTool:
    """Tests for the classical_index_file MCP tool."""

    async def test_calls_use_case_with_correct_params(self) -> None:
        """Should call use_case.execute with file_name, working_dir, and optional params."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="File indexed",
            file_path="/tmp/output/docs/report.pdf",
            file_name="docs/report.pdf",
            processing_time_ms=100.0,
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_file_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_file

            await classical_index_file(
                file_name="docs/report.pdf",
                working_dir="/tmp/rag/project_1",
            )

        mock_use_case.execute.assert_called_once_with(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_1",
            chunk_size=1000,
            chunk_overlap=200,
        )

    async def test_passes_custom_chunk_params(self) -> None:
        """Should forward chunk_size and chunk_overlap to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="File indexed",
            file_path="/tmp/output/report.pdf",
            file_name="report.pdf",
            processing_time_ms=50.0,
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_file_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_file

            await classical_index_file(
                file_name="report.pdf",
                working_dir="/tmp/rag/test",
                chunk_size=500,
                chunk_overlap=100,
            )

        mock_use_case.execute.assert_called_once_with(
            file_name="report.pdf",
            working_dir="/tmp/rag/test",
            chunk_size=500,
            chunk_overlap=100,
        )

    async def test_returns_file_indexing_result(self) -> None:
        """Should return the FileIndexingResult from the use case."""
        expected = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Document indexed successfully",
            file_path="/tmp/output/report.pdf",
            file_name="report.pdf",
            processing_time_ms=75.0,
        )
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = expected

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_file_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_file

            result = await classical_index_file(
                file_name="report.pdf",
                working_dir="/tmp/rag/project_1",
            )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "report.pdf"

    async def test_propagates_use_case_error(self) -> None:
        """Should let exceptions from the use case propagate."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = RuntimeError("Vector store down")

        with (
            patch(
                "application.api.mcp_classical_tools.get_classical_index_file_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(RuntimeError, match="Vector store down"),
        ):
            from application.api.mcp_classical_tools import classical_index_file

            await classical_index_file(
                file_name="report.pdf",
                working_dir="/tmp/rag/test",
            )


class TestClassicalIndexFolderTool:
    """Tests for the classical_index_folder MCP tool."""

    async def test_calls_use_case_with_correct_params(self) -> None:
        """Should call use_case.execute with working_dir and optional params."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Folder indexed",
            folder_path="project/docs",
            recursive=True,
            stats=FolderIndexingStats(
                total_files=2,
                files_processed=2,
                files_failed=0,
                files_skipped=0,
            ),
            processing_time_ms=200.0,
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_folder_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_folder

            await classical_index_folder(
                working_dir="project/docs",
                recursive=True,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="project/docs",
            recursive=True,
            file_extensions=None,
            chunk_size=1000,
            chunk_overlap=200,
        )

    async def test_passes_custom_params(self) -> None:
        """Should forward all parameters to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Folder indexed",
            folder_path="project/docs",
            recursive=False,
            stats=FolderIndexingStats(
                total_files=1,
                files_processed=1,
                files_failed=0,
                files_skipped=0,
            ),
            processing_time_ms=100.0,
        )

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_folder_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_folder

            await classical_index_folder(
                working_dir="project/docs",
                recursive=False,
                file_extensions=[".pdf"],
                chunk_size=500,
                chunk_overlap=50,
            )

        mock_use_case.execute.assert_called_once_with(
            working_dir="project/docs",
            recursive=False,
            file_extensions=[".pdf"],
            chunk_size=500,
            chunk_overlap=50,
        )

    async def test_returns_folder_indexing_result(self) -> None:
        """Should return the FolderIndexingResult from the use case."""
        expected = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Folder indexed successfully",
            folder_path="project/docs",
            recursive=True,
            stats=FolderIndexingStats(
                total_files=5,
                files_processed=5,
                files_failed=0,
                files_skipped=0,
            ),
            processing_time_ms=300.0,
        )
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = expected

        with patch(
            "application.api.mcp_classical_tools.get_classical_index_folder_use_case",
            return_value=mock_use_case,
        ):
            from application.api.mcp_classical_tools import classical_index_folder

            result = await classical_index_folder(
                working_dir="project/docs",
            )

        assert result.status == IndexingStatus.SUCCESS


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
            vector_distance_threshold=None,
            enable_llm_judge=True,
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
            vector_distance_threshold=None,
            enable_llm_judge=True,
        )

    async def test_returns_classical_query_response(self) -> None:
        """Should return the ClassicalQueryResponse from the use case."""
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

        assert result.status == "success"
        assert len(result.queries) == 3

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
