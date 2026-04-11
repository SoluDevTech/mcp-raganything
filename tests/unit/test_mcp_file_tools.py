"""Tests for mcp_file_tools.py — file MCP tools registered with FastMCP."""

from unittest.mock import AsyncMock, patch

import pytest

from application.api.mcp_file_tools import (
    list_files,
    mcp_files,
    read_file,
)
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata
from domain.ports.storage_port import FileInfo


class TestMCPFilesInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_files_has_correct_name(self) -> None:
        """mcp_files should be named 'RAGAnythingFiles'."""
        assert mcp_files.name == "RAGAnythingFiles"


class TestListFiles:
    """Tests for the list_files MCP tool."""

    @pytest.fixture
    def mock_files(self) -> list[FileInfo]:
        return [
            FileInfo(
                object_name="project/doc1.pdf",
                size=1024,
                last_modified="2026-01-01 00:00:00+00:00",
            ),
            FileInfo(
                object_name="project/doc2.pdf",
                size=2048,
                last_modified="2026-01-02 00:00:00+00:00",
            ),
        ]

    async def test_returns_file_info_list(self, mock_files: list[FileInfo]) -> None:
        """Should call use_case.execute and return file info list."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            result = await list_files(prefix="project/")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].object_name == "project/doc1.pdf"
        assert result[0].size == 1024

    async def test_uses_default_prefix_and_recursive(
        self, mock_files: list[FileInfo]
    ) -> None:
        """Should use default prefix='' and recursive=True."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            await list_files()

        mock_use_case.execute.assert_called_once_with(prefix="", recursive=True)

    async def test_calls_use_case_with_custom_prefix(
        self, mock_files: list[FileInfo]
    ) -> None:
        """Should forward custom prefix and recursive to use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            await list_files(prefix="reports/", recursive=False)

        mock_use_case.execute.assert_called_once_with(
            prefix="reports/", recursive=False
        )

    async def test_returns_empty_list_when_no_files(self) -> None:
        """Should return empty list when no files match prefix."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = []

        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            result = await list_files(prefix="nonexistent/")

        assert result == []


class TestReadFile:
    """Tests for the read_file MCP tool."""

    @pytest.fixture
    def mock_document_content(self) -> DocumentContent:
        return DocumentContent(
            content="Extracted text from the document.",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )

    async def test_returns_file_content_response(
        self, mock_document_content: DocumentContent
    ) -> None:
        """Should call use_case.execute and return FileContentResponse."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_document_content

        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ):
            result = await read_file(file_path="documents/report.pdf")

        assert result.content == "Extracted text from the document."
        assert result.metadata.mime_type == "application/pdf"

    async def test_raises_value_error_for_file_not_found(self) -> None:
        """Should convert FileNotFoundError to ValueError with helpful message."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = FileNotFoundError

        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ), pytest.raises(ValueError, match="File not found: missing.pdf"):
            await read_file(file_path="missing.pdf")

    async def test_raises_runtime_error_for_generic_failure(self) -> None:
        """Should convert generic exceptions to RuntimeError."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Disk full")

        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ), pytest.raises(RuntimeError, match="Failed to read file"):
            await read_file(file_path="documents/broken.pdf")

    async def test_includes_tables_in_response(self) -> None:
        """Should include tables in the FileContentResponse."""
        from domain.ports.document_reader_port import TableData

        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = DocumentContent(
            content="Report with table",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[TableData(markdown="| A | B |\n|---|---|")],
        )

        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ):
            result = await read_file(file_path="docs/table.pdf")

        assert len(result.tables) == 1
        assert result.tables[0].markdown == "| A | B |\n|---|---|"
