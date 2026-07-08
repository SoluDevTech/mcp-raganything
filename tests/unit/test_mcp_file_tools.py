"""Tests for mcp_file_tools.py — file MCP tools registered with FastMCP."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import ToolError

from application.api.mcp_file_tools import (
    _validate_prefix,
    list_files,
    list_folders,
    mcp_files,
    read_file,
)
from domain.errors.storage import StorageNotFoundError
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata
from domain.ports.storage_port import FileInfo


class TestMCPFilesInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_files_has_correct_name(self) -> None:
        """mcp_files should be named 'Files'."""
        # Assert
        assert mcp_files.name == "Files"


class TestValidatePrefix:
    """Tests for the _validate_prefix helper."""

    def test_empty_prefix_returns_empty(self) -> None:
        # Act & Assert
        assert _validate_prefix("") == ""

    def test_valid_prefix_passes_through(self) -> None:
        # Act & Assert
        assert _validate_prefix("docs/reports/") == "docs/reports/"

    def test_trailing_slash_preserved(self) -> None:
        # Act & Assert
        assert _validate_prefix("docs/") == "docs/"

    def test_backslash_normalized(self) -> None:
        # Act & Assert
        assert _validate_prefix("docs\\reports/") == "docs/reports/"

    def test_dot_normalized_to_empty(self) -> None:
        # Act & Assert
        assert _validate_prefix(".") == ""

    def test_path_traversal_rejected(self) -> None:
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            _validate_prefix("../../etc")

    def test_absolute_path_rejected(self) -> None:
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            _validate_prefix("/etc/passwd")


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
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            result = await list_files(prefix="project/")

        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].object_name == "project/doc1.pdf"
        assert result[0].size == 1024

    async def test_uses_default_prefix_and_recursive(
        self, mock_files: list[FileInfo]
    ) -> None:
        """Should use default prefix='' and recursive=True."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            await list_files()

        # Assert
        mock_use_case.execute.assert_called_once_with(prefix="", recursive=True)

    async def test_calls_use_case_with_custom_prefix(
        self, mock_files: list[FileInfo]
    ) -> None:
        """Should forward custom prefix and recursive to use case."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_files

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            await list_files(prefix="reports/", recursive=False)

        # Assert
        mock_use_case.execute.assert_called_once_with(
            prefix="reports/", recursive=False
        )

    async def test_returns_empty_list_when_no_files(self) -> None:
        """Should return empty list when no files match prefix."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = []

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_files_use_case",
            return_value=mock_use_case,
        ):
            result = await list_files(prefix="nonexistent/")

        # Assert
        assert result == []

    async def test_rejects_path_traversal_prefix(self) -> None:
        """Should raise ToolError for path traversal in prefix."""
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            await list_files(prefix="../../etc")

    async def test_rejects_absolute_prefix(self) -> None:
        """Should raise ToolError for absolute path in prefix."""
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            await list_files(prefix="/etc/passwd")


class TestListFolders:
    """Tests for the list_folders MCP tool."""

    async def test_returns_folder_list(self) -> None:
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = ["docs/", "photos/"]

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_folders_use_case",
            return_value=mock_use_case,
        ):
            result = await list_folders()

        # Assert
        assert result == ["docs/", "photos/"]

    async def test_forwards_prefix_to_use_case(self) -> None:
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = ["reports/"]

        # Act
        with patch(
            "application.api.mcp_file_tools.get_list_folders_use_case",
            return_value=mock_use_case,
        ):
            result = await list_folders(prefix="docs/")

        # Assert
        mock_use_case.execute.assert_called_once_with(prefix="docs/")
        assert result == ["reports/"]

    async def test_rejects_path_traversal_prefix(self) -> None:
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            await list_folders(prefix="../../etc")

    async def test_rejects_absolute_prefix(self) -> None:
        # Act & Assert
        with pytest.raises(ToolError, match="prefix must be a relative path"):
            await list_folders(prefix="/etc/passwd")


class TestReadFile:
    """Tests for the read_file MCP tool."""

    @pytest.fixture
    def mock_document_content(self) -> DocumentContent:
        return DocumentContent(
            content=["Extracted text from the document."],
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )

    async def test_returns_file_content_response(
        self, mock_document_content: DocumentContent
    ) -> None:
        """Should call use_case.execute and return FileContentResponse."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_document_content

        # Act
        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ):
            result = await read_file(file_path="documents/report.pdf")

        # Assert
        assert result.content == ["Extracted text from the document."]
        assert result.metadata.mime_type == "application/pdf"

    async def test_raises_tool_error_for_file_not_found(self) -> None:
        """Should convert FileNotFoundError to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = StorageNotFoundError("not found")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_file_tools.get_read_file_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="File not found: missing.pdf"),
        ):
            await read_file(file_path="missing.pdf")

    async def test_raises_tool_error_for_generic_failure(self) -> None:
        """Should convert generic exceptions to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Disk full")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_file_tools.get_read_file_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="Failed to read file"),
        ):
            await read_file(file_path="documents/broken.pdf")

    async def test_includes_tables_in_response(self) -> None:
        """Should include tables in the FileContentResponse."""
        # Arrange
        from domain.ports.document_reader_port import TableData

        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = DocumentContent(
            content=["Report with table"],
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[TableData(markdown="| A | B |\n|---|---|")],
        )

        # Act
        with patch(
            "application.api.mcp_file_tools.get_read_file_use_case",
            return_value=mock_use_case,
        ):
            result = await read_file(file_path="docs/table.pdf")

        # Assert
        assert len(result.tables or []) == 1
        assert (result.tables or [])[0].markdown == "| A | B |\n|---|---|"
