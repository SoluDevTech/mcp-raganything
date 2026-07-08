"""Tests for mcp_bricks_tools.py — Bricks MCP tools registered with FastMCP.

Follows the exact pattern of test_mcp_file_tools.py.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import ToolError

from application.api.mcp_bricks_tools import (
    list_bricks_documents,
    mcp_bricks,
    publish_section_version,
    read_bricks_document,
)
from domain.errors.bricks import BricksConnectionError
from domain.ports.bricks_api_port import BricksDocumentInfo, SectionVersionResult
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata


class TestMCPBricksInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_bricks_has_correct_name(self) -> None:
        """mcp_bricks should be named 'Bricks'."""
        # Assert
        assert mcp_bricks.name == "Bricks"


class TestListBricksDocuments:
    """Tests for the list_bricks_documents MCP tool."""

    @pytest.fixture
    def mock_documents(self) -> list[BricksDocumentInfo]:
        return [
            BricksDocumentInfo(
                id="doc-1",
                fileName="report.pdf",
                url="https://s3.example.com/doc1.pdf",
                mimeType="application/pdf",
                size=1024,
                status="PROCESSED",
            ),
            BricksDocumentInfo(
                id="doc-2",
                fileName="notes.docx",
                url="https://s3.example.com/doc2.docx",
                mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=2048,
                status="PROCESSED",
            ),
        ]

    async def test_returns_document_list(
        self, mock_documents: list[BricksDocumentInfo]
    ) -> None:
        """Should call use_case.execute and return document list."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_documents

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            result = await list_bricks_documents(project_unique_id="proj-123")

        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].file_name == "report.pdf"
        assert result[0].size == 1024

    async def test_calls_use_case_with_project_id(
        self, mock_documents: list[BricksDocumentInfo]
    ) -> None:
        """Should forward project_unique_id to the use case."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_documents

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            await list_bricks_documents(project_unique_id="my-project")

        # Assert
        mock_use_case.execute.assert_called_once_with(project_id="my-project")

    async def test_returns_empty_list_when_no_documents(self) -> None:
        """Should return empty list when no documents exist."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = []

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            result = await list_bricks_documents(project_unique_id="empty-proj")

        # Assert
        assert result == []

    async def test_raises_tool_error_for_api_failure(self) -> None:
        """Should convert API errors to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = BricksConnectionError("API unreachable")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="API unreachable"),
        ):
            await list_bricks_documents(project_unique_id="proj-123")


class TestReadBricksDocument:
    """Tests for the read_bricks_document MCP tool."""

    @pytest.fixture
    def mock_document_content(self) -> DocumentContent:
        return DocumentContent(
            content=["Extracted text from bricks document."],
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
            "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
            return_value=mock_use_case,
        ):
            result = await read_bricks_document(
                document_id="doc-abc123", project_unique_id="proj-456"
            )

        # Assert
        assert result.content == ["Extracted text from bricks document."]
        assert result.metadata.mime_type == "application/pdf"

    async def test_raises_tool_error_for_download_failure(self) -> None:
        """Should convert download errors to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = BricksConnectionError("S3 download failed")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="S3 download failed"),
        ):
            await read_bricks_document(
                document_id="doc-expired", project_unique_id="proj-1"
            )

    async def test_raises_tool_error_for_generic_failure(self) -> None:
        """Should convert generic exceptions to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Unexpected error")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="Failed to read bricks document"),
        ):
            await read_bricks_document(
                document_id="doc-broken", project_unique_id="proj-1"
            )

    async def test_includes_tables_in_response(self) -> None:
        """Should include tables in the response."""
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
            "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
            return_value=mock_use_case,
        ):
            result = await read_bricks_document(
                document_id="doc-table", project_unique_id="proj-1"
            )

        # Assert
        assert len(result.tables or []) == 1
        assert (result.tables or [])[0].markdown == "| A | B |\n|---|---|"


class TestPublishSectionVersion:
    """Tests for the publish_section_version MCP tool."""

    async def test_returns_section_version_result(self) -> None:
        """Should call use_case.execute and return SectionVersionResult."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(
            success=True,
            message="Dry run — no API call made",
            dry_run=True,
            payload_preview={
                "section_key": "consolidated_data",
                "content": {"text": "Hello"},
            },
        )

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            result = await publish_section_version(
                project_unique_id="proj-123",
                content={"text": "Hello world"},
                field_sources=[],
            )

        # Assert
        assert result.success is True
        assert result.dry_run is True

    async def test_forward_all_params_to_use_case(self) -> None:
        """Should forward all parameters to the use case with hardcoded defaults."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(success=True)

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            await publish_section_version(
                project_unique_id="proj-abc",
                content={"text": "Summary content"},
                field_sources=[{"field": "text", "source": "doc-1"}],
            )

        # Assert
        call_args = mock_use_case.execute.call_args
        assert call_args.kwargs["project_unique_id"] == "proj-abc"
        assert call_args.kwargs["section_key"] == "consolidated_data"
        assert call_args.kwargs["content"] == {"text": "Summary content"}
        assert call_args.kwargs["field_sources"] == [
            {"field": "text", "source": "doc-1"}
        ]
        assert call_args.kwargs["workflow_id"] == "agent-haiku-files-v1"
        assert call_args.kwargs["workflow_name"] == "agent-haiku-files-v1"
        assert call_args.kwargs["workflow_metadata"] is not None
        assert "execution_id" in call_args.kwargs["workflow_metadata"]
        assert "executed_at" in call_args.kwargs["workflow_metadata"]
        assert call_args.kwargs["workflow_metadata"]["version"] == "1.0.0"

    async def test_raises_tool_error_for_api_failure(self) -> None:
        """Should convert API errors to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = BricksConnectionError(
            "API connection failed"
        )

        # Act & Assert
        with (
            patch(
                "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="API connection failed"),
        ):
            await publish_section_version(
                project_unique_id="proj-123",
                content={"text": "Hello"},
                field_sources=[],
            )

    async def test_raises_tool_error_for_generic_failure(self) -> None:
        """Should convert generic exceptions to ToolError."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Unexpected error")

        # Act & Assert
        with (
            patch(
                "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(
                ToolError, match="Failed to publish section version: Unexpected error"
            ),
        ):
            await publish_section_version(
                project_unique_id="proj-123",
                content={"text": "Hello"},
                field_sources=[],
            )

    async def test_returns_dry_run_result_with_preview(self) -> None:
        """Should return payload_preview when dry_run is True."""
        # Arrange
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(
            success=True,
            message="Dry run",
            dry_run=True,
            payload_preview={
                "projectUniqueId": "proj-123",
                "sectionKey": "consolidated_data",
                "content": {"text": "Hello"},
                "fieldSources": [],
                "workflowId": "agent-haiku-files-v1",
                "workflowName": "agent-haiku-files-v1",
                "workflowMetadata": {},
            },
        )

        # Act
        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            result = await publish_section_version(
                project_unique_id="proj-123",
                content={"text": "Hello"},
                field_sources=[],
            )

        # Assert
        assert result.dry_run is True
        assert result.payload_preview is not None
        assert result.payload_preview["sectionKey"] == "consolidated_data"
