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
from domain.ports.bricks_api_port import BricksDocumentInfo, SectionVersionResult
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata


class TestMCPBricksInstance:
    """Verify the FastMCP instance configuration."""

    def test_mcp_bricks_has_correct_name(self) -> None:
        """mcp_bricks should be named 'RAGAnythingBricks'."""
        assert mcp_bricks.name == "RAGAnythingBricks"


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
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_documents

        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            result = await list_bricks_documents(project_unique_id="proj-123")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].file_name == "report.pdf"
        assert result[0].size == 1024

    async def test_calls_use_case_with_project_id(
        self, mock_documents: list[BricksDocumentInfo]
    ) -> None:
        """Should forward project_unique_id to the use case."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = mock_documents

        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            await list_bricks_documents(project_unique_id="my-project")

        mock_use_case.execute.assert_called_once_with(project_id="my-project")

    async def test_returns_empty_list_when_no_documents(self) -> None:
        """Should return empty list when no documents exist."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = []

        with patch(
            "application.api.mcp_bricks_tools.get_list_bricks_documents_use_case",
            return_value=mock_use_case,
        ):
            result = await list_bricks_documents(project_unique_id="empty-proj")

        assert result == []

    async def test_raises_tool_error_for_api_failure(self) -> None:
        """Should convert API errors to ToolError."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = ConnectionError("API unreachable")

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
            content="Extracted text from bricks document.",
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
            "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
            return_value=mock_use_case,
        ):
            result = await read_bricks_document(
                document_id="doc-abc123", project_unique_id="proj-456"
            )

        assert result.content == "Extracted text from bricks document."
        assert result.metadata.mime_type == "application/pdf"

    async def test_raises_tool_error_for_download_failure(self) -> None:
        """Should convert download errors to ToolError."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = ConnectionError("S3 download failed")

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
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Unexpected error")

        with (
            patch(
                "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="Failed to read bricks document"),
        ):
            await read_bricks_document(document_id="doc-broken", project_unique_id="proj-1")

    async def test_includes_tables_in_response(self) -> None:
        """Should include tables in the response."""
        from domain.ports.document_reader_port import TableData

        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = DocumentContent(
            content="Report with table",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[TableData(markdown="| A | B |\n|---|---|")],
        )

        with patch(
            "application.api.mcp_bricks_tools.get_read_bricks_document_use_case",
            return_value=mock_use_case,
        ):
            result = await read_bricks_document(
                document_id="doc-table", project_unique_id="proj-1"
            )

        assert len(result.tables) == 1
        assert result.tables[0].markdown == "| A | B |\n|---|---|"


class TestPublishSectionVersion:
    """Tests for the publish_section_version MCP tool."""

    async def test_returns_section_version_result(self) -> None:
        """Should call use_case.execute and return SectionVersionResult."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(
            success=True,
            message="Dry run — no API call made",
            dry_run=True,
            payload_preview={"section_key": "consolidated_data", "content": "Hello"},
        )

        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            result = await publish_section_version(
                project_unique_id="proj-123",
                content="Hello world",
            )

        assert result.success is True
        assert result.dry_run is True

    async def test_forward_all_params_to_use_case(self) -> None:
        """Should forward all parameters to the use case with hardcoded defaults."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(success=True)

        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            await publish_section_version(
                project_unique_id="proj-abc",
                content="Summary content",
            )

        call_args = mock_use_case.execute.call_args
        assert call_args.kwargs["project_unique_id"] == "proj-abc"
        assert call_args.kwargs["section_key"] == "consolidated_data"
        assert call_args.kwargs["content"] == "Summary content"
        assert call_args.kwargs["workflow_id"] == "agent-haiku-files-v1"
        assert call_args.kwargs["workflow_name"] == "agent-haiku-files-v1"
        assert call_args.kwargs["workflow_metadata"] is not None
        assert "execution_id" in call_args.kwargs["workflow_metadata"]
        assert "executed_at" in call_args.kwargs["workflow_metadata"]
        assert call_args.kwargs["workflow_metadata"]["version"] == "1.0.0"

    async def test_raises_tool_error_for_api_failure(self) -> None:
        """Should convert API errors to ToolError."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = ConnectionError("API connection failed")

        with (
            patch(
                "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="API connection failed"),
        ):
            await publish_section_version(
                project_unique_id="proj-123",
                content="Hello",
            )

    async def test_raises_tool_error_for_generic_failure(self) -> None:
        """Should convert generic exceptions to ToolError."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.side_effect = Exception("Unexpected error")

        with (
            patch(
                "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
                return_value=mock_use_case,
            ),
            pytest.raises(ToolError, match="Failed to publish section version: Unexpected error"),
        ):
            await publish_section_version(
                project_unique_id="proj-123",
                content="Hello",
            )

    async def test_returns_dry_run_result_with_preview(self) -> None:
        """Should return payload_preview when dry_run is True."""
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = SectionVersionResult(
            success=True,
            message="Dry run",
            dry_run=True,
            payload_preview={
                "projectUniqueId": "proj-123",
                "sectionKey": "consolidated_data",
                "content": "Hello",
                "workflowId": "agent-haiku-files-v1",
                "workflowName": "agent-haiku-files-v1",
                "workflowMetadata": {},
            },
        )

        with patch(
            "application.api.mcp_bricks_tools.get_publish_section_version_use_case",
            return_value=mock_use_case,
        ):
            result = await publish_section_version(
                project_unique_id="proj-123",
                content="Hello",
            )

        assert result.dry_run is True
        assert result.payload_preview is not None
        assert result.payload_preview["sectionKey"] == "consolidated_data"
