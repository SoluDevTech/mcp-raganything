"""Tests for ListBricksDocumentsUseCase — simple delegation to BricksApiPort."""

from unittest.mock import AsyncMock

import pytest

from application.use_cases.list_bricks_documents_use_case import (
    ListBricksDocumentsUseCase,
)
from domain.ports.bricks_api_port import BricksDocumentInfo


class TestListBricksDocumentsUseCase:
    """Tests for ListBricksDocumentsUseCase."""

    @pytest.fixture
    def use_case(self, mock_bricks_api: AsyncMock) -> ListBricksDocumentsUseCase:
        return ListBricksDocumentsUseCase(bricks_api=mock_bricks_api)

    async def test_execute_delegates_to_port(
        self,
        use_case: ListBricksDocumentsUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should delegate to bricks_api.list_project_documents."""
        await use_case.execute(project_id="proj-123")

        mock_bricks_api.list_project_documents.assert_called_once_with(
            project_id="proj-123"
        )

    async def test_execute_returns_list_of_document_infos(
        self,
        use_case: ListBricksDocumentsUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should return the list of BricksDocumentInfo from the port."""
        expected = [
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
            ),
        ]
        mock_bricks_api.list_project_documents.return_value = expected

        result = await use_case.execute(project_id="proj-123")

        assert len(result) == 2
        assert result[0].file_name == "report.pdf"
        assert result[1].file_name == "notes.docx"

    async def test_execute_returns_empty_list(
        self,
        use_case: ListBricksDocumentsUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should return empty list when no documents exist."""
        mock_bricks_api.list_project_documents.return_value = []

        result = await use_case.execute(project_id="empty-project")

        assert result == []

    async def test_execute_propagates_errors(
        self,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should propagate errors from the API port."""
        mock_bricks_api.list_project_documents.side_effect = ConnectionError(
            "API unreachable"
        )
        use_case = ListBricksDocumentsUseCase(bricks_api=mock_bricks_api)

        with pytest.raises(ConnectionError, match="API unreachable"):
            await use_case.execute(project_id="proj-123")

    async def test_execute_propagates_timeout(
        self,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should propagate TimeoutError from the API port."""
        mock_bricks_api.list_project_documents.side_effect = TimeoutError(
            "Request timed out"
        )
        use_case = ListBricksDocumentsUseCase(bricks_api=mock_bricks_api)

        with pytest.raises(TimeoutError):
            await use_case.execute(project_id="proj-123")
