"""Tests for BricksApiAdapter — the httpx-based implementation of BricksApiPort.

The Bricks API is an external dependency, so we mock httpx responses
while testing our adapter logic for parsing, headers, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from domain.ports.bricks_api_port import BricksDocumentInfo, SectionVersionResult
from infrastructure.bricks.bricks_api_adapter import BricksApiAdapter


@pytest.fixture
def bricks_config() -> MagicMock:
    """Provide a mock BricksConfig."""
    config = MagicMock()
    config.BRICKS_API_BASE_URL = "https://api.bricks.example.com"
    config.BRICKS_API_KEY = "test-api-key-12345"
    config.BRICKS_BEARER_TOKEN = "test-bearer-token"
    config.BRICKS_PUBLISH_DRY_RUN = True
    config.BRICKS_PUBLISH_TARGET_URL = (
        "https://api.bricks.example.com/api/section-versions"
    )
    return config


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Provide a mocked httpx AsyncClient."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def adapter(bricks_config: MagicMock, mock_httpx_client: AsyncMock) -> BricksApiAdapter:
    """Provide a BricksApiAdapter with mocked httpx client."""
    with patch(
        "infrastructure.bricks.bricks_api_adapter.httpx.AsyncClient",
        return_value=mock_httpx_client,
    ):
        adapter = BricksApiAdapter(config=bricks_config)
    adapter._client = mock_httpx_client  # type: ignore[attr-defined]
    return adapter


class TestListProjectDocuments:
    """Tests for BricksApiAdapter.list_project_documents."""

    async def test_calls_api_with_bearer_token(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should call GET /api/projects/{id}/documents/ai with Bearer token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        await adapter.list_project_documents(project_id="proj-123")

        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert "api/projects/proj-123/documents/ai" in call_args[0][
            0
        ] or "proj-123/documents/ai" in str(call_args)
        headers = (
            call_args[1].get("headers", {}) or call_args[0][0]
            if len(call_args[0]) > 1
            else call_args[1].get("headers", {})
        )
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-bearer-token"

    async def test_returns_list_of_bricks_document_info(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should parse JSON response and return list of BricksDocumentInfo."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "doc-1",
                    "fileName": "report.pdf",
                    "url": "https://s3.example.com/presigned1",
                    "hash": "abc123",
                    "mimeType": "application/pdf",
                    "size": 1024,
                    "status": "PROCESSED",
                    "uploadedAt": "2025-12-03T17:02:50.916Z",
                    "imageAnalysisStatus": None,
                    "imageAnalysisConfidence": None,
                    "imageAnalysisReasoning": None,
                    "imageAnalysisDate": None,
                    "category": None,
                    "categoryConfidence": None,
                    "categorySource": None,
                    "categoryClassifiedAt": None,
                    "isIgnored": False,
                    "projectId": "proj-123",
                },
                {
                    "id": "doc-2",
                    "fileName": "notes.docx",
                    "url": "https://s3.example.com/presigned2",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "size": 2048,
                    "status": "PROCESSED",
                    "isIgnored": True,
                    "projectId": "proj-123",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.list_project_documents(project_id="proj-123")

        assert len(result) == 2
        assert isinstance(result[0], BricksDocumentInfo)
        assert result[0].file_name == "report.pdf"
        assert result[0].mime_type == "application/pdf"
        assert result[0].size == 1024
        assert result[1].file_name == "notes.docx"
        assert result[1].is_ignored is True

    async def test_camel_case_to_snake_case_mapping(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should correctly map camelCase API fields to snake_case model fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "doc-uuid",
                    "fileName": "image.png",
                    "url": "https://s3.example.com/img",
                    "imageAnalysisStatus": "relevant",
                    "imageAnalysisConfidence": 95.0,
                    "imageAnalysisReasoning": "Contains relevant diagrams",
                    "imageAnalysisDate": "2025-12-04T10:00:00.000Z",
                    "categoryConfidence": 72.3,
                    "categorySource": "llm",
                    "categoryClassifiedAt": "2025-12-05T08:00:00.000Z",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.list_project_documents(project_id="proj-123")

        doc = result[0]
        assert doc.image_analysis_status == "relevant"
        assert doc.image_analysis_confidence == 95.0
        assert doc.image_analysis_reasoning == "Contains relevant diagrams"
        assert doc.image_analysis_date == "2025-12-04T10:00:00.000Z"
        assert doc.category_confidence == 72.3
        assert doc.category_source == "llm"
        assert doc.category_classified_at == "2025-12-05T08:00:00.000Z"

    async def test_returns_empty_list_when_no_documents(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should return empty list when API returns empty items."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.list_project_documents(project_id="proj-empty")

        assert result == []

    async def test_raises_on_401_unauthorized(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise proper exception on 401 Unauthorized."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.return_value = mock_response

        with pytest.raises(PermissionError):
            await adapter.list_project_documents(project_id="proj-123")

    async def test_raises_on_403_forbidden(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise PermissionError on 403 Forbidden."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Forbidden",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.return_value = mock_response

        with pytest.raises(PermissionError):
            await adapter.list_project_documents(project_id="proj-123")

    async def test_raises_on_404_not_found(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise FileNotFoundError on 404 Not Found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Not Found",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.return_value = mock_response

        with pytest.raises(FileNotFoundError):
            await adapter.list_project_documents(project_id="nonexistent")

    async def test_raises_on_timeout(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise TimeoutError on request timeout."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(TimeoutError):
            await adapter.list_project_documents(project_id="proj-123")


class TestDownloadDocument:
    """Tests for BricksApiAdapter.download_document."""

    async def test_downloads_from_s3_url(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should download file content from S3 presigned URL (no auth header)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"PDF binary content"
        mock_response.headers = {
            "content-disposition": 'attachment; filename="report.pdf"'
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        content, filename = await adapter.download_document(
            download_url="https://s3.example.com/presigned-url"
        )

        assert content == b"PDF binary content"
        assert filename == "report.pdf"

    async def test_extracts_filename_from_content_disposition(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should extract filename from Content-Disposition header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"data"
        mock_response.headers = {
            "content-disposition": 'attachment; filename="my document.docx"'
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        _, filename = await adapter.download_document(
            download_url="https://s3.example.com/doc"
        )

        assert filename == "my document.docx"

    async def test_no_auth_header_on_s3_download(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """S3 presigned URLs should not include authentication headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"data"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        await adapter.download_document(
            download_url="https://s3.example.com/presigned-url"
        )

        call_args = mock_httpx_client.get.call_args
        headers = call_args[1].get("headers", {})
        assert "Authorization" not in headers
        assert "X-API-Key" not in headers

    async def test_raises_on_download_failure(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise exception on download failure."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Forbidden",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.return_value = mock_response

        with pytest.raises(RuntimeError):
            await adapter.download_document(
                download_url="https://s3.example.com/expired-url"
            )

    async def test_raises_on_timeout(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise TimeoutError on download timeout."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Download timed out")

        with pytest.raises(TimeoutError):
            await adapter.download_document(
                download_url="https://s3.example.com/slow-url"
            )


class TestPublishSectionVersion:
    """Tests for BricksApiAdapter.publish_section_version."""

    async def test_calls_post_with_x_api_key_header(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should call POST /api/section-versions with X-API-Key header."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "sv-1",
            "sectionKey": "intro",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_response

        payload = {
            "project_unique_id": "proj-123",
            "section_key": "intro",
            "content": "Hello world",
            "workflow_id": "wf-1",
            "workflow_name": "draft",
            "workflow_metadata": {},
        }

        await adapter.publish_section_version(payload=payload)

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        headers = call_args[1].get("headers", {})
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == "test-api-key-12345"

    async def test_returns_section_version_result(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should return SectionVersionResult on successful publish."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "sv-uuid",
            "sectionKey": "summary",
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_response

        result = await adapter.publish_section_version(
            payload={"section_key": "summary", "content": "Summary text"}
        )

        assert isinstance(result, SectionVersionResult)
        assert result.success is True

    async def test_raises_on_401_unauthorized(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise proper exception on 401 when publishing."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.post.return_value = mock_response

        with pytest.raises(PermissionError):
            await adapter.publish_section_version(payload={"section_key": "intro"})

    async def test_raises_on_timeout(
        self, adapter: BricksApiAdapter, mock_httpx_client: AsyncMock
    ) -> None:
        """Should raise TimeoutError on publish timeout."""
        mock_httpx_client.post.side_effect = httpx.TimeoutException("Publish timed out")

        with pytest.raises(TimeoutError):
            await adapter.publish_section_version(payload={"section_key": "intro"})
