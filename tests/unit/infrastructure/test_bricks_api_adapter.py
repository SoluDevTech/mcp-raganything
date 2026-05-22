"""Tests for BricksApiAdapter — the urllib-based implementation of BricksApiPort.

The Bricks API is an external dependency protected by Cloudflare,
so we mock urllib.request.urlopen responses while testing our adapter
logic for parsing, headers, and error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import urllib.error

from domain.ports.bricks_api_port import BricksDocumentInfo, SectionVersionResult
from infrastructure.bricks.bricks_api_adapter import _MAX_RETRIES
from infrastructure.bricks.bricks_api_adapter import (
    BricksApiAdapter,
    _extract_filename,
    _normalize_extension,
)


@pytest.fixture
def bricks_config() -> MagicMock:
    config = MagicMock()
    config.BRICKS_API_BASE_URL = "https://api.bricks.example.com"
    config.BRICKS_API_KEY = "test-api-key-12345"
    config.BRICKS_BEARER_TOKEN = "test-bearer-token"
    config.BRICKS_PUBLISH_DRY_RUN = True
    return config


@pytest.fixture
def adapter(bricks_config: MagicMock) -> BricksApiAdapter:
    return BricksApiAdapter(config=bricks_config)


def _mock_urlopen_response(body: bytes, headers: dict | None = None, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.headers = headers or {}
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestListProjectDocuments:

    async def test_calls_api_with_bearer_token(self, adapter: BricksApiAdapter) -> None:
        body = json.dumps({"items": []}).encode()
        mock_resp = _mock_urlopen_response(body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            await adapter.list_project_documents(project_id="proj-123")

            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert "api/projects/proj-123/documents/ai" in req.full_url
            assert req.get_header("Authorization") == "Bearer test-bearer-token"

    async def test_returns_list_of_bricks_document_info(self, adapter: BricksApiAdapter) -> None:
        api_response = {
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
        body = json.dumps(api_response).encode()
        mock_resp = _mock_urlopen_response(body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp):
            result = await adapter.list_project_documents(project_id="proj-123")

        assert len(result) == 2
        assert isinstance(result[0], BricksDocumentInfo)
        assert result[0].file_name == "report.pdf"
        assert result[0].mime_type == "application/pdf"
        assert result[0].size == 1024
        assert result[1].file_name == "notes.docx"
        assert result[1].is_ignored is True

    async def test_camel_case_to_snake_case_mapping(self, adapter: BricksApiAdapter) -> None:
        api_response = {
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
        body = json.dumps(api_response).encode()
        mock_resp = _mock_urlopen_response(body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp):
            result = await adapter.list_project_documents(project_id="proj-123")

        doc = result[0]
        assert doc.image_analysis_status == "relevant"
        assert doc.image_analysis_confidence == 95.0
        assert doc.image_analysis_reasoning == "Contains relevant diagrams"
        assert doc.image_analysis_date == "2025-12-04T10:00:00.000Z"
        assert doc.category_confidence == 72.3
        assert doc.category_source == "llm"
        assert doc.category_classified_at == "2025-12-05T08:00:00.000Z"

    async def test_returns_empty_list_when_no_documents(self, adapter: BricksApiAdapter) -> None:
        body = json.dumps({"items": []}).encode()
        mock_resp = _mock_urlopen_response(body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp):
            result = await adapter.list_project_documents(project_id="proj-empty")

        assert result == []

    async def test_raises_on_401_unauthorized(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.bricks.example.com/api/projects/proj-123/documents/ai",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None,
            )

            with pytest.raises(PermissionError):
                await adapter.list_project_documents(project_id="proj-123")

    async def test_raises_on_403_forbidden(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.bricks.example.com/api/projects/proj-123/documents/ai",
                code=403,
                msg="Forbidden",
                hdrs={},
                fp=None,
            )

            with pytest.raises(PermissionError):
                await adapter.list_project_documents(project_id="proj-123")

    async def test_raises_on_404_not_found(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.bricks.example.com/api/projects/nonexistent/documents/ai",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None,
            )

            with pytest.raises(FileNotFoundError):
                await adapter.list_project_documents(project_id="nonexistent")

    async def test_raises_on_connection_error(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [urllib.error.URLError(reason="Connection refused")] * _MAX_RETRIES

            with pytest.raises(ConnectionError):
                await adapter.list_project_documents(project_id="proj-123")

    async def test_raises_on_timeout(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [TimeoutError("Request timed out")] * _MAX_RETRIES

            with pytest.raises(TimeoutError):
                await adapter.list_project_documents(project_id="proj-123")

    async def test_does_not_retry_non_retryable_exception(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = ValueError("bad value")

            with pytest.raises(ValueError):
                await adapter.list_project_documents(project_id="proj-123")
            assert mock_urlopen.call_count == 1


class TestDownloadDocument:

    async def test_resolves_document_id_and_downloads_presigned_url(self, adapter: BricksApiAdapter) -> None:
        presigned_url = "https://s3.example.com/projects/proj-1/doc.pdf?X-Amz-Signature=abc123"
        doc_list_body = json.dumps({
            "items": [{
                "id": "doc-1",
                "fileName": "doc.pdf",
                "url": presigned_url,
                "mimeType": "application/pdf",
                "size": 100,
                "status": "PROCESSED",
            }]
        }).encode()
        list_resp = _mock_urlopen_response(doc_list_body)

        file_body = b"PDF content"
        file_resp = _mock_urlopen_response(file_body, headers={"Content-Disposition": 'attachment; filename="doc.pdf"'})

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", side_effect=[list_resp, file_resp]) as mock_urlopen:
            content, filename, mime_type = await adapter.download_document(document_id="doc-1", project_id="proj-1")

            assert mock_urlopen.call_count == 2
            list_req = mock_urlopen.call_args_list[0][0][0]
            assert "api/projects/proj-1/documents/ai" in list_req.full_url
        assert content == b"PDF content"
        assert filename == "doc.pdf"
        assert mime_type == "application/pdf"

    async def test_extracts_filename_from_url_when_no_content_disposition(self, adapter: BricksApiAdapter) -> None:
        presigned_url = "https://s3.example.com/projects/proj-1/Dossier_Financement.pdf?X-Amz-Signature=abc"
        doc_list_body = json.dumps({
            "items": [{
                "id": "doc-2",
                "fileName": "Dossier_Financement.pdf",
                "url": presigned_url,
                "mimeType": "application/pdf",
                "size": 200,
                "status": "PROCESSED",
            }]
        }).encode()
        list_resp = _mock_urlopen_response(doc_list_body)
        file_resp = _mock_urlopen_response(b"data", headers={})

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", side_effect=[list_resp, file_resp]):
            _, filename, _ = await adapter.download_document(document_id="doc-2", project_id="proj-1")

        assert filename == "Dossier_Financement.pdf"

    async def test_defaults_to_document_bin_when_no_filename(self, adapter: BricksApiAdapter) -> None:
        doc_list_body = json.dumps({
            "items": [{
                "id": "doc-3",
                "fileName": "noext",
                "url": "https://s3.example.com/projects/proj-1/noext?X-Amz-Signature=abc",
                "mimeType": "application/octet-stream",
                "size": 50,
                "status": "PROCESSED",
            }]
        }).encode()
        list_resp = _mock_urlopen_response(doc_list_body)
        file_resp = _mock_urlopen_response(b"data", headers={})

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", side_effect=[list_resp, file_resp]):
            _, filename, _ = await adapter.download_document(document_id="doc-3", project_id="proj-1")

        assert filename == "document.bin"

    async def test_raises_when_document_id_not_found(self, adapter: BricksApiAdapter) -> None:
        doc_list_body = json.dumps({"items": []}).encode()
        list_resp = _mock_urlopen_response(doc_list_body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=list_resp):
            with pytest.raises(FileNotFoundError, match="Document missing-id not found"):
                await adapter.download_document(document_id="missing-id", project_id="proj-1")

    async def test_raises_permission_error_on_403(self, adapter: BricksApiAdapter) -> None:
        presigned_url = "https://s3.example.com/doc.pdf?X-Amz-Signature=abc"
        doc_list_body = json.dumps({"items": [{"id": "doc-1", "fileName": "doc.pdf", "url": presigned_url, "mimeType": "application/pdf", "size": 100, "status": "PROCESSED"}]}).encode()
        list_resp = _mock_urlopen_response(doc_list_body)
        http_err_403 = urllib.error.HTTPError(
            url="https://s3.example.com/doc.pdf",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [list_resp, http_err_403]

            with pytest.raises(PermissionError):
                await adapter.download_document(document_id="doc-1", project_id="proj-1")

    async def test_raises_connection_error(self, adapter: BricksApiAdapter) -> None:
        presigned_url = "https://s3.example.com/doc.pdf?X-Amz-Signature=abc"
        doc_list_body = json.dumps({"items": [{"id": "doc-1", "fileName": "doc.pdf", "url": presigned_url, "mimeType": "application/pdf", "size": 100, "status": "PROCESSED"}]}).encode()
        list_resp = _mock_urlopen_response(doc_list_body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [list_resp] + [urllib.error.URLError(reason="Connection refused")] * _MAX_RETRIES

            with pytest.raises(ConnectionError):
                await adapter.download_document(document_id="doc-1", project_id="proj-1")

    async def test_raises_timeout(self, adapter: BricksApiAdapter) -> None:
        presigned_url = "https://s3.example.com/doc.pdf?X-Amz-Signature=abc"
        doc_list_body = json.dumps({"items": [{"id": "doc-1", "fileName": "doc.pdf", "url": presigned_url, "mimeType": "application/pdf", "size": 100, "status": "PROCESSED"}]}).encode()
        list_resp = _mock_urlopen_response(doc_list_body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [list_resp] + [TimeoutError("Download timed out")] * _MAX_RETRIES

            with pytest.raises(TimeoutError):
                await adapter.download_document(document_id="doc-1", project_id="proj-1")


class TestExtractFilename:

    def test_extracts_from_content_disposition_quoted(self) -> None:
        assert _extract_filename('attachment; filename="report.pdf"') == "report.pdf"

    def test_extracts_from_content_disposition_unquoted(self) -> None:
        assert _extract_filename("attachment; filename=report.pdf") == "report.pdf"

    def test_extracts_from_url_path(self) -> None:
        assert _extract_filename("", "https://s3.example.com/path/to/report.pdf?X-Amz-Sig=abc") == "report.pdf"

    def test_extracts_decoded_filename_from_url(self) -> None:
        assert _extract_filename("", "https://s3.example.com/Dossier_de_Financement.pdf?X-Amz-Sig=abc") == "Dossier_de_Financement.pdf"

    def test_defaults_to_document_bin(self) -> None:
        assert _extract_filename("", "") == "document.bin"

    def test_url_without_extension_falls_back(self) -> None:
        assert _extract_filename("", "https://s3.example.com/path/noext") == "document.bin"


class TestNormalizeExtension:

    def test_normal_dot_pdf(self) -> None:
        assert _normalize_extension("report.pdf") == "report.pdf"

    def test_dot_underscore_pdf(self) -> None:
        assert _normalize_extension("report._pdf") == "report.pdf"

    def test_double_dot_extension(self) -> None:
        assert _normalize_extension("report..pdf") == "report..pdf"

    def test_no_extension(self) -> None:
        assert _normalize_extension("noext") == "noext"

    def test_empty_extension(self) -> None:
        assert _normalize_extension("report.") == "report."

    def test_normal_dot_docx(self) -> None:
        assert _normalize_extension("file.docx") == "file.docx"


class TestPublishSectionVersion:

    async def test_calls_post_with_x_api_key_header(self, adapter: BricksApiAdapter) -> None:
        api_response = {"id": "sv-1", "sectionKey": "intro"}
        body = json.dumps(api_response).encode()
        mock_resp = _mock_urlopen_response(body)

        payload = {
            "project_unique_id": "proj-123",
            "section_key": "intro",
            "content": "Hello world",
            "workflow_id": "wf-1",
            "workflow_name": "draft",
            "workflow_metadata": {},
        }

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            await adapter.publish_section_version(payload=payload)

            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.get_header("X-api-key") == "test-api-key-12345" or req.get_header("X-API-Key") == "test-api-key-12345"
            assert req.method == "POST"
            assert "api/section-versions" in req.full_url

    async def test_returns_section_version_result(self, adapter: BricksApiAdapter) -> None:
        api_response = {"id": "sv-uuid", "sectionKey": "summary"}
        body = json.dumps(api_response).encode()
        mock_resp = _mock_urlopen_response(body)

        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen", return_value=mock_resp):
            result = await adapter.publish_section_version(
                payload={"section_key": "summary", "content": "Summary text"}
            )

        assert isinstance(result, SectionVersionResult)
        assert result.success is True

    async def test_raises_on_401_unauthorized(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.bricks.example.com/api/section-versions",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None,
            )

            with pytest.raises(PermissionError):
                await adapter.publish_section_version(payload={"section_key": "intro"})

    async def test_raises_on_connection_error(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [urllib.error.URLError(reason="Connection refused")] * _MAX_RETRIES

            with pytest.raises(ConnectionError):
                await adapter.publish_section_version(payload={"section_key": "intro"})

    async def test_raises_on_timeout(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen, \
             patch("infrastructure.bricks.bricks_api_adapter.time.sleep"), \
             patch("infrastructure.bricks.bricks_api_adapter.random.uniform", return_value=0.5):
            mock_urlopen.side_effect = [TimeoutError("Publish timed out")] * _MAX_RETRIES

            with pytest.raises(TimeoutError):
                await adapter.publish_section_version(payload={"section_key": "intro"})

    async def test_logs_warning_on_429_rate_limit(self, adapter: BricksApiAdapter) -> None:
        with patch("infrastructure.bricks.bricks_api_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.bricks.example.com/api/section-versions",
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=None,
            )

            with pytest.raises(RuntimeError, match="HTTP 429"):
                await adapter.publish_section_version(payload={"section_key": "intro"})