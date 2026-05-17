"""Tests for BricksDocumentInfo, SectionVersionResult models and BricksApiPort ABC."""

from abc import ABC
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from domain.ports.bricks_api_port import (
    BricksApiPort,
    BricksDocumentInfo,
    SectionVersionResult,
)


class TestBricksDocumentInfo:
    """Tests for the BricksDocumentInfo pydantic model."""

    def test_creates_with_camel_case_aliases(self) -> None:
        """Should accept camelCase field names via populate_by_name=True."""
        doc = BricksDocumentInfo(
            id="abc-123",
            fileName="document.pdf",
            url="https://s3.example.com/doc.pdf",
            mimeType="application/pdf",
            size=869723,
            status="PROCESSED",
            uploadedAt="2025-12-03T17:02:50.916Z",
            imageAnalysisStatus="relevant",
            imageAnalysisConfidence=95,
            imageAnalysisReasoning="Contains diagrams",
            imageAnalysisDate="2025-12-04T10:00:00.000Z",
            isIgnored=False,
            projectId="proj-456",
        )

        assert doc.id == "abc-123"
        assert doc.file_name == "document.pdf"
        assert doc.url == "https://s3.example.com/doc.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.size == 869723
        assert doc.status == "PROCESSED"
        assert doc.uploaded_at == "2025-12-03T17:02:50.916Z"
        assert doc.image_analysis_status == "relevant"
        assert doc.image_analysis_confidence == 95
        assert doc.image_analysis_reasoning == "Contains diagrams"
        assert doc.image_analysis_date == "2025-12-04T10:00:00.000Z"
        assert doc.is_ignored is False
        assert doc.project_id == "proj-456"

    def test_creates_with_snake_case_field_names(self) -> None:
        """Should also accept snake_case field names directly."""
        doc = BricksDocumentInfo(
            id="abc-123",
            file_name="document.pdf",
            url="https://s3.example.com/doc.pdf",
        )

        assert doc.file_name == "document.pdf"

    def test_defaults_for_optional_fields(self) -> None:
        """Should provide defaults for fields with defaults."""
        doc = BricksDocumentInfo(
            id="abc-123",
            fileName="test.pdf",
            url="https://s3.example.com/test.pdf",
        )

        assert doc.hash == ""
        assert doc.mime_type == ""
        assert doc.size == 0
        assert doc.status == ""
        assert doc.uploaded_at is None
        assert doc.image_analysis_status is None
        assert doc.image_analysis_confidence is None
        assert doc.image_analysis_reasoning is None
        assert doc.image_analysis_date is None
        assert doc.category is None
        assert doc.category_confidence is None
        assert doc.category_source is None
        assert doc.category_classified_at is None
        assert doc.is_ignored is False
        assert doc.project_id == ""

    def test_parses_full_api_response_item(self) -> None:
        """Should parse a full API response item with all fields."""
        api_item = {
            "id": "uuid-1",
            "fileName": "report.docx",
            "url": "https://neon-project-analysis.s3.amazonaws.com/presigned",
            "hash": "sha256hexstring",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "size": 1234567,
            "status": "PROCESSED",
            "uploadedAt": "2025-12-03T17:02:50.916Z",
            "imageAnalysisStatus": None,
            "imageAnalysisConfidence": None,
            "imageAnalysisReasoning": None,
            "imageAnalysisDate": None,
            "category": "financial",
            "categoryConfidence": 88.5,
            "categorySource": "llm",
            "categoryClassifiedAt": "2025-12-05T08:00:00.000Z",
            "isIgnored": True,
            "projectId": "proj-uuid",
        }

        doc = BricksDocumentInfo(**api_item)

        assert doc.id == "uuid-1"
        assert doc.file_name == "report.docx"
        assert doc.hash == "sha256hexstring"
        assert (
            doc.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert doc.size == 1234567
        assert doc.category == "financial"
        assert doc.category_confidence == 88.5
        assert doc.category_source == "llm"
        assert doc.category_classified_at == "2025-12-05T08:00:00.000Z"
        assert doc.is_ignored is True
        assert doc.project_id == "proj-uuid"

    def test_requires_id_file_name_and_url(self) -> None:
        """Should raise ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            BricksDocumentInfo(id="abc", fileName="test.pdf")  # missing url


class TestSectionVersionResult:
    """Tests for the SectionVersionResult model."""

    def test_creates_minimal_result(self) -> None:
        result = SectionVersionResult(success=True)

        assert result.success is True
        assert result.message == ""
        assert result.data is None
        assert result.dry_run is False
        assert result.payload_preview is None
        assert result.target_url is None

    def test_creates_with_all_fields(self) -> None:
        result = SectionVersionResult(
            success=True,
            message="Published successfully",
            data={"id": "sv-123"},
            dry_run=True,
            payload_preview={"section_key": "intro", "content": "Hello"},
            target_url="https://api.example.com/section-versions",
        )

        assert result.success is True
        assert result.message == "Published successfully"
        assert result.data == {"id": "sv-123"}
        assert result.dry_run is True
        assert result.payload_preview == {"section_key": "intro", "content": "Hello"}
        assert result.target_url == "https://api.example.com/section-versions"

    def test_dry_run_result(self) -> None:
        result = SectionVersionResult(
            success=True,
            message="Dry run — no API call made",
            dry_run=True,
            payload_preview={"key": "value"},
            target_url="https://api.example.com/section-versions",
        )

        assert result.dry_run is True
        assert result.data is None
        assert result.payload_preview is not None


class TestBricksApiPortAbstract:
    """Tests for the BricksApiPort abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """BricksApiPort is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BricksApiPort()  # type: ignore[abstract]

    def test_is_abc_subclass(self) -> None:
        """BricksApiPort should inherit from ABC."""
        assert issubclass(BricksApiPort, ABC)

    def test_concrete_subclass_must_implement_all_methods(self) -> None:
        """A subclass that doesn't implement all abstract methods cannot be instantiated."""

        class PartialAdapter(BricksApiPort):
            async def list_project_documents(
                self, project_id: str
            ) -> list[BricksDocumentInfo]:
                return []

        with pytest.raises(TypeError):
            PartialAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        """A subclass that implements all abstract methods can be instantiated."""

        class FullAdapter(BricksApiPort):
            async def list_project_documents(
                self, project_id: str
            ) -> list[BricksDocumentInfo]:
                return []

            async def download_document(self, download_url: str) -> tuple[bytes, str]:
                return (b"", "file.pdf")

            async def publish_section_version(
                self, payload: dict
            ) -> SectionVersionResult:
                return SectionVersionResult(success=True)

        adapter = FullAdapter()
        assert isinstance(adapter, BricksApiPort)

    def test_concrete_subclass_methods_are_callable(self) -> None:
        """Concrete subclass methods should be callable."""
        mock = AsyncMock(spec=BricksApiPort)
        mock.list_project_documents.return_value = [
            BricksDocumentInfo(id="1", fileName="a.pdf", url="https://s3/a.pdf")
        ]

        assert mock.list_project_documents.return_value is not None
