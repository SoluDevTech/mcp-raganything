"""TDD tests for the upload file route — the implementation does not exist yet."""

from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from application.use_cases.upload_file_use_case import UploadFileUseCase
from dependencies import get_upload_file_use_case
from domain.ports.storage_port import FileInfo
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


class TestUploadFileRoute:
    """Tests for POST /api/v1/files/upload.

    The route will:
    - Accept multipart form with `file` (UploadFile) and `prefix` (optional, default "")
    - Validate prefix (reject path traversal `..` and absolute paths)
    - Call UploadFileUseCase
    - Return HTTP 201 with {object_name, size, message}
    """

    @pytest.fixture
    def mock_upload_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=UploadFileUseCase)
        mock.execute.return_value = FileInfo(
            object_name="documents/report.pdf",
            size=2048,
            last_modified=None,
        )
        return mock

    async def test_successful_upload_returns_201(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should return 201 with object_name and size on successful upload."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                files={"file": ("report.pdf", b"fake pdf content", "application/pdf")},
                data={"prefix": "documents/"},
            )

        # Assert
        assert response.status_code == 201
        body = response.json()
        assert body["object_name"] == "documents/report.pdf"
        assert body["size"] == 2048

    async def test_successful_upload_includes_message(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should include a success message in the response."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                files={"file": ("report.pdf", b"data", "application/pdf")},
                data={"prefix": "documents/"},
            )

        # Assert
        body = response.json()
        assert "message" in body

    async def test_upload_without_prefix_uses_empty_string(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should default to empty prefix when prefix is not provided."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                files={"file": ("photo.jpg", b"image data", "image/jpeg")},
            )

        # Assert
        assert response.status_code == 201
        mock_upload_use_case.execute.assert_called_once_with(
            # Arrange
            file_data=b"image data",
            file_name="photo.jpg",
            prefix="",
            content_type="image/jpeg",
        )

    async def test_upload_with_path_traversal_in_prefix_returns_422(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should reject prefix containing '..' path traversal."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                files={"file": ("report.pdf", b"data", "application/pdf")},
                data={"prefix": "../../etc/"},
            )

        # Assert
        assert response.status_code == 422

    async def test_upload_with_absolute_path_in_prefix_returns_422(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should reject prefix that is an absolute path."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                files={"file": ("report.pdf", b"data", "application/pdf")},
                data={"prefix": "/etc/secrets/"},
            )

        # Assert
        assert response.status_code == 422

    async def test_upload_missing_file_returns_422(
        self, mock_upload_use_case: AsyncMock
    ) -> None:
        """Should return 422 when no file is provided in the request."""
        # Arrange
        app.dependency_overrides[get_upload_file_use_case] = lambda: (
            mock_upload_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.post(
                # Arrange
                "/api/v1/files/upload",
                data={"prefix": "documents/"},
            )

        # Assert
        assert response.status_code == 422
