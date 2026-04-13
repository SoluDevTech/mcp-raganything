from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from dependencies import (
    get_list_files_use_case,
    get_list_folders_use_case,
    get_read_file_use_case,
)
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata
from domain.ports.storage_port import FileInfo
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


class TestListFilesRoute:
    @pytest.fixture
    def mock_list_files_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=ListFilesUseCase)
        mock.execute.return_value = [
            FileInfo(
                object_name="docs/report.pdf", size=1024, last_modified="2026-01-01"
            ),
            FileInfo(
                object_name="docs/notes.txt", size=512, last_modified="2026-01-02"
            ),
        ]
        return mock

    async def test_list_files_returns_200(
        self, mock_list_files_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_list_files_use_case] = lambda: (
            mock_list_files_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/list")

        assert response.status_code == 200

    async def test_list_files_returns_file_list(
        self, mock_list_files_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_list_files_use_case] = lambda: (
            mock_list_files_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/list")

        body = response.json()
        assert len(body) == 2
        assert body[0]["object_name"] == "docs/report.pdf"
        assert body[0]["size"] == 1024

    async def test_list_files_with_prefix_param(
        self, mock_list_files_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_list_files_use_case] = lambda: (
            mock_list_files_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/files/list", params={"prefix": "docs/", "recursive": "false"}
            )

        assert response.status_code == 200
        mock_list_files_use_case.execute.assert_called_once_with(
            prefix="docs/", recursive=False
        )

    async def test_list_files_empty_result(
        self, mock_list_files_use_case: AsyncMock
    ) -> None:
        mock_list_files_use_case.execute.return_value = []
        app.dependency_overrides[get_list_files_use_case] = lambda: (
            mock_list_files_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/list")

        body = response.json()
        assert body == []


class TestReadFileRoute:
    @pytest.fixture
    def mock_read_file_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=ReadFileUseCase)
        mock.execute.return_value = DocumentContent(
            content="Extracted text from PDF",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )
        return mock

    async def test_read_file_returns_200(
        self, mock_read_file_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_read_file_use_case] = lambda: (
            mock_read_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/read",
                json={"file_path": "docs/report.pdf"},
            )

        assert response.status_code == 200

    async def test_read_file_returns_content(
        self, mock_read_file_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_read_file_use_case] = lambda: (
            mock_read_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/read",
                json={"file_path": "docs/report.pdf"},
            )

        body = response.json()
        assert body["content"] == "Extracted text from PDF"
        assert body["metadata"]["mime_type"] == "application/pdf"

    async def test_read_file_returns_404_for_missing_file(
        self, mock_read_file_use_case: AsyncMock
    ) -> None:
        mock_read_file_use_case.execute.side_effect = FileNotFoundError("not found")
        app.dependency_overrides[get_read_file_use_case] = lambda: (
            mock_read_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/read",
                json={"file_path": "nonexistent.pdf"},
            )

        assert response.status_code == 404
        assert (
            "not found" in response.json()["detail"].lower()
            or "File not found" in response.json()["detail"]
        )

    async def test_read_file_returns_422_for_unsupported_format(
        self, mock_read_file_use_case: AsyncMock
    ) -> None:
        mock_read_file_use_case.execute.side_effect = ValueError(
            "Unsupported file format: video.mp4"
        )
        app.dependency_overrides[get_read_file_use_case] = lambda: (
            mock_read_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/read",
                json={"file_path": "video.mp4"},
            )

        assert response.status_code == 422

    async def test_read_file_rejects_missing_file_path(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/files/read", json={})

        assert response.status_code == 422

    async def test_read_file_rejects_path_traversal(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/read",
                json={"file_path": "../../etc/passwd"},
            )

        assert response.status_code == 422


class TestListFoldersRoute:
    @pytest.fixture
    def mock_list_folders_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=ListFoldersUseCase)
        mock.execute.return_value = ["docs/", "photos/"]
        return mock

    async def test_list_folders_returns_200(
        self, mock_list_folders_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_list_folders_use_case] = lambda: (
            mock_list_folders_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/folders")

        assert response.status_code == 200

    async def test_list_folders_returns_folder_list(
        self, mock_list_folders_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_list_folders_use_case] = lambda: (
            mock_list_folders_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/folders")

        body = response.json()
        assert body == ["docs/", "photos/"]

    async def test_list_folders_empty_result(
        self, mock_list_folders_use_case: AsyncMock
    ) -> None:
        mock_list_folders_use_case.execute.return_value = []
        app.dependency_overrides[get_list_folders_use_case] = lambda: (
            mock_list_folders_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/folders")

        body = response.json()
        assert body == []

    async def test_list_folders_with_prefix_param(
        self, mock_list_folders_use_case: AsyncMock
    ) -> None:
        mock_list_folders_use_case.execute.return_value = ["reports/", "exports/"]
        app.dependency_overrides[get_list_folders_use_case] = lambda: (
            mock_list_folders_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/files/folders", params={"prefix": "docs/"}
            )

        assert response.status_code == 200
        assert response.json() == ["reports/", "exports/"]
        mock_list_folders_use_case.execute.assert_called_once_with(prefix="docs/")

    async def test_list_folders_returns_404_for_missing_bucket(
        self, mock_list_folders_use_case: AsyncMock
    ) -> None:
        mock_list_folders_use_case.execute.side_effect = FileNotFoundError(
            "Bucket not found"
        )
        app.dependency_overrides[get_list_folders_use_case] = lambda: (
            mock_list_folders_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/files/folders")

        assert response.status_code == 404
