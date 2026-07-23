from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from application.use_cases.create_folder_use_case import CreateFolderUseCase
from application.use_cases.delete_file_use_case import DeleteFileUseCase
from application.use_cases.delete_folder_use_case import DeleteFolderUseCase
from dependencies import (
    get_create_folder_use_case,
    get_delete_file_use_case,
    get_delete_folder_use_case,
)
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


class TestCreateFolderRoute:
    @pytest.fixture
    def mock_create_folder_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=CreateFolderUseCase)
        mock.execute.return_value = "docs/"
        return mock

    async def test_creates_folder_returns_201(
        self, mock_create_folder_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_create_folder_use_case] = lambda: (
            mock_create_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/folders",
                json={"prefix": "docs"},
            )

        assert response.status_code == 201
        body = response.json()
        assert body == {"message": "Folder created", "prefix": "docs/"}

    async def test_calls_use_case_with_prefix(
        self, mock_create_folder_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_create_folder_use_case] = lambda: (
            mock_create_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/api/v1/files/folders", json={"prefix": "docs"})

        mock_create_folder_use_case.execute.assert_called_once_with(prefix="docs")

    async def test_rejects_empty_prefix(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/folders",
                json={"prefix": ""},
            )

        assert response.status_code == 422

    async def test_rejects_absolute_prefix(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/files/folders",
                json={"prefix": "/docs"},
            )

        assert response.status_code == 422


class TestDeleteFileRoute:
    @pytest.fixture
    def mock_delete_file_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=DeleteFileUseCase)
        mock.execute.return_value = "docs/report.pdf"
        return mock

    async def test_deletes_file_returns_200(
        self, mock_delete_file_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_delete_file_use_case] = lambda: (
            mock_delete_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/v1/files",
                params={
                    "object_name": "docs/report.pdf",
                    "working_dir": "docs/",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body == {"message": "File deleted", "object_name": "docs/report.pdf"}

    async def test_calls_use_case_with_object_name_and_working_dir(
        self, mock_delete_file_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_delete_file_use_case] = lambda: (
            mock_delete_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete(
                "/api/v1/files",
                params={
                    "object_name": "docs/report.pdf",
                    "working_dir": "docs/",
                },
            )

        mock_delete_file_use_case.execute.assert_called_once_with(
            object_path="docs/report.pdf",
            working_dir="docs/",
        )

    async def test_rejects_missing_object_name(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/v1/files",
                params={"working_dir": "docs/"},
            )

        assert response.status_code == 422

    async def test_delete_file_without_working_dir_returns_200(
        self, mock_delete_file_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_delete_file_use_case] = lambda: (
            mock_delete_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/v1/files",
                params={"object_name": "report.pdf"},
            )

        assert response.status_code == 200
        mock_delete_file_use_case.execute.assert_called_once_with(
            object_path="report.pdf",
            working_dir="default-composable",
        )


class TestDeleteFolderRoute:
    @pytest.fixture
    def mock_delete_folder_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=DeleteFolderUseCase)
        mock.execute.return_value = "docs/"
        return mock

    async def test_deletes_folder_returns_200(
        self, mock_delete_folder_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_delete_folder_use_case] = lambda: (
            mock_delete_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/v1/files/folders",
                params={"prefix": "docs"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body == {"message": "Folder deleted", "prefix": "docs/"}

    async def test_calls_use_case_with_prefix(
        self, mock_delete_folder_use_case: AsyncMock
    ) -> None:
        app.dependency_overrides[get_delete_folder_use_case] = lambda: (
            mock_delete_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete(
                "/api/v1/files/folders",
                params={"prefix": "docs"},
            )

        mock_delete_folder_use_case.execute.assert_called_once_with(prefix="docs")

    async def test_rejects_missing_prefix(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.delete("/api/v1/files/folders")

        assert response.status_code == 422
