"""Tests for classical indexing routes — TDD Red phase.

These tests will FAIL until the production code is implemented.
Tests cover POST /classical/file/index and POST /classical/folder/index
with background task execution (202 accepted).
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestClassicalIndexFileRoute:
    """Tests for POST /classical/file/index."""

    @pytest.fixture
    def mock_classical_index_file_use_case(self) -> AsyncMock:
        mock = AsyncMock()
        mock.execute.return_value = None
        return mock

    async def test_index_file_returns_202(
        self,
        mock_classical_index_file_use_case: AsyncMock,
    ) -> None:
        """POST with file_name and working_dir should return 202 accepted."""
        # Will fail until get_classical_index_file_use_case dependency exists
        from dependencies import get_classical_index_file_use_case

        app.dependency_overrides[get_classical_index_file_use_case] = lambda: (
            mock_classical_index_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/file/index",
                json={
                    "file_name": "docs/report.pdf",
                    "working_dir": "/tmp/rag/project_1",
                },
            )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"

    async def test_index_file_rejects_missing_file_name(self) -> None:
        """Missing file_name should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/file/index",
                json={"working_dir": "/tmp/rag/test"},
            )

        assert response.status_code == 422

    async def test_index_file_rejects_missing_working_dir(self) -> None:
        """Missing working_dir should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/file/index",
                json={"file_name": "doc.pdf"},
            )

        assert response.status_code == 422

    async def test_index_file_accepts_optional_chunk_params(
        self,
        mock_classical_index_file_use_case: AsyncMock,
    ) -> None:
        """Should accept optional chunk_size and chunk_overlap."""
        from dependencies import get_classical_index_file_use_case

        app.dependency_overrides[get_classical_index_file_use_case] = lambda: (
            mock_classical_index_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/file/index",
                json={
                    "file_name": "doc.pdf",
                    "working_dir": "/tmp/rag/test",
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                },
            )

        assert response.status_code == 202


class TestClassicalIndexFolderRoute:
    """Tests for POST /classical/folder/index."""

    @pytest.fixture
    def mock_classical_index_folder_use_case(self) -> AsyncMock:
        mock = AsyncMock()
        mock.execute.return_value = None
        return mock

    async def test_index_folder_returns_202(
        self,
        mock_classical_index_folder_use_case: AsyncMock,
    ) -> None:
        """POST with working_dir should return 202 accepted."""
        from dependencies import get_classical_index_folder_use_case

        app.dependency_overrides[get_classical_index_folder_use_case] = lambda: (
            mock_classical_index_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/folder/index",
                json={"working_dir": "/tmp/rag/project_1"},
            )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"

    async def test_index_folder_accepts_optional_params(
        self,
        mock_classical_index_folder_use_case: AsyncMock,
    ) -> None:
        """Should accept optional recursive, file_extensions, chunk_size, chunk_overlap."""
        from dependencies import get_classical_index_folder_use_case

        app.dependency_overrides[get_classical_index_folder_use_case] = lambda: (
            mock_classical_index_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/folder/index",
                json={
                    "working_dir": "/tmp/rag/project_1",
                    "recursive": False,
                    "file_extensions": [".pdf", ".txt"],
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                },
            )

        assert response.status_code == 202

    async def test_index_folder_rejects_missing_working_dir(self) -> None:
        """Missing working_dir should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/folder/index",
                json={},
            )

        assert response.status_code == 422
