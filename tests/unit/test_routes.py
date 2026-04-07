from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from application.requests.query_request import MultimodalContentItem
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.multimodal_query_use_case import MultimodalQueryUseCase
from application.use_cases.query_use_case import QueryUseCase
from dependencies import (
    get_index_file_use_case,
    get_index_folder_use_case,
    get_multimodal_query_use_case,
    get_query_use_case,
)
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestHealthRoute:
    async def test_health_returns_200(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")

        assert response.status_code == 200

    async def test_health_returns_status_message(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")

        body = response.json()
        assert body["message"] == "RAG Anything API is running"


class TestIndexFileRoute:
    @pytest.fixture
    def mock_index_file_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=IndexFileUseCase)
        mock.execute.return_value = None
        return mock

    async def test_index_file_returns_202(
        self,
        mock_index_file_use_case: AsyncMock,
    ) -> None:
        """POST JSON with file_name and working_dir should return 202."""
        app.dependency_overrides[get_index_file_use_case] = lambda: (
            mock_index_file_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/file/index",
                json={
                    "file_name": "doc.pdf",
                    "working_dir": "/tmp/rag/test",
                },
            )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"
        assert "background" in body["message"].lower()

    async def test_index_file_rejects_missing_file_name(self) -> None:
        """Missing file_name in JSON body should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/file/index",
                json={"working_dir": "/tmp/rag/test"},
            )

        assert response.status_code == 422

    async def test_index_file_rejects_missing_working_dir(self) -> None:
        """Missing working_dir in JSON body should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/file/index",
                json={"file_name": "doc.pdf"},
            )

        assert response.status_code == 422


class TestIndexFolderRoute:
    @pytest.fixture
    def mock_index_folder_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=IndexFolderUseCase)
        mock.execute.return_value = None
        return mock

    async def test_index_folder_returns_202(
        self,
        mock_index_folder_use_case: AsyncMock,
    ) -> None:
        """POST JSON with working_dir should return 202."""
        app.dependency_overrides[get_index_folder_use_case] = lambda: (
            mock_index_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/folder/index",
                json={"working_dir": "/tmp/rag/test"},
            )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"
        assert "background" in body["message"].lower()

    async def test_index_folder_accepts_optional_fields(
        self,
        mock_index_folder_use_case: AsyncMock,
    ) -> None:
        """Optional recursive and file_extensions should be accepted."""
        app.dependency_overrides[get_index_folder_use_case] = lambda: (
            mock_index_folder_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/folder/index",
                json={
                    "working_dir": "/tmp/rag/project_1",
                    "recursive": False,
                    "file_extensions": [".pdf", ".docx"],
                },
            )

        assert response.status_code == 202

    async def test_index_folder_rejects_missing_working_dir(self) -> None:
        """Missing working_dir in JSON body should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/folder/index",
                json={},
            )

        assert response.status_code == 422


class TestQueryRoute:
    @pytest.fixture
    def mock_query_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=QueryUseCase)
        mock.execute.return_value = {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
        }
        return mock

    async def test_query_returns_200(
        self,
        mock_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_query_use_case] = lambda: mock_query_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "What is the summary?",
                },
            )

        assert response.status_code == 200

    async def test_query_calls_use_case_with_correct_params(
        self,
        mock_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_query_use_case] = lambda: mock_query_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/query",
                json={
                    "working_dir": "/tmp/rag/project_42",
                    "query": "What are the findings?",
                    "mode": "hybrid",
                    "top_k": 20,
                },
            )

        mock_query_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_42",
            query="What are the findings?",
            mode="hybrid",
            top_k=20,
        )

    async def test_query_returns_response_body(
        self,
        mock_query_use_case: AsyncMock,
    ) -> None:
        """Route now returns list[ChunkResponse] (data.chunks), not the full dict."""
        app.dependency_overrides[get_query_use_case] = lambda: mock_query_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "summarize",
                },
            )

        body = response.json()
        assert isinstance(body, list)
        assert body == []

    async def test_query_uses_default_mode_and_top_k(
        self,
        mock_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_query_use_case] = lambda: mock_query_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "test query",
                },
            )

        mock_query_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="test query",
            mode="naive",
            top_k=10,
        )

    async def test_query_rejects_missing_query_field(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={"working_dir": "/tmp/rag/test"},
            )

        assert response.status_code == 422

    async def test_query_rejects_missing_working_dir(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "some question"},
            )

        assert response.status_code == 422

    async def test_query_rejects_invalid_mode(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "test",
                    "mode": "invalid_mode",
                },
            )

        assert response.status_code == 422


class TestMultimodalQueryRoute:
    @pytest.fixture
    def mock_multimodal_query_use_case(self) -> AsyncMock:
        mock = AsyncMock(spec=MultimodalQueryUseCase)
        mock.execute.return_value = {
            "status": "success",
            "data": "Multimodal analysis result",
        }
        return mock

    async def test_multimodal_query_returns_200(
        self,
        mock_multimodal_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_multimodal_query_use_case] = lambda: (
            mock_multimodal_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "What does this image show?",
                    "multimodal_content": [
                        {"type": "image", "img_path": "/tmp/img.png"},
                    ],
                },
            )

        assert response.status_code == 200

    async def test_multimodal_query_calls_use_case_with_correct_params(
        self,
        mock_multimodal_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_multimodal_query_use_case] = lambda: (
            mock_multimodal_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/project_42",
                    "query": "Analyze this table",
                    "multimodal_content": [
                        {
                            "type": "table",
                            "table_data": "A,B\n1,2",
                            "table_caption": "Test",
                        },
                    ],
                    "mode": "global",
                    "top_k": 20,
                },
            )

        mock_multimodal_query_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_42",
            query="Analyze this table",
            multimodal_content=[
                MultimodalContentItem(
                    type="table", table_data="A,B\n1,2", table_caption="Test"
                ),
            ],
            mode="global",
            top_k=20,
        )

    async def test_multimodal_query_returns_response_body(
        self,
        mock_multimodal_query_use_case: AsyncMock,
    ) -> None:
        app.dependency_overrides[get_multimodal_query_use_case] = lambda: (
            mock_multimodal_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "Describe this",
                    "multimodal_content": [
                        {"type": "image", "img_path": "/tmp/img.png"},
                    ],
                },
            )

        body = response.json()
        assert body["status"] == "success"
        assert body["data"] == "Multimodal analysis result"

    async def test_multimodal_query_rejects_missing_query(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/test",
                    "multimodal_content": [
                        {"type": "image", "img_path": "/tmp/img.png"},
                    ],
                },
            )

        assert response.status_code == 422

    async def test_multimodal_query_rejects_missing_multimodal_content(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "test",
                },
            )

        assert response.status_code == 422

    async def test_multimodal_query_rejects_invalid_content_type(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/query/multimodal",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "test",
                    "multimodal_content": [
                        {"type": "video", "path": "/tmp/video.mp4"},
                    ],
                },
            )

        assert response.status_code == 422
