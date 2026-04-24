"""Tests for classical query routes — TDD Red phase.

These tests will FAIL until the production code is implemented.
Tests cover POST /classical/query which runs synchronously.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from application.responses.classical_query_response import ClassicalQueryResponse
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestClassicalQueryRoute:
    """Tests for POST /classical/query."""

    @pytest.fixture
    def mock_classical_query_use_case(self) -> AsyncMock:
        mock = AsyncMock()
        mock.execute.return_value = ClassicalQueryResponse(
            status="success",
            message="",
            queries=["What is ML?"],
            chunks=[],
        )
        return mock

    async def test_query_returns_200(
        self,
        mock_classical_query_use_case: AsyncMock,
    ) -> None:
        """POST with working_dir and query should return 200."""
        from dependencies import get_classical_query_use_case

        app.dependency_overrides[get_classical_query_use_case] = lambda: (
            mock_classical_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/query",
                json={
                    "working_dir": "/tmp/rag/project_1",
                    "query": "What is machine learning?",
                },
            )

        assert response.status_code == 200

    async def test_query_calls_use_case_with_correct_params(
        self,
        mock_classical_query_use_case: AsyncMock,
    ) -> None:
        """Should forward working_dir, query, and params to the use case."""
        from dependencies import get_classical_query_use_case

        app.dependency_overrides[get_classical_query_use_case] = lambda: (
            mock_classical_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/classical/query",
                json={
                    "working_dir": "/tmp/rag/project_42",
                    "query": "What are the findings?",
                    "top_k": 20,
                    "num_variations": 5,
                    "relevance_threshold": 7.0,
                },
            )

        mock_classical_query_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/project_42",
            query="What are the findings?",
            top_k=20,
            num_variations=5,
            relevance_threshold=7.0,
            vector_distance_threshold=None,
            enable_llm_judge=True,
        )

    async def test_query_uses_default_params(
        self,
        mock_classical_query_use_case: AsyncMock,
    ) -> None:
        """Should use defaults for top_k, num_variations, relevance_threshold."""
        from dependencies import get_classical_query_use_case

        app.dependency_overrides[get_classical_query_use_case] = lambda: (
            mock_classical_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/classical/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "test query",
                },
            )

        mock_classical_query_use_case.execute.assert_called_once_with(
            working_dir="/tmp/rag/test",
            query="test query",
            top_k=10,
            num_variations=3,
            relevance_threshold=5.0,
            vector_distance_threshold=None,
            enable_llm_judge=True,
        )

    async def test_query_returns_response_body(
        self,
        mock_classical_query_use_case: AsyncMock,
    ) -> None:
        """Should return the chunks list from the ClassicalQueryResponse."""
        from dependencies import get_classical_query_use_case

        app.dependency_overrides[get_classical_query_use_case] = lambda: (
            mock_classical_query_use_case
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/query",
                json={
                    "working_dir": "/tmp/rag/test",
                    "query": "What is ML?",
                },
            )

        body = response.json()
        assert isinstance(body, list)
        assert "chunks" not in body

    async def test_query_rejects_missing_query(self) -> None:
        """Missing query field should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/query",
                json={"working_dir": "/tmp/rag/test"},
            )

        assert response.status_code == 422

    async def test_query_rejects_missing_working_dir(self) -> None:
        """Missing working_dir field should return 422."""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/classical/query",
                json={"query": "some question"},
            )

        assert response.status_code == 422
