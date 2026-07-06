import httpx
import pytest
from httpx import ASGITransport

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
            # Act
            response = await client.get("/api/v1/health")

        # Assert
        assert response.status_code == 200

    async def test_health_returns_status_message(self) -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Act
            response = await client.get("/api/v1/health")

        # Assert
        body = response.json()
        assert body["message"] == "RAG Anything API is running"
