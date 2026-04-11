from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from application.use_cases.liveness_check_use_case import LivenessCheckUseCase
from dependencies import get_liveness_check_use_case
from main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


class TestLivenessRoute:
    async def test_returns_200_when_all_healthy(self) -> None:
        mock_use_case = AsyncMock(spec=LivenessCheckUseCase)
        mock_use_case.execute.return_value = {
            "status": "healthy",
            "checks": {"postgres": "ok", "minio": "ok"},
        }
        app.dependency_overrides[get_liveness_check_use_case] = lambda: mock_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health/live")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["checks"]["postgres"] == "ok"
        assert body["checks"]["minio"] == "ok"

    async def test_returns_503_when_postgres_down(self) -> None:
        mock_use_case = AsyncMock(spec=LivenessCheckUseCase)
        mock_use_case.execute.return_value = {
            "status": "degraded",
            "checks": {"postgres": "unreachable", "minio": "ok"},
        }
        app.dependency_overrides[get_liveness_check_use_case] = lambda: mock_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health/live")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["checks"]["postgres"] == "unreachable"

    async def test_returns_503_when_minio_down(self) -> None:
        mock_use_case = AsyncMock(spec=LivenessCheckUseCase)
        mock_use_case.execute.return_value = {
            "status": "degraded",
            "checks": {"postgres": "ok", "minio": "unreachable"},
        }
        app.dependency_overrides[get_liveness_check_use_case] = lambda: mock_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health/live")

        assert response.status_code == 503
        body = response.json()
        assert body["checks"]["minio"] == "unreachable"

    async def test_returns_503_when_both_down(self) -> None:
        mock_use_case = AsyncMock(spec=LivenessCheckUseCase)
        mock_use_case.execute.return_value = {
            "status": "degraded",
            "checks": {"postgres": "unreachable", "minio": "unreachable"},
        }
        app.dependency_overrides[get_liveness_check_use_case] = lambda: mock_use_case

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health/live")

        assert response.status_code == 503
        body = response.json()
        assert body["checks"]["postgres"] == "unreachable"
        assert body["checks"]["minio"] == "unreachable"


class TestLivenessCheckUseCase:
    async def test_returns_healthy_when_both_ok(self) -> None:
        mock_storage = AsyncMock()
        mock_storage.ping.return_value = True
        mock_pg = AsyncMock()
        mock_pg.ping.return_value = True

        use_case = LivenessCheckUseCase(
            storage=mock_storage,
            postgres_health=mock_pg,
            bucket="test-bucket",
        )
        result = await use_case.execute()

        assert result == {
            "status": "healthy",
            "checks": {"postgres": "ok", "minio": "ok"},
        }
        mock_storage.ping.assert_awaited_once_with("test-bucket")
        mock_pg.ping.assert_awaited_once()

    async def test_returns_degraded_when_postgres_down(self) -> None:
        mock_storage = AsyncMock()
        mock_storage.ping.return_value = True
        mock_pg = AsyncMock()
        mock_pg.ping.return_value = False

        use_case = LivenessCheckUseCase(
            storage=mock_storage,
            postgres_health=mock_pg,
            bucket="test-bucket",
        )
        result = await use_case.execute()

        assert result["status"] == "degraded"
        assert result["checks"]["postgres"] == "unreachable"

    async def test_returns_degraded_when_minio_down(self) -> None:
        mock_storage = AsyncMock()
        mock_storage.ping.return_value = False
        mock_pg = AsyncMock()
        mock_pg.ping.return_value = True

        use_case = LivenessCheckUseCase(
            storage=mock_storage,
            postgres_health=mock_pg,
            bucket="test-bucket",
        )
        result = await use_case.execute()

        assert result["status"] == "degraded"
        assert result["checks"]["minio"] == "unreachable"


class TestAsyncpgHealthAdapter:
    @patch("infrastructure.database.asyncpg_health_adapter.asyncpg")
    async def test_returns_true_on_success(self, mock_asyncpg: MagicMock) -> None:
        from config import DatabaseConfig
        from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        adapter = AsyncpgHealthAdapter(DatabaseConfig())
        result = await adapter.ping()

        assert result is True
        mock_conn.close.assert_awaited_once()

    @patch("infrastructure.database.asyncpg_health_adapter.asyncpg")
    async def test_returns_false_on_connection_error(
        self, mock_asyncpg: MagicMock
    ) -> None:
        from config import DatabaseConfig
        from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter

        mock_asyncpg.connect = AsyncMock(side_effect=ConnectionRefusedError)

        adapter = AsyncpgHealthAdapter(DatabaseConfig())
        result = await adapter.ping()

        assert result is False


class TestMinioAdapterPing:
    @patch("infrastructure.storage.minio_adapter.Minio")
    async def test_returns_true_when_bucket_exists(
        self, mock_minio_cls: MagicMock
    ) -> None:
        from infrastructure.storage.minio_adapter import MinioAdapter

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_cls.return_value = mock_client

        adapter = MinioAdapter("localhost:9000", "access", "secret")
        result = await adapter.ping("test-bucket")

        assert result is True
        mock_client.bucket_exists.assert_called_once_with("test-bucket")

    @patch("infrastructure.storage.minio_adapter.Minio")
    async def test_returns_false_on_exception(self, mock_minio_cls: MagicMock) -> None:
        from infrastructure.storage.minio_adapter import MinioAdapter

        mock_client = MagicMock()
        mock_client.bucket_exists.side_effect = Exception("connection refused")
        mock_minio_cls.return_value = mock_client

        adapter = MinioAdapter("localhost:9000", "access", "secret")
        result = await adapter.ping("test-bucket")

        assert result is False
