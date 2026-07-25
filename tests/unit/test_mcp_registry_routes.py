"""Tests for MCP server registry FastAPI routes.

Mirrors the pattern from ``test_classical_query_routes.py``: dependencies are
wired through ``app.dependency_overrides`` (FastAPI stays in control of its own
providers), the API key check is bypassed (empty master key in tests), and
requests go through ``httpx.AsyncClient`` with ``ASGITransport``.

The use cases are mocked at the use-case boundary (AsyncMock). Each use case is
replaced with an AsyncMock so the route layer is exercised against its real
FastAPI wiring while the persistence and external MCP boundaries are stubbed.

CRUD endpoints under ``/api/v1/mcp/servers`` + ``/validate`` + ``/{name}/reveal``.
Tests cover masking of secrets in responses, 404/409 errors, the openapi create
flow, and validate dry-run.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from application.api.mcp_registry_routes import mcp_registry_router
from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from dependencies import (
    get_create_mcp_server_use_case,
    get_delete_mcp_server_use_case,
    get_get_mcp_server_use_case,
    get_list_mcp_servers_use_case,
    get_update_mcp_server_use_case,
    get_validate_mcp_server_use_case,
)
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import (
    McpServerAlreadyExistsError,
    McpServerNotFoundError,
    McpServerUnreachableError,
)
from main import app

BASE_URL = "/api/v1/mcp/servers"


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    auth_token: str | None = "secret-token",
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        url=url,
        headers={"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token=auth_token,
        tool_count=tool_count,
        source_type=source_type,
        openapi_url=openapi_url,
        created_at=_now(),
        updated_at=_now(),
    )


def _masked_entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
) -> RegisteredMcpServer:
    """An entry as returned by masked endpoints (secrets stripped)."""
    return RegisteredMcpServer(
        name=name,
        url=url,
        headers={},
        env={},
        auth_token=None,
        tool_count=tool_count,
        source_type=source_type,
        openapi_url=openapi_url,
        created_at=_now(),
        updated_at=_now(),
    )


def _openapi_entry(
    name: str = "petstore",
    url: str = "http://raganything-api:8000/generated/petstore/mcp",
    openapi_url: str = "https://petstore.example.com/openapi.json",
    tool_count: int = 3,
) -> RegisteredMcpServer:
    return _entry(
        name=name,
        url=url,
        auth_token=None,
        tool_count=tool_count,
        source_type="openapi",
        openapi_url=openapi_url,
    )


def _masked_openapi_entry(
    name: str = "petstore",
    url: str = "http://raganything-api:8000/generated/petstore/mcp",
    openapi_url: str = "https://petstore.example.com/openapi.json",
    tool_count: int = 3,
) -> RegisteredMcpServer:
    return _masked_entry(
        name=name,
        url=url,
        tool_count=tool_count,
        source_type="openapi",
        openapi_url=openapi_url,
    )


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_create_use_case() -> AsyncMock:
    uc = AsyncMock(spec=CreateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry(tool_count=2))
    return uc


@pytest.fixture
def mock_validate_use_case() -> AsyncMock:
    uc = AsyncMock(spec=ValidateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=2)
    uc.execute_openapi = AsyncMock(return_value=2)
    return uc


@pytest.fixture
def mock_list_use_case() -> AsyncMock:
    uc = AsyncMock(spec=ListMcpServersUseCase)
    uc.execute = AsyncMock(
        return_value=[_masked_entry("server-a"), _masked_entry("server-b")]
    )
    return uc


@pytest.fixture
def mock_get_use_case() -> AsyncMock:
    uc = AsyncMock(spec=GetMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry())
    return uc


@pytest.fixture
def mock_update_use_case() -> AsyncMock:
    uc = AsyncMock(spec=UpdateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry(tool_count=5))
    return uc


@pytest.fixture
def mock_delete_use_case() -> AsyncMock:
    uc = AsyncMock(spec=DeleteMcpServerUseCase)
    uc.execute = AsyncMock(return_value=None)
    return uc


@pytest.fixture(autouse=True)
def _wire_router_and_overrides(
    mock_create_use_case: AsyncMock,
    mock_validate_use_case: AsyncMock,
    mock_list_use_case: AsyncMock,
    mock_get_use_case: AsyncMock,
    mock_update_use_case: AsyncMock,
    mock_delete_use_case: AsyncMock,
):
    """Include the MCP registry router (idempotently) and wire mocked use cases."""
    # Include the router on the app if not already present
    existing_paths = {getattr(r, "path", "") for r in app.routes}
    if not any(p.startswith(BASE_URL) for p in existing_paths):
        app.include_router(mcp_registry_router)

    app.dependency_overrides[get_create_mcp_server_use_case] = lambda: mock_create_use_case
    app.dependency_overrides[get_validate_mcp_server_use_case] = lambda: mock_validate_use_case
    app.dependency_overrides[get_list_mcp_servers_use_case] = lambda: mock_list_use_case
    app.dependency_overrides[get_get_mcp_server_use_case] = lambda: mock_get_use_case
    app.dependency_overrides[get_update_mcp_server_use_case] = lambda: mock_update_use_case
    app.dependency_overrides[get_delete_mcp_server_use_case] = lambda: mock_delete_use_case

    yield

    app.dependency_overrides.clear()


@pytest.fixture
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# -- POST /api/v1/mcp/servers (create, external) -------------------------------


class TestCreateMcpServerRoute:
    """Tests for POST /api/v1/mcp/servers (external source_type)."""

    async def test_returns_201_with_name(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=2)
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "headers": {},
            "env": {},
            "auth_token": "tok",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["name"] == "web-search"

    async def test_returns_201_with_tool_count(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=2)
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["tool_count"] == 2

    async def test_masks_auth_token_in_response(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange — the route must NOT return the plaintext auth_token
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "auth_token": "secret",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["auth_token"] is None

    async def test_masks_headers_in_response(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "headers": {"Authorization": "Bearer secret"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["headers"] == {}

    async def test_masks_env_in_response(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "env": {"API_KEY": "abc"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["env"] == {}

    async def test_returns_409_when_already_exists(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            side_effect=McpServerAlreadyExistsError("already exists")
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 409

    async def test_returns_422_when_unreachable(self, client: AsyncClient, mock_create_use_case: AsyncMock):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            side_effect=McpServerUnreachableError("unreachable")
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 422

    async def test_rejects_missing_name(self, client: AsyncClient):
        # Act
        resp = await client.post(BASE_URL, json={"url": "http://x/mcp"})

        # Assert
        assert resp.status_code == 422

    async def test_rejects_missing_url_for_external(self, client: AsyncClient):
        # Act
        resp = await client.post(BASE_URL, json={"name": "web-search"})

        # Assert — external source_type requires url
        assert resp.status_code == 422


# -- POST /api/v1/mcp/servers (create, openapi) --------------------------------


class TestCreateMcpServerRouteOpenApi:
    """Tests for POST /api/v1/mcp/servers with source_type=openapi."""

    async def test_returns_201_with_source_type_openapi(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry("petstore", tool_count=2)
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://petstore.example.com/openapi.json",
            "headers": {"Authorization": "Bearer upstream-token"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["source_type"] == "openapi"

    async def test_returns_201_with_absolute_mounted_url(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry(
                "petstore",
                url="http://raganything-api:8000/generated/petstore/mcp",
                tool_count=2,
            )
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["url"] == "http://raganything-api:8000/generated/petstore/mcp"

    async def test_rejects_openapi_without_openapi_url(self, client: AsyncClient):
        # Act
        resp = await client.post(
            BASE_URL, json={"name": "petstore", "source_type": "openapi"}
        )

        # Assert — openapi_url required when source_type=openapi
        assert resp.status_code == 422

    async def test_masks_headers_in_openapi_response(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — upstream headers must NOT be leaked
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry("petstore")
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
            "headers": {"Authorization": "Bearer secret"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["headers"] == {}


# -- POST /api/v1/mcp/servers/validate -----------------------------------------


class TestValidateMcpServerRoute:
    """Tests for POST /api/v1/mcp/servers/validate (dry-run)."""

    async def test_returns_200_with_tool_count(self, client: AsyncClient, mock_validate_use_case: AsyncMock):
        # Arrange
        mock_validate_use_case.execute = AsyncMock(return_value=5)
        body = {"url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 5

    async def test_returns_422_when_unreachable(self, client: AsyncClient, mock_validate_use_case: AsyncMock):
        # Arrange
        mock_validate_use_case.execute = AsyncMock(
            side_effect=McpServerUnreachableError("unreachable")
        )
        body = {"url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 422

    async def test_openapi_validate_returns_tool_count(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute_openapi = AsyncMock(return_value=10)
        body = {
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 10

    async def test_openapi_validate_calls_execute_openapi(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute_openapi = AsyncMock(return_value=3)
        body = {
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
            "headers": {"Authorization": "Bearer tok"},
        }

        # Act
        await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        mock_validate_use_case.execute_openapi.assert_awaited_once()


# -- GET /api/v1/mcp/servers (list) --------------------------------------------


class TestListMcpServersRoute:
    """Tests for GET /api/v1/mcp/servers."""

    async def test_returns_200_with_list(self, client: AsyncClient, mock_list_use_case: AsyncMock):
        # Act
        resp = await client.get(BASE_URL)

        # Assert
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 2

    async def test_masks_secrets_in_list_response(self, client: AsyncClient, mock_list_use_case: AsyncMock):
        # Act
        resp = await client.get(BASE_URL)

        # Assert
        for item in resp.json():
            assert item["auth_token"] is None
            assert item["headers"] == {}
            assert item["env"] == {}


# -- GET /api/v1/mcp/servers/{name} (get) --------------------------------------


class TestGetMcpServerRoute:
    """Tests for GET /api/v1/mcp/servers/{name}."""

    async def test_returns_200_with_entry(self, client: AsyncClient, mock_get_use_case: AsyncMock):
        # Arrange
        mock_get_use_case.execute = AsyncMock(return_value=_masked_entry("web-search"))
        # Act
        resp = await client.get(f"{BASE_URL}/web-search")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["name"] == "web-search"

    async def test_masks_secrets_in_get_response(self, client: AsyncClient, mock_get_use_case: AsyncMock):
        # Arrange
        mock_get_use_case.execute = AsyncMock(return_value=_masked_entry("web-search"))

        # Act
        resp = await client.get(f"{BASE_URL}/web-search")

        # Assert
        assert resp.json()["auth_token"] is None
        assert resp.json()["headers"] == {}

    async def test_returns_404_when_not_found(self, client: AsyncClient, mock_get_use_case: AsyncMock):
        # Arrange
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.get(f"{BASE_URL}/missing")

        # Assert
        assert resp.status_code == 404


# -- GET /api/v1/mcp/servers/{name}/reveal -------------------------------------


class TestRevealMcpServerRoute:
    """Tests for GET /api/v1/mcp/servers/{name}/reveal (plaintext secrets)."""

    async def test_returns_200_with_plaintext_secrets(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange — reveal returns the entry with secrets intact
        mock_get_use_case.execute = AsyncMock(return_value=_entry("web-search", auth_token="secret"))

        # Act
        resp = await client.get(f"{BASE_URL}/web-search/reveal")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["auth_token"] == "secret"
        assert resp.json()["headers"] == {"Authorization": "Bearer tok"}

    async def test_returns_404_when_not_found(self, client: AsyncClient, mock_get_use_case: AsyncMock):
        # Arrange
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.get(f"{BASE_URL}/missing/reveal")

        # Assert
        assert resp.status_code == 404


# -- PUT /api/v1/mcp/servers/{name} (update) -----------------------------------


class TestUpdateMcpServerRoute:
    """Tests for PUT /api/v1/mcp/servers/{name}."""

    async def test_returns_200_with_updated_entry(
        self, client: AsyncClient, mock_update_use_case: AsyncMock
    ):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=5)
        )
        body = {"url": "http://new:3001/mcp"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 5

    async def test_masks_secrets_in_update_response(
        self, client: AsyncClient, mock_update_use_case: AsyncMock
    ):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {"url": "http://new:3001/mcp", "auth_token": "new-secret"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.json()["auth_token"] is None

    async def test_returns_404_when_not_found(self, client: AsyncClient, mock_update_use_case: AsyncMock):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )
        body = {"url": "http://new:3001/mcp"}

        # Act
        resp = await client.put(f"{BASE_URL}/missing", json=body)

        # Assert
        assert resp.status_code == 404


# -- DELETE /api/v1/mcp/servers/{name} -----------------------------------------


class TestDeleteMcpServerRoute:
    """Tests for DELETE /api/v1/mcp/servers/{name}."""

    async def test_returns_204_on_success(self, client: AsyncClient, mock_delete_use_case: AsyncMock):
        # Act
        resp = await client.delete(f"{BASE_URL}/web-search")

        # Assert
        assert resp.status_code == 204

    async def test_returns_404_when_not_found(self, client: AsyncClient, mock_delete_use_case: AsyncMock):
        # Arrange
        mock_delete_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.delete(f"{BASE_URL}/missing")

        # Assert
        assert resp.status_code == 404
