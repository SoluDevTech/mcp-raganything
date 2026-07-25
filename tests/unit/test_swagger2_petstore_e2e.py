"""End-to-end test: convert the real Petstore Swagger 2.0 spec and build a FastMCP server.

This locks in compatibility against a real-world Swagger 2.0 spec (the public
Petstore at petstore.swagger.io/v2). The spec is stored as a fixture to avoid
network calls in tests. The conversion + FastMCP.from_openapi chain must
produce a non-zero tool count.
"""

import json
from pathlib import Path

import httpx
import pytest

from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory
from infrastructure.openapi_mcp.swagger2_converter import convert_swagger2_to_openapi3

_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "openapi" / "petstore_swagger2.json"
)


@pytest.fixture(scope="module")
def petstore_swagger2_spec() -> dict:
    return json.loads(_FIXTURE.read_text())


class TestPetstoreEndToEnd:
    """Convert the real Petstore v2 spec and build a FastMCP server from it."""

    def test_fixture_is_swagger_2_0(self, petstore_swagger2_spec: dict):
        assert petstore_swagger2_spec["swagger"] == "2.0"
        assert "paths" in petstore_swagger2_spec
        assert len(petstore_swagger2_spec["paths"]) > 0

    def test_converts_to_openapi_3(self, petstore_swagger2_spec: dict):
        result = convert_swagger2_to_openapi3(petstore_swagger2_spec)
        assert result["openapi"].startswith("3.")
        assert "servers" in result
        assert "https://petstore.swagger.io/v2" in result["servers"][0]["url"]
        assert "components" in result
        assert "schemas" in result["components"]
        # Pet definition should be present
        assert "Pet" in result["components"]["schemas"]
        # $refs should be rewritten
        add_pet_op = result["paths"]["/pet"]["post"]
        body_schema = add_pet_op["requestBody"]["content"]["application/json"]["schema"]
        assert body_schema == {"$ref": "#/components/schemas/Pet"}

    async def test_fastmcp_builds_tools_from_converted_spec(self, petstore_swagger2_spec: dict):
        # Convert
        spec3 = convert_swagger2_to_openapi3(petstore_swagger2_spec)

        # Build via the real adapter (FastMCP.from_openapi + httpx.AsyncClient mocked)
        factory = FastMcpOpenApiFactory()
        client = httpx.AsyncClient(base_url="https://petstore.swagger.io/v2")
        mcp = await factory.build_mcp(spec3, base_url="", headers={})

        # Count tools via the real FastMCP server
        tools = await mcp.list_tools()
        assert len(tools) >= 15  # the Petstore exposes 20 operations
        tool_names = {t.name for t in tools}
        # A few well-known Petstore operations
        assert "addPet" in tool_names or "add_pet" in tool_names
        assert "getPetById" in tool_names or "get_pet_by_id" in tool_names

        await client.aclose()
