"""Unit tests for the Swagger 2.0 -> OpenAPI 3.0 converter.

Each test covers one mapping rule from the conversion table in the module
docstring. The converter is a pure function (no I/O, no network).
"""

import pytest

from infrastructure.openapi_mcp.swagger2_converter import (
    convert_swagger2_to_openapi3,
)


def _minimal_swagger2(**overrides) -> dict:
    spec = {
        "swagger": "2.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {},
    }
    spec.update(overrides)
    return spec


class TestVersionGuard:
    """Input validation — only Swagger 2.0 docs are accepted."""

    def test_raises_on_non_dict(self):
        with pytest.raises(ValueError, match="not a Swagger 2.0"):
            convert_swagger2_to_openapi3("not a dict")  # type: ignore[arg-type]

    def test_raises_when_swagger_key_missing(self):
        with pytest.raises(ValueError, match="not a Swagger 2.0"):
            convert_swagger2_to_openapi3({"openapi": "3.0.0", "paths": {}})

    def test_raises_when_swagger_version_is_not_2_0(self):
        with pytest.raises(ValueError, match="not a Swagger 2.0"):
            convert_swagger2_to_openapi3({"swagger": "1.2", "paths": {}})


class TestServers:
    """host + basePath + schemes -> servers[{url}]."""

    def test_derives_server_url_from_host_basepath_schemes(self):
        spec = _minimal_swagger2(host="api.example.com", basePath="/v2", schemes=["https", "http"])
        result = convert_swagger2_to_openapi3(spec)
        assert result["servers"] == [{"url": "https://api.example.com/v2"}]

    def test_uses_first_scheme_when_multiple(self):
        spec = _minimal_swagger2(host="api.example.com", basePath="/v1", schemes=["http", "https"])
        result = convert_swagger2_to_openapi3(spec)
        assert result["servers"] == [{"url": "http://api.example.com/v1"}]

    def test_defaults_to_https_when_no_schemes(self):
        spec = _minimal_swagger2(host="api.example.com", basePath="/v1")
        result = convert_swagger2_to_openapi3(spec)
        assert result["servers"] == [{"url": "https://api.example.com/v1"}]

    def test_no_servers_when_host_absent(self):
        result = convert_swagger2_to_openapi3(_minimal_swagger2())
        assert "servers" not in result


class TestComponents:
    """definitions/parameters/responses -> components.*."""

    def test_definitions_become_components_schemas(self):
        spec = _minimal_swagger2(definitions={"Pet": {"type": "object"}})
        result = convert_swagger2_to_openapi3(spec)
        assert result["components"]["schemas"] == {"Pet": {"type": "object"}}

    def test_top_level_parameters_become_components_parameters(self):
        spec = _minimal_swagger2(parameters={"limitParam": {"name": "limit", "in": "query", "type": "integer"}})
        result = convert_swagger2_to_openapi3(spec)
        assert "limitParam" in result["components"]["parameters"]

    def test_top_level_responses_become_components_responses(self):
        spec = _minimal_swagger2(responses={"NotFound": {"description": "not found"}})
        result = convert_swagger2_to_openapi3(spec)
        assert "NotFound" in result["components"]["responses"]

    def test_no_components_key_when_none_present(self):
        result = convert_swagger2_to_openapi3(_minimal_swagger2())
        assert "components" not in result


class TestRefRewriting:
    """$ref paths are rewritten recursively."""

    def test_definitions_ref_rewritten(self):
        spec = _minimal_swagger2(
            definitions={"Pet": {"type": "object"}},
            paths={
                "/pet": {
                    "post": {
                        "operationId": "addPet",
                        "parameters": [{"in": "body", "name": "body", "schema": {"$ref": "#/definitions/Pet"}}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        )
        result = convert_swagger2_to_openapi3(spec)
        schema = result["paths"]["/pet"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/Pet"}

    def test_parameters_ref_rewritten(self):
        spec = _minimal_swagger2(
            parameters={"limitParam": {"name": "limit", "in": "query", "type": "integer"}},
            paths={
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "parameters": [{"$ref": "#/parameters/limitParam"}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["paths"]["/pets"]["get"]["parameters"][0] == {"$ref": "#/components/parameters/limitParam"}

    def test_responses_ref_rewritten(self):
        spec = _minimal_swagger2(
            responses={"NotFound": {"description": "nf"}},
            paths={
                "/pets": {
                    "get": {
                        "operationId": "list",
                        "responses": {"404": {"$ref": "#/responses/NotFound"}},
                    }
                }
            },
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["paths"]["/pets"]["get"]["responses"]["404"] == {"$ref": "#/components/responses/NotFound"}

    def test_refs_rewritten_in_nested_definitions(self):
        spec = _minimal_swagger2(
            definitions={
                "Order": {"type": "object", "properties": {"pet": {"$ref": "#/definitions/Pet"}}},
                "Pet": {"type": "object"},
            },
        )
        result = convert_swagger2_to_openapi3(spec)
        order = result["components"]["schemas"]["Order"]
        assert order["properties"]["pet"] == {"$ref": "#/components/schemas/Pet"}


class TestBodyParameter:
    """in: body parameter -> requestBody."""

    def test_body_param_becomes_request_body(self):
        spec = _minimal_swagger2(
            paths={
                "/pet": {
                    "post": {
                        "operationId": "addPet",
                        "parameters": [
                            {"in": "body", "name": "body", "required": True, "schema": {"type": "object"}}
                        ],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        rb = result["paths"]["/pet"]["post"]["requestBody"]
        assert rb["required"] is True
        assert rb["content"]["application/json"]["schema"] == {"type": "object"}

    def test_body_param_uses_operation_consumes(self):
        spec = _minimal_swagger2(
            paths={
                "/pet": {
                    "post": {
                        "operationId": "addPet",
                        "consumes": ["application/xml"],
                        "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        assert "application/xml" in result["paths"]["/pet"]["post"]["requestBody"]["content"]

    def test_body_param_uses_global_consumes(self):
        spec = _minimal_swagger2(
            consumes=["application/xml"],
            paths={
                "/pet": {
                    "post": {
                        "operationId": "addPet",
                        "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        )
        result = convert_swagger2_to_openapi3(spec)
        assert "application/xml" in result["paths"]["/pet"]["post"]["requestBody"]["content"]


class TestFormDataParameters:
    """in: formData parameters -> requestBody (urlencoded or multipart)."""

    def test_form_params_become_urlencoded_request_body(self):
        spec = _minimal_swagger2(
            paths={
                "/upload": {
                    "post": {
                        "operationId": "upload",
                        "parameters": [
                            {"in": "formData", "name": "name", "type": "string", "required": True},
                            {"in": "formData", "name": "age", "type": "integer"},
                        ],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        rb = result["paths"]["/upload"]["post"]["requestBody"]
        assert "application/x-www-form-urlencoded" in rb["content"]
        schema = rb["content"]["application/x-www-form-urlencoded"]["schema"]
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["age"] == {"type": "integer"}
        assert schema["required"] == ["name"]

    def test_file_form_param_becomes_multipart_with_binary(self):
        spec = _minimal_swagger2(
            paths={
                "/upload": {
                    "post": {
                        "operationId": "uploadFile",
                        "parameters": [
                            {"in": "formData", "name": "file", "type": "file"},
                            {"in": "formData", "name": "note", "type": "string"},
                        ],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        rb = result["paths"]["/upload"]["post"]["requestBody"]
        assert "multipart/form-data" in rb["content"]
        schema = rb["content"]["multipart/form-data"]["schema"]
        assert schema["properties"]["file"] == {"type": "string", "format": "binary"}


class TestResponses:
    """response schema -> content[produces[0]].schema."""

    def test_response_schema_becomes_content(self):
        spec = _minimal_swagger2(
            paths={
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "produces": ["application/xml"],
                        "responses": {
                            "200": {"description": "ok", "schema": {"type": "array", "items": {"type": "string"}}}
                        },
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        resp = result["paths"]["/pets"]["get"]["responses"]["200"]
        assert resp["content"]["application/xml"]["schema"] == {"type": "array", "items": {"type": "string"}}

    def test_response_without_schema_has_no_content(self):
        spec = _minimal_swagger2(
            paths={
                "/pets": {
                    "get": {
                        "operationId": "list",
                        "responses": {"204": {"description": "no content"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        assert "content" not in result["paths"]["/pets"]["get"]["responses"]["204"]

    def test_file_response_schema_becomes_binary(self):
        spec = _minimal_swagger2(
            paths={
                "/download": {
                    "get": {
                        "operationId": "download",
                        "produces": ["application/octet-stream"],
                        "responses": {"200": {"description": "file", "schema": {"type": "file"}}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        schema = result["paths"]["/download"]["get"]["responses"]["200"]["content"]["application/octet-stream"]["schema"]
        assert schema == {"type": "string", "format": "binary"}


class TestSecurityDefinitions:
    """securityDefinitions -> components.securitySchemes."""

    def test_basic_auth(self):
        spec = _minimal_swagger2(securityDefinitions={"basicAuth": {"type": "basic"}})
        result = convert_swagger2_to_openapi3(spec)
        assert result["components"]["securitySchemes"]["basicAuth"] == {"type": "http", "scheme": "basic"}

    def test_api_key(self):
        spec = _minimal_swagger2(
            securityDefinitions={"apiKey": {"type": "apiKey", "name": "X-API-Key", "in": "header"}}
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["components"]["securitySchemes"]["apiKey"] == {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
        }

    def test_oauth2_implicit(self):
        spec = _minimal_swagger2(
            securityDefinitions={
                "petstore_auth": {
                    "type": "oauth2",
                    "flow": "implicit",
                    "authorizationUrl": "https://example.com/auth",
                    "scopes": {"read:pets": "read pets"},
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        scheme = result["components"]["securitySchemes"]["petstore_auth"]
        assert scheme["type"] == "oauth2"
        assert scheme["flows"]["implicit"]["authorizationUrl"] == "https://example.com/auth"
        assert scheme["flows"]["implicit"]["scopes"] == {"read:pets": "read pets"}

    def test_oauth2_access_code(self):
        spec = _minimal_swagger2(
            securityDefinitions={
                "auth": {
                    "type": "oauth2",
                    "flow": "accessCode",
                    "authorizationUrl": "https://example.com/auth",
                    "tokenUrl": "https://example.com/token",
                    "scopes": {},
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        flow = result["components"]["securitySchemes"]["auth"]["flows"]["authorizationCode"]
        assert flow["authorizationUrl"] == "https://example.com/auth"
        assert flow["tokenUrl"] == "https://example.com/token"

    def test_oauth2_password(self):
        spec = _minimal_swagger2(
            securityDefinitions={"pwd": {"type": "oauth2", "flow": "password", "tokenUrl": "https://example.com/token", "scopes": {}}}
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["components"]["securitySchemes"]["pwd"]["flows"]["password"]["tokenUrl"] == "https://example.com/token"

    def test_oauth2_application(self):
        spec = _minimal_swagger2(
            securityDefinitions={"app": {"type": "oauth2", "flow": "application", "tokenUrl": "https://example.com/token", "scopes": {}}}
        )
        result = convert_swagger2_to_openapi3(spec)
        assert "clientCredentials" in result["components"]["securitySchemes"]["app"]["flows"]


class TestNonBodyParameters:
    """query/path/header parameters are kept; collectionFormat is dropped."""

    def test_query_parameter_kept(self):
        spec = _minimal_swagger2(
            paths={
                "/pets": {
                    "get": {
                        "operationId": "list",
                        "parameters": [
                            {"name": "limit", "in": "query", "type": "integer", "collectionFormat": "multi"}
                        ],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        param = result["paths"]["/pets"]["get"]["parameters"][0]
        assert param["name"] == "limit"
        assert param["in"] == "query"
        assert "collectionFormat" not in param

    def test_path_level_parameters_kept(self):
        spec = _minimal_swagger2(
            paths={
                "/pets/{id}": {
                    "parameters": [{"name": "id", "in": "path", "type": "integer", "required": True}],
                    "get": {"operationId": "getPet", "responses": {"200": {"description": "ok"}}},
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["paths"]["/pets/{id}"]["parameters"][0]["name"] == "id"


class TestPassthrough:
    """Top-level keys like security/tags/externalDocs pass through."""

    def test_security_passthrough(self):
        spec = _minimal_swagger2(
            securityDefinitions={"apiKey": {"type": "apiKey", "name": "X-Key", "in": "header"}},
            security=[{"apiKey": []}],
        )
        result = convert_swagger2_to_openapi3(spec)
        assert result["security"] == [{"apiKey": []}]

    def test_operation_id_preserved(self):
        spec = _minimal_swagger2(
            paths={
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "summary": "List pets",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        )
        result = convert_swagger2_to_openapi3(spec)
        op = result["paths"]["/pets"]["get"]
        assert op["operationId"] == "listPets"
        assert op["summary"] == "List pets"

    def test_openapi_version_is_3_0_0(self):
        result = convert_swagger2_to_openapi3(_minimal_swagger2())
        assert result["openapi"] == "3.0.0"

    def test_does_not_mutate_input(self):
        spec = _minimal_swagger2(
            definitions={"Pet": {"type": "object"}},
            paths={
                "/pet": {
                    "post": {
                        "operationId": "add",
                        "parameters": [{"in": "body", "name": "body", "schema": {"$ref": "#/definitions/Pet"}}],
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        )
        convert_swagger2_to_openapi3(spec)
        # Input untouched
        assert spec["swagger"] == "2.0"
        assert spec["paths"]["/pet"]["post"]["parameters"][0]["schema"] == {"$ref": "#/definitions/Pet"}
