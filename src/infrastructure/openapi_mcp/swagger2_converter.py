"""Convert a Swagger 2.0 document into an OpenAPI 3.0 document (pure, offline).

``fastmcp.FastMCP.from_openapi`` only understands OpenAPI 3.x; many real-world
specs (e.g. the public Petstore at ``petstore.swagger.io/v2/swagger.json``) are
still Swagger 2.0. Rather than relying on an external conversion service (which
would leak the user's spec to a third party and add a network dependency), this
module performs the subset of the 2.0 -> 3.0 mapping that ``FastMCP`` consumes.

Mapping coverage (the constructs ``FastMCP.from_openapi`` reads):

============================== ========================================
Swagger 2.0                   OpenAPI 3.0
============================== ========================================
``host`` + ``basePath`` +     ``servers: [{url}]``
``schemes[0]``
``definitions``                ``components.schemas``
``parameters`` / ``responses`` ``components.parameters`` / ``.responses``
(top-level)
``$ref: #/definitions/...``   ``#/components/schemas/...`` (recursive)
parameter ``in: body``        ``requestBody.content[consumes[0]]``
parameters ``in: formData``   ``requestBody`` (x-www-form-urlencoded,
                               or multipart/form-data when a field is
                               ``type: file``)
``type: file``                ``type: string, format: binary``
``produces`` + response       ``responses.*.content[produces[0]]``
``schema``
``securityDefinitions``       ``components.securitySchemes``
(basic / apiKey / oauth2 x4)  (http / apiKey / oauth2 flows)
============================== ========================================

Best-effort / deliberately out of scope:

- ``collectionFormat`` is dropped (``FastMCP`` does not use it).
- External ``$ref`` (file/URL) are passed through unchanged; resolving them is
  the spec author's job (or a separate concern).
- ``allOf`` / ``oneOf`` / ``anyOf`` / discriminators pass through unchanged.
- Swagger 1.x is not supported (raise ``ValueError``).

The function is pure: it takes a parsed dict and returns a new dict; no I/O,
no network, no global state. Any unexpected input shape raises ``ValueError``
so the adapter can wrap it into an :class:`OpenApiInvalidSpecError`.
"""

from __future__ import annotations

import copy
from typing import Any

#: Default ``consumes`` when neither the spec nor the operation declares one.
_DEFAULT_CONSUMES: list[str] = ["application/json"]

#: Default ``produces`` when neither the spec nor the operation declares one.
_DEFAULT_PRODUCES: list[str] = ["application/json"]

#: Mapping from Swagger 2.0 oauth2 ``flow`` names to OpenAPI 3.0 flow names.
_OAUTH2_FLOW_MAP: dict[str, str] = {
    "implicit": "implicit",
    "password": "password",
    "application": "clientCredentials",
    "accessCode": "authorizationCode",
}


def convert_swagger2_to_openapi3(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed Swagger 2.0 document to an OpenAPI 3.0 document.

    Args:
        spec: The parsed Swagger 2.0 document (must contain ``swagger: "2.0"``).

    Returns:
        A new dict containing an OpenAPI 3.0 document.

    Raises:
        ValueError: If the input is not a Swagger 2.0 document
            (missing ``swagger`` key or version != ``"2.0"``).
    """
    if not isinstance(spec, dict) or spec.get("swagger") != "2.0":
        raise ValueError("not a Swagger 2.0 document (missing 'swagger: 2.0')")

    src = copy.deepcopy(spec)

    global_consumes = src.pop("consumes", _DEFAULT_CONSUMES) or _DEFAULT_CONSUMES
    global_produces = src.pop("produces", _DEFAULT_PRODUCES) or _DEFAULT_PRODUCES
    host = src.pop("host", "")
    base_path = src.pop("basePath", "/") or "/"
    schemes = src.pop("schemes", ["https"]) or ["https"]

    out: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": src.pop("info", {"title": "Untitled", "version": "0.0.0"}),
    }
    if host:
        scheme = schemes[0] if schemes else "https"
        out["servers"] = [{"url": f"{scheme}://{host}{base_path}"}]

    components = _build_components(src)
    if components:
        out["components"] = components

    out["paths"] = _convert_paths(src.pop("paths", {}) or {}, global_consumes, global_produces)

    for passthrough in ("security", "tags", "externalDocs"):
        if passthrough in src:
            out[passthrough] = src[passthrough]

    return _rewrite_refs(out)


def _build_components(src: dict[str, Any]) -> dict[str, Any]:
    """Move definitions/parameters/responses/securityDefinitions into components."""
    components: dict[str, Any] = {}
    if "definitions" in src:
        components["schemas"] = src.pop("definitions")
    if "parameters" in src:
        components["parameters"] = src.pop("parameters")
    if "responses" in src:
        components["responses"] = src.pop("responses")
    if "securityDefinitions" in src:
        components["securitySchemes"] = _convert_security_definitions(src.pop("securityDefinitions"))
    return components


def _convert_security_definitions(security_defs: dict[str, Any]) -> dict[str, Any]:
    """Convert Swagger 2.0 ``securityDefinitions`` to OpenAPI 3.0 ``securitySchemes``."""
    out: dict[str, Any] = {}
    for name, d in (security_defs or {}).items():
        scheme_type = d.get("type")
        if scheme_type == "basic":
            out[name] = {"type": "http", "scheme": "basic"}
        elif scheme_type == "apiKey":
            out[name] = {
                "type": "apiKey",
                "name": d.get("name"),
                "in": d.get("in"),
            }
        elif scheme_type == "oauth2":
            flow_name = d.get("flow")
            flow: dict[str, Any] = {"scopes": d.get("scopes", {})}
            if flow_name in ("implicit", "accessCode"):
                flow["authorizationUrl"] = d.get("authorizationUrl", "")
            if flow_name in ("password", "application", "accessCode"):
                flow["tokenUrl"] = d.get("tokenUrl", "")
            out[name] = {
                "type": "oauth2",
                "flows": {_OAUTH2_FLOW_MAP.get(flow_name, flow_name or "implicit"): flow},
            }
    return out


def _convert_paths(
    paths: dict[str, Any],
    global_consumes: list[str],
    global_produces: list[str],
) -> dict[str, Any]:
    """Convert each path item, handling body/formData params and consumes/produces."""
    out: dict[str, Any] = {}
    for path, item in paths.items():
        out[path] = _convert_path_item(item or {}, global_consumes, global_produces)
    return out


def _convert_path_item(
    item: dict[str, Any],
    global_consumes: list[str],
    global_produces: list[str],
) -> dict[str, Any]:
    """Convert a single path item object."""
    new_item: dict[str, Any] = {}
    # Path-level parameters (shared across operations) are kept as-is.
    if "parameters" in item:
        new_item["parameters"] = [_convert_non_body_parameter(p) for p in item["parameters"]]

    http_methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
    for method, op in item.items():
        if method not in http_methods:
            new_item[method] = op
            continue
        new_item[method] = _convert_operation(op or {}, global_consumes, global_produces)

    return new_item


def _convert_operation(
    op: dict[str, Any],
    global_consumes: list[str],
    global_produces: list[str],
) -> dict[str, Any]:
    """Convert a single operation object."""
    new_op: dict[str, Any] = {}
    consumes = op.pop("consumes", global_consumes) or global_consumes
    produces = op.pop("produces", global_produces) or global_produces

    params, body, form = [], None, []
    for p in op.pop("parameters", []) or []:
        location = p.get("in")
        if location == "body":
            body = p
        elif location == "formData":
            form.append(p)
        else:
            params.append(_convert_non_body_parameter(p))

    request_body = _build_request_body(body, form, consumes)
    if request_body is not None:
        new_op["requestBody"] = request_body
    if params:
        new_op["parameters"] = params
    new_op["responses"] = _convert_responses(op.pop("responses", {}) or {}, produces)

    # Pass through the rest (operationId, summary, description, tags, security, ...).
    for key, value in op.items():
        new_op[key] = value
    return new_op


def _convert_non_body_parameter(param: dict[str, Any]) -> dict[str, Any]:
    """Convert a query/path/header parameter: drop collectionFormat, pass through."""
    p = dict(param)
    p.pop("collectionFormat", None)
    return p


def _build_request_body(
    body: dict[str, Any] | None,
    form: list[dict[str, Any]],
    consumes: list[str],
) -> dict[str, Any] | None:
    """Build an OpenAPI 3.0 ``requestBody`` from a body param or formData params."""
    if body is not None:
        body_schema = body.get("schema", {})
        return {
            "description": body.get("description", ""),
            "required": body.get("required", False),
            "content": {consumes[0]: {"schema": body_schema}},
        }

    if not form:
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []
    multipart = False
    for p in form:
        name = p.get("name")
        if name is None:
            continue
        field_schema = {
            k: v
            for k, v in p.items()
            if k not in ("name", "in", "required", "collectionFormat")
        }
        if field_schema.get("type") == "file":
            field_schema = {"type": "string", "format": "binary"}
            multipart = True
        properties[name] = field_schema
        if p.get("required"):
            required.append(name)

    content_type = "multipart/form-data" if multipart else "application/x-www-form-urlencoded"
    form_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        form_schema["required"] = required
    return {"content": {content_type: {"schema": form_schema}}}


def _convert_responses(
    responses: dict[str, Any],
    produces: list[str],
) -> dict[str, Any]:
    """Convert Swagger 2.0 responses to OpenAPI 3.0 responses (schema -> content)."""
    out: dict[str, Any] = {}
    for code, resp in responses.items():
        new_resp: dict[str, Any] = {}
        resp = dict(resp or {})
        schema = resp.pop("schema", None)
        if schema is not None:
            if isinstance(schema, dict) and schema.get("type") == "file":
                schema = {"type": "string", "format": "binary"}
            new_resp["content"] = {produces[0]: {"schema": schema}}
        for key, value in resp.items():
            new_resp[key] = value
        out[code] = new_resp
    return out


def _rewrite_refs(node: Any) -> Any:
    """Recursively rewrite Swagger 2.0 ``$ref`` paths to OpenAPI 3.0 paths."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                out[key] = (
                    value.replace("#/definitions/", "#/components/schemas/")
                    .replace("#/parameters/", "#/components/parameters/")
                    .replace("#/responses/", "#/components/responses/")
                )
            else:
                out[key] = _rewrite_refs(value)
        return out
    if isinstance(node, list):
        return [_rewrite_refs(item) for item in node]
    return node
