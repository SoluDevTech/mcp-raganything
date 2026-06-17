import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from domain.ports.bricks_api_port import (
    BricksApiPort,
    BricksDocumentInfo,
    SectionVersionResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


class BricksApiAdapter(BricksApiPort):
    def __init__(self, config) -> None:
        self._base_url = config.BRICKS_API_BASE_URL.rstrip("/")
        self._api_key = config.BRICKS_API_KEY
        self._bearer_token = config.BRICKS_BEARER_TOKEN

    async def close(self) -> None:
        pass

    def _get(self, url: str, headers: dict | None = None) -> tuple[bytes, dict]:
        logger.debug("GET %s", url)
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = resp.read()
                resp_headers = dict(resp.headers)
                logger.debug(
                    "GET %s -> %d bytes (status=%s)", url, len(body), resp.status
                )
                return body, resp_headers
        except urllib.error.HTTPError as e:
            error_body = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            logger.error("GET %s -> HTTP %d: %s", url, e.code, error_body[:500])
            raise
        except Exception as e:
            logger.error("GET %s -> error: %s", url, e)
            raise

    def _post(self, url: str, payload: dict, headers: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        logger.debug("POST %s (body=%d bytes)", url, len(data))
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = resp.read()
                logger.debug(
                    "POST %s -> %d bytes (status=%s)", url, len(body), resp.status
                )
                return body
        except urllib.error.HTTPError as e:
            error_body = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            logger.error("POST %s -> HTTP %d: %s", url, e.code, error_body[:500])
            raise
        except Exception as e:
            logger.error("POST %s -> error: %s", url, e)
            raise

    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        url = f"{self._base_url}/api/projects/{project_id}/documents/ai"
        logger.info("Listing Bricks documents for project %s", project_id)
        try:
            body, _ = await asyncio.to_thread(
                self._get, url, {"Authorization": f"Bearer {self._bearer_token}"}
            )
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    f"Bricks API authentication failed (HTTP {e.code})"
                ) from e
            if e.code == 404:
                raise FileNotFoundError(
                    f"Bricks project not found: {project_id}"
                ) from e
            raise RuntimeError(f"Bricks API error (HTTP {e.code})") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Bricks API connection failed: {e.reason}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Bricks API request timed out: {e}") from e
        items = json.loads(body).get("items", [])
        logger.info("Found %d Bricks documents for project %s", len(items), project_id)
        documents = [BricksDocumentInfo(**item) for item in items]
        return documents

    async def download_document(
        self,
        document_id: str,
        project_id: str,
    ) -> tuple[bytes, str, str]:
        logger.info(
            "Downloading Bricks document %s from project %s", document_id, project_id
        )
        documents = await self.list_project_documents(project_id)
        url = None
        mime_type = ""
        for doc in documents:
            if doc.id == document_id and doc.url:
                url = doc.url
                mime_type = doc.mime_type
                break
        if not url:
            raise FileNotFoundError(
                f"Document {document_id} not found in project {project_id}"
            )
        try:
            body, resp_headers = await asyncio.to_thread(self._get, url)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    f"Document download authentication failed (HTTP {e.code})"
                ) from e
            if e.code == 404:
                raise FileNotFoundError(
                    f"Document {document_id} not found (project {project_id})"
                ) from e
            raise RuntimeError(f"Failed to download document (HTTP {e.code})") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Document download connection failed: {e.reason}"
            ) from e
        except TimeoutError as e:
            raise TimeoutError(f"Document download timed out: {e}") from e

        filename = _extract_filename(resp_headers.get("Content-Disposition", ""), url)
        logger.info(
            "Downloaded Bricks document %s (%d bytes, mime=%s, filename=%s)",
            document_id,
            len(body),
            mime_type,
            filename,
        )
        return body, filename, mime_type

    async def publish_section_version(self, payload: dict) -> SectionVersionResult:
        url = f"{self._base_url}/api/section-versions"
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        logger.info(
            "Publishing section version: project=%s section=%s workflow=%s",
            payload.get("projectUniqueId"),
            payload.get("sectionKey"),
            payload.get("workflowId"),
        )
        logger.info(
            "Publish payload: %s", json.dumps(payload, ensure_ascii=False, default=str)
        )
        try:
            body = await asyncio.to_thread(self._post, url, payload, headers)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    f"Publish authentication failed (HTTP {e.code})"
                ) from e
            raise RuntimeError(f"Publish failed (HTTP {e.code})") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Publish connection failed: {e.reason}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Publish request timed out: {e}") from e
        data = json.loads(body)
        logger.info("Published section version successfully: %s", data)
        return SectionVersionResult(success=True, message="Published", data=data)


def _extract_filename(content_disposition: str, url: str = "") -> str:
    match = re.search(r'filename="([^"]+)"', content_disposition)
    if match:
        filename = match.group(1)
        logger.debug("Filename from Content-Disposition (quoted): %s", filename)
        return _normalize_extension(filename)
    match = re.search(r"filename=([^\s;]+)", content_disposition)
    if match:
        filename = match.group(1)
        logger.debug("Filename from Content-Disposition (unquoted): %s", filename)
        return _normalize_extension(filename)
    if url:
        decoded_path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        path_filename = decoded_path.rsplit("/", 1)[-1]
        if path_filename and "." in path_filename:
            logger.debug(
                "Filename from URL path: %s (url=%s)", path_filename, url[:200]
            )
            return _normalize_extension(path_filename)
    logger.warning(
        "Could not extract filename, falling back to document.bin (url=%s)",
        url[:200] if url else "",
    )
    return "document.bin"


def _normalize_extension(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    if not ext or ext == ".":
        return filename
    cleaned = re.sub(r"^[^a-zA-Z]+", ".", ext.lower())
    if cleaned == ext.lower():
        return filename
    logger.warning(
        "Normalized file extension: %s -> %s (filename=%s)", ext, cleaned, filename
    )
    return name + cleaned
