import asyncio
import json
import logging
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
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            return resp.read(), dict(resp.headers)

    def _post(self, url: str, payload: dict, headers: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            return resp.read()

    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        url = f"{self._base_url}/api/projects/{project_id}/documents/ai"
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
        documents = [BricksDocumentInfo(**item) for item in items]
        return documents

    async def download_document(
        self,
        document_id: str,
        project_id: str,
    ) -> tuple[bytes, str]:
        documents = await self.list_project_documents(project_id)
        url = None
        for doc in documents:
            if doc.id == document_id and doc.url:
                url = doc.url
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
            raise RuntimeError(
                f"Failed to download document (HTTP {e.code})"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Document download connection failed: {e.reason}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Document download timed out: {e}") from e

        filename = _extract_filename(resp_headers.get("Content-Disposition", ""), url)
        return body, filename

    async def publish_section_version(self, payload: dict) -> SectionVersionResult:
        url = f"{self._base_url}/api/section-versions"
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
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
        return SectionVersionResult(success=True, message="Published", data=data)


def _extract_filename(content_disposition: str, url: str = "") -> str:
    match = re.search(r'filename="([^"]+)"', content_disposition)
    if match:
        return match.group(1)
    match = re.search(r"filename=([^\s;]+)", content_disposition)
    if match:
        return match.group(1)
    if url:
        decoded_path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        path_filename = decoded_path.rsplit("/", 1)[-1]
        if path_filename and "." in path_filename:
            return path_filename
    return "document.bin"