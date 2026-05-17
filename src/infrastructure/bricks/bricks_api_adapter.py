import logging
import re

import httpx

from domain.ports.bricks_api_port import (
    BricksApiPort,
    BricksDocumentInfo,
    SectionVersionResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(30.0)


class BricksApiAdapter(BricksApiPort):
    def __init__(self, config) -> None:
        self._base_url = config.BRICKS_API_BASE_URL
        self._api_key = config.BRICKS_API_KEY
        self._bearer_token = config.BRICKS_BEARER_TOKEN
        self._publish_target_url = config.BRICKS_PUBLISH_TARGET_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=_DEFAULT_TIMEOUT
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        url = f"api/projects/{project_id}/documents/ai"
        try:
            response = await self._client.get(
                url,
                headers={"Authorization": f"Bearer {self._bearer_token}"},
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Bricks API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise PermissionError(
                    f"Bricks API authentication failed (HTTP {e.response.status_code})"
                ) from e
            if e.response.status_code == 404:
                raise FileNotFoundError(
                    f"Bricks project not found: {project_id}"
                ) from e
            raise RuntimeError(
                f"Bricks API error (HTTP {e.response.status_code})"
            ) from e
        except httpx.RequestError as e:
            raise ConnectionError(f"Bricks API connection failed: {e}") from e
        items = response.json().get("items", [])
        return [BricksDocumentInfo(**item) for item in items]

    async def download_document(self, download_url: str) -> tuple[bytes, str]:
        try:
            response = await self._client.get(download_url)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Document download timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Failed to download document (HTTP {e.response.status_code})"
            ) from e
        except httpx.RequestError as e:
            raise ConnectionError(f"Document download connection failed: {e}") from e
        content_disposition = response.headers.get("content-disposition", "")
        filename = "document.bin"
        match = re.search(r'filename="([^"]+)"', content_disposition)
        if match:
            filename = match.group(1)
        else:
            match = re.search(r"filename=([^\s;]+)", content_disposition)
            if match:
                filename = match.group(1)
        return response.content, filename

    async def publish_section_version(self, payload: dict) -> SectionVersionResult:
        try:
            response = await self._client.post(
                self._publish_target_url,
                headers={
                    "X-API-Key": self._api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Publish request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise PermissionError(
                    f"Publish authentication failed (HTTP {e.response.status_code})"
                ) from e
            raise RuntimeError(f"Publish failed (HTTP {e.response.status_code})") from e
        except httpx.RequestError as e:
            raise ConnectionError(f"Publish connection failed: {e}") from e
        data = response.json()
        return SectionVersionResult(success=True, message="Published", data=data)
