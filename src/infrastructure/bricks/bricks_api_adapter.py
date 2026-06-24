import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from domain.errors.bricks import (
    BricksApiError,
    BricksConnectionError,
    BricksNotFoundError,
    BricksPermissionError,
    BricksTimeoutError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.bricks_api_port import (
    BricksApiPort,
    BricksDocumentInfo,
    SectionVersionResult,
)

logger = logging.getLogger(__name__)


class BricksApiAdapter(BricksApiPort):
    def __init__(self, config) -> None:
        self._base_url = config.BRICKS_API_BASE_URL.rstrip("/")
        self._api_key = config.BRICKS_API_KEY
        self._bearer_token = config.BRICKS_BEARER_TOKEN
        self._http_timeout = config.BRICKS_HTTP_TIMEOUT

    async def close(self) -> None:
        pass

    def _get(self, url: str, headers: dict | None = None) -> tuple[bytes, dict]:
        logger.debug(LogMessage.BRICKS_GET, url)
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                body = resp.read()
                resp_headers = dict(resp.headers)
                logger.debug(
                    LogMessage.BRICKS_GET_BYTES, url, len(body), resp.status
                )
                return body, resp_headers
        except urllib.error.HTTPError as e:
            error_body = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            logger.error(LogMessage.BRICKS_GET_HTTP_ERROR, url, e.code, error_body[:500])
            raise
        except Exception as e:
            logger.error(LogMessage.BRICKS_GET_ERROR, url, e)
            raise

    def _post(self, url: str, payload: dict, headers: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        logger.debug(LogMessage.BRICKS_POST, url, len(data))
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                body = resp.read()
                logger.debug(
                    LogMessage.BRICKS_POST_BYTES, url, len(body), resp.status
                )
                return body
        except urllib.error.HTTPError as e:
            error_body = (
                e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            )
            logger.error(LogMessage.BRICKS_POST_HTTP_ERROR, url, e.code, error_body[:500])
            raise
        except Exception as e:
            logger.error(LogMessage.BRICKS_POST_ERROR, url, e)
            raise

    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        url = f"{self._base_url}/api/projects/{project_id}/documents/ai"
        logger.info(LogMessage.BRICKS_LISTING_DOCUMENTS, project_id)
        try:
            body, _ = await asyncio.to_thread(
                self._get, url, {"X-API-Key": f"{self._bearer_token}"}
            )
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise BricksPermissionError(
                    ErrorMessage.BRICKS_AUTH_FAILED.format(code=e.code)
                ) from e
            if e.code == 404:
                raise BricksNotFoundError(
                    ErrorMessage.BRICKS_PROJECT_NOT_FOUND.format(project_id=project_id)
                ) from e
            raise BricksApiError(
                ErrorMessage.BRICKS_API_ERROR.format(code=e.code)
            ) from e
        except urllib.error.URLError as e:
            raise BricksConnectionError(
                ErrorMessage.BRICKS_CONNECTION_FAILED.format(reason=e.reason)
            ) from e
        except TimeoutError as e:
            raise BricksTimeoutError(
                ErrorMessage.BRICKS_REQUEST_TIMED_OUT.format(error=e)
            ) from e
        items = json.loads(body).get("items", [])
        logger.info(LogMessage.BRICKS_FOUND_DOCUMENTS, len(items), project_id)
        documents = [BricksDocumentInfo(**item) for item in items]
        return documents

    async def download_document(
        self,
        document_id: str,
        project_id: str,
    ) -> tuple[bytes, str, str]:
        logger.info(
            LogMessage.BRICKS_DOWNLOADING_DOCUMENT, document_id, project_id
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
            raise BricksNotFoundError(
                ErrorMessage.BRICKS_DOCUMENT_NOT_FOUND.format(
                    document_id=document_id, project_id=project_id
                )
            )
        try:
            body, resp_headers = await asyncio.to_thread(self._get, url)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise BricksPermissionError(
                    ErrorMessage.DOCUMENT_DOWNLOAD_AUTH_FAILED.format(code=e.code)
                ) from e
            if e.code == 404:
                raise BricksNotFoundError(
                    ErrorMessage.DOCUMENT_NOT_FOUND.format(
                        document_id=document_id, project_id=project_id
                    )
                ) from e
            raise BricksApiError(
                ErrorMessage.DOCUMENT_DOWNLOAD_FAILED.format(code=e.code)
            ) from e
        except urllib.error.URLError as e:
            raise BricksConnectionError(
                ErrorMessage.DOCUMENT_DOWNLOAD_CONNECTION_FAILED.format(
                    reason=e.reason
                )
            ) from e
        except TimeoutError as e:
            raise BricksTimeoutError(
                ErrorMessage.DOCUMENT_DOWNLOAD_TIMED_OUT.format(error=e)
            ) from e

        filename = _extract_filename(resp_headers.get("Content-Disposition", ""), url)
        logger.info(
            LogMessage.BRICKS_DOWNLOADED_DOCUMENT,
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
            LogMessage.BRICKS_PUBLISHING_VERSION,
            payload.get("projectUniqueId"),
            payload.get("sectionKey"),
            payload.get("workflowId"),
        )
        logger.info(
            LogMessage.BRICKS_PUBLISH_PAYLOAD, json.dumps(payload, ensure_ascii=False, default=str)
        )
        try:
            body = await asyncio.to_thread(self._post, url, payload, headers)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise BricksPermissionError(
                    ErrorMessage.PUBLISH_AUTH_FAILED.format(code=e.code)
                ) from e
            raise BricksApiError(
                ErrorMessage.PUBLISH_FAILED.format(code=e.code)
            ) from e
        except urllib.error.URLError as e:
            raise BricksConnectionError(
                ErrorMessage.PUBLISH_CONNECTION_FAILED.format(reason=e.reason)
            ) from e
        except TimeoutError as e:
            raise BricksTimeoutError(
                ErrorMessage.PUBLISH_TIMED_OUT.format(error=e)
            ) from e
        data = json.loads(body)
        logger.info(LogMessage.BRICKS_PUBLISHED_VERSION, data)
        return SectionVersionResult(success=True, message="Published", data=data)


def _extract_filename(content_disposition: str, url: str = "") -> str:
    match = re.search(r'filename="([^"]+)"', content_disposition)
    if match:
        filename = match.group(1)
        logger.debug(LogMessage.BRICKS_FILENAME_QUOTED, filename)
        return _normalize_extension(filename)
    match = re.search(r"filename=([^\s;]+)", content_disposition)
    if match:
        filename = match.group(1)
        logger.debug(LogMessage.BRICKS_FILENAME_UNQUOTED, filename)
        return _normalize_extension(filename)
    if url:
        decoded_path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        path_filename = decoded_path.rsplit("/", 1)[-1]
        if path_filename and "." in path_filename:
            logger.debug(
                LogMessage.BRICKS_FILENAME_FROM_URL, path_filename, url[:200]
            )
            return _normalize_extension(path_filename)
    logger.warning(
        LogMessage.BRICKS_FILENAME_FALLBACK,
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
        LogMessage.BRICKS_NORMALIZED_EXTENSION, ext, cleaned, filename
    )
    return name + cleaned
