import json
import logging
import os
import re
import urllib.parse

import httpx

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
    _client: httpx.AsyncClient

    def __init__(self, config) -> None:
        self._base_url = config.BRICKS_API_BASE_URL.rstrip("/")
        self._api_key = config.BRICKS_API_KEY
        self._bearer_token = config.BRICKS_BEARER_TOKEN
        self._http_timeout = config.BRICKS_HTTP_TIMEOUT
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._http_timeout),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _raise_for_status_error(
        self,
        error: httpx.HTTPStatusError,
        auth_msg: str,
        not_found_msg: str | None,
        api_error_msg: str,
    ) -> None:
        code = error.response.status_code
        if code in (401, 403):
            raise BricksPermissionError(auth_msg.format(code=code)) from error
        if code == 404 and not_found_msg is not None:
            raise BricksNotFoundError(not_found_msg) from error
        raise BricksApiError(api_error_msg.format(code=code)) from error

    def _raise_for_connection_error(self, error: httpx.ConnectError, msg: str) -> None:
        raise BricksConnectionError(msg.format(reason=str(error))) from error

    def _raise_for_timeout(self, error: httpx.TimeoutException, msg: str) -> None:
        raise BricksTimeoutError(msg.format(error=error)) from error

    async def _get(self, url: str, headers: dict | None = None) -> httpx.Response:
        logger.debug(LogMessage.BRICKS_GET, url)
        try:
            response = await self._client.get(url, headers=headers or {})
            response.raise_for_status()
            logger.debug(
                LogMessage.BRICKS_GET_BYTES,
                url,
                len(response.content),
                response.status_code,
            )
            return response
        except httpx.HTTPStatusError:
            logger.error(
                LogMessage.BRICKS_GET_HTTP_ERROR,
                url,
                response.status_code,
                response.text[:500],
            )
            raise
        except httpx.RequestError as e:
            logger.error(LogMessage.BRICKS_GET_ERROR, url, e)
            raise

    async def _post(self, url: str, payload: dict, headers: dict) -> httpx.Response:
        logger.debug(LogMessage.BRICKS_POST, url)
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.debug(
                LogMessage.BRICKS_POST_BYTES,
                url,
                len(response.content),
                response.status_code,
            )
            return response
        except httpx.HTTPStatusError:
            logger.error(
                LogMessage.BRICKS_POST_HTTP_ERROR,
                url,
                response.status_code,
                response.text[:500],
            )
            raise
        except httpx.RequestError as e:
            logger.error(LogMessage.BRICKS_POST_ERROR, url, e)
            raise

    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        url = f"{self._base_url}/api/projects/{project_id}/documents/ai"
        logger.info(LogMessage.BRICKS_LISTING_DOCUMENTS, project_id)
        try:
            response = await self._get(url, {"X-API-Key": f"{self._bearer_token}"})
        except httpx.HTTPStatusError as e:
            self._raise_for_status_error(
                e,
                auth_msg=ErrorMessage.BRICKS_AUTH_FAILED,
                not_found_msg=ErrorMessage.BRICKS_PROJECT_NOT_FOUND.format(
                    project_id=project_id
                ),
                api_error_msg=ErrorMessage.BRICKS_API_ERROR,
            )
        except httpx.ConnectError as e:
            self._raise_for_connection_error(e, ErrorMessage.BRICKS_CONNECTION_FAILED)
        except httpx.TimeoutException as e:
            self._raise_for_timeout(e, ErrorMessage.BRICKS_REQUEST_TIMED_OUT)
        items = json.loads(response.content).get("items", [])
        logger.info(LogMessage.BRICKS_FOUND_DOCUMENTS, len(items), project_id)
        return [BricksDocumentInfo(**item) for item in items]

    async def download_document(
        self,
        document_id: str,
        project_id: str,
    ) -> tuple[bytes, str, str]:
        logger.info(LogMessage.BRICKS_DOWNLOADING_DOCUMENT, document_id, project_id)
        documents = await self.list_project_documents(project_id)
        url, mime_type = self._find_document_url(documents, document_id)
        if not url:
            raise BricksNotFoundError(
                ErrorMessage.BRICKS_DOCUMENT_NOT_FOUND.format(
                    document_id=document_id, project_id=project_id
                )
            )
        try:
            response = await self._get(url)
        except httpx.HTTPStatusError as e:
            self._raise_for_status_error(
                e,
                auth_msg=ErrorMessage.DOCUMENT_DOWNLOAD_AUTH_FAILED,
                not_found_msg=ErrorMessage.DOCUMENT_NOT_FOUND.format(
                    document_id=document_id, project_id=project_id
                ),
                api_error_msg=ErrorMessage.DOCUMENT_DOWNLOAD_FAILED,
            )
        except httpx.ConnectError as e:
            self._raise_for_connection_error(
                e, ErrorMessage.DOCUMENT_DOWNLOAD_CONNECTION_FAILED
            )
        except httpx.TimeoutException as e:
            self._raise_for_timeout(e, ErrorMessage.DOCUMENT_DOWNLOAD_TIMED_OUT)

        body = response.content
        resp_headers = dict(response.headers)
        filename = _extract_filename(resp_headers.get("Content-Disposition", ""), url)
        logger.info(
            LogMessage.BRICKS_DOWNLOADED_DOCUMENT,
            document_id,
            len(body),
            mime_type,
            filename,
        )
        return body, filename, mime_type

    @staticmethod
    def _find_document_url(
        documents: list[BricksDocumentInfo], document_id: str
    ) -> tuple[str | None, str]:
        for doc in documents:
            if doc.id == document_id and doc.url:
                return doc.url, doc.mime_type
        return None, ""

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
            LogMessage.BRICKS_PUBLISH_PAYLOAD,
            json.dumps(payload, ensure_ascii=False, default=str),
        )
        try:
            response = await self._post(url, payload, headers)
        except httpx.HTTPStatusError as e:
            self._raise_for_status_error(
                e,
                auth_msg=ErrorMessage.PUBLISH_AUTH_FAILED,
                not_found_msg=None,
                api_error_msg=ErrorMessage.PUBLISH_FAILED,
            )
        except httpx.ConnectError as e:
            self._raise_for_connection_error(e, ErrorMessage.PUBLISH_CONNECTION_FAILED)
        except httpx.TimeoutException as e:
            self._raise_for_timeout(e, ErrorMessage.PUBLISH_TIMED_OUT)
        data = json.loads(response.content)
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
            logger.debug(LogMessage.BRICKS_FILENAME_FROM_URL, path_filename, url[:200])
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
    logger.warning(LogMessage.BRICKS_NORMALIZED_EXTENSION, ext, cleaned, filename)
    return name + cleaned
