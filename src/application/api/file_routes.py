import posixpath
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from application.requests.file_request import ReadFileRequest
from application.responses.file_response import FileContentResponse, FileInfoResponse
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from dependencies import (
    get_list_files_use_case,
    get_list_folders_use_case,
    get_read_file_use_case,
    get_upload_file_use_case,
)
from domain.errors.file import FileTooLargeError, FileValidationError
from domain.errors.messages import ErrorMessage

file_router = APIRouter(tags=["Files"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".xlsx",
    ".pptx",
    ".md",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".html",
    ".xml",
    ".json",
    ".rtf",
    ".odt",
    ".ods",
}

ALLOWED_MIME_PREFIXES = (
    "application/pdf",
    "text/",
    "image/",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.ms-",
    "application/json",
    "application/rtf",
    "application/vnd.oasis.opendocument",
)


def _sanitize_filename(filename: str | None) -> str:
    if not filename:
        raise FileValidationError(ErrorMessage.FILENAME_REQUIRED)
    clean = posixpath.basename(filename.replace("\\", "/"))
    if not clean or clean.startswith("."):
        raise FileValidationError(ErrorMessage.INVALID_FILENAME)
    return clean


def _validate_file_type(filename: str, content_type: str) -> None:
    ext = posixpath.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(ErrorMessage.FILE_TYPE_NOT_ALLOWED.format(ext=ext))
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise FileValidationError(
            ErrorMessage.CONTENT_TYPE_NOT_ALLOWED.format(content_type=content_type)
        )


def _validate_prefix(prefix: str) -> str:
    normalized = posixpath.normpath(prefix.replace("\\", "/"))
    if normalized == ".":
        normalized = ""
    if normalized.startswith("..") or posixpath.isabs(normalized):
        raise FileValidationError(ErrorMessage.PREFIX_MUST_BE_RELATIVE)
    if prefix.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


@file_router.get(
    "/files/list",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Validation error — prefix is absolute or traverses parent.",
        },
    },
)
async def list_files(
    use_case: Annotated[ListFilesUseCase, Depends(get_list_files_use_case)],
    prefix: str = "",
    recursive: bool = True,
) -> list[FileInfoResponse]:
    prefix = _validate_prefix(prefix)
    files = await use_case.execute(prefix=prefix, recursive=recursive)
    return [FileInfoResponse(**asdict(f)) for f in files]


@file_router.get(
    "/files/folders",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Validation error — prefix is absolute or traverses parent.",
        },
    },
)
async def list_folders(
    use_case: Annotated[ListFoldersUseCase, Depends(get_list_folders_use_case)],
    prefix: str = "",
) -> list[str]:
    prefix = _validate_prefix(prefix)
    return await use_case.execute(prefix=prefix)


@file_router.post(
    "/files/read",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Validation error — file_path is missing or invalid.",
        },
    },
)
async def read_file(
    request: ReadFileRequest,
    use_case: Annotated[ReadFileUseCase, Depends(get_read_file_use_case)],
) -> FileContentResponse:
    result = await use_case.execute(file_path=request.file_path)
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )


@file_router.post(
    "/files/upload",
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "description": "Validation error — invalid prefix, filename, or file type.",
        },
    },
)
async def upload_file(
    use_case: Annotated[UploadFileUseCase, Depends(get_upload_file_use_case)],
    file: Annotated[UploadFile, File(description="The file to upload")],
    prefix: Annotated[str, Form(description="Optional prefix/folder")] = "",
):
    normalized = posixpath.normpath(prefix.replace("\\", "/"))
    if normalized == ".":
        normalized = ""
    if normalized.startswith("..") or posixpath.isabs(normalized):
        raise FileValidationError(ErrorMessage.PREFIX_MUST_BE_RELATIVE_SHORT)
    if prefix.endswith("/") and not normalized.endswith("/"):
        normalized += "/"

    safe_name = _sanitize_filename(file.filename)
    content_type = file.content_type or "application/octet-stream"
    _validate_file_type(safe_name, content_type)

    file_data = await file.read()
    if len(file_data) > MAX_UPLOAD_SIZE:
        raise FileTooLargeError(
            ErrorMessage.FILE_TOO_LARGE.format(max_mb=MAX_UPLOAD_SIZE // (1024 * 1024))
        )

    result = await use_case.execute(
        file_data=file_data,
        file_name=safe_name,
        prefix=normalized,
        content_type=content_type,
    )
    return {
        "object_name": result.object_name,
        "size": result.size,
        "message": "File uploaded successfully",
    }
