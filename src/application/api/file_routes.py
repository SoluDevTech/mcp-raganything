import logging
import os
import posixpath
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from application.requests.file_request import (
    CreateFolderRequest,
    ReadFileRequest,
)
from application.responses.file_response import FileContentResponse, FileInfoResponse
from application.use_cases.create_folder_use_case import CreateFolderUseCase
from application.use_cases.delete_file_use_case import DeleteFileUseCase
from application.use_cases.delete_folder_use_case import DeleteFolderUseCase
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from dependencies import (
    get_create_folder_use_case,
    get_delete_file_use_case,
    get_delete_folder_use_case,
    get_list_files_use_case,
    get_list_folders_use_case,
    get_read_file_use_case,
    get_upload_file_use_case,
)
from domain.constants import DEFAULT_WORKING_DIR
from domain.errors.file import FileTooLargeError, FileValidationError
from domain.errors.messages import ErrorMessage

logger = logging.getLogger(__name__)

file_router = APIRouter(tags=["Files"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
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


def _validate_object_path(object_name: str) -> str:
    if not object_name or not object_name.strip():
        raise FileValidationError(ErrorMessage.FILE_PATH_EMPTY)
    normalized = os.path.normpath(object_name).replace("\\", "/")
    if normalized in (".", "..") or normalized.startswith("../") or "/.." in normalized:
        raise FileValidationError(ErrorMessage.FILE_PATH_EMPTY)
    if os.path.isabs(normalized):
        raise FileValidationError(ErrorMessage.FILE_PATH_EMPTY)
    return normalized


def _validate_delete_prefix(prefix: str) -> str:
    if not prefix or not prefix.strip():
        raise FileValidationError(ErrorMessage.PREFIX_MUST_BE_RELATIVE)
    normalized = os.path.normpath(prefix).replace("\\", "/")
    if normalized in (".", "..") or normalized.startswith("../") or "/.." in normalized:
        raise FileValidationError(ErrorMessage.PREFIX_MUST_BE_RELATIVE)
    if os.path.isabs(normalized):
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


@file_router.post(
    "/files/folders",
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "description": "Validation error — prefix is missing, absolute or traverses parent.",
        },
    },
)
async def create_folder(
    request: CreateFolderRequest,
    use_case: Annotated[CreateFolderUseCase, Depends(get_create_folder_use_case)],
):
    result = await use_case.execute(prefix=request.prefix)
    return {"message": "Folder created", "prefix": result}


@file_router.delete(
    "/files",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Validation error — object_name is missing or invalid.",
        },
    },
)
async def delete_file(
    use_case: Annotated[DeleteFileUseCase, Depends(get_delete_file_use_case)],
    object_name: str = "",
    working_dir: str = "",
):
    object_name = _validate_object_path(object_name)
    if working_dir and working_dir.strip():
        working_dir = _validate_prefix(working_dir)
    else:
        logger.warning(
            "working_dir not provided for delete_file, falling back to %s",
            DEFAULT_WORKING_DIR,
        )
        working_dir = DEFAULT_WORKING_DIR
    result = await use_case.execute(object_path=object_name, working_dir=working_dir)
    return {"message": "File deleted", "object_name": result}


@file_router.delete(
    "/files/folders",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Validation error — prefix is missing, absolute or traverses parent.",
        },
    },
)
async def delete_folder(
    use_case: Annotated[DeleteFolderUseCase, Depends(get_delete_folder_use_case)],
    prefix: str = "",
):
    prefix = _validate_delete_prefix(prefix)
    result = await use_case.execute(prefix=prefix)
    return {"message": "Folder deleted", "prefix": result}
