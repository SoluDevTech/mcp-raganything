import posixpath
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from application.requests.file_request import ReadFileRequest
from application.responses.file_response import FileContentResponse, FileInfoResponse
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from dependencies import (
    get_list_files_use_case,
    get_read_file_use_case,
    get_upload_file_use_case,
)

file_router = APIRouter(tags=["Files"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".docx", ".xlsx", ".pptx", ".md", ".csv",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp",
    ".html", ".xml", ".json", ".rtf", ".odt", ".ods",
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
        raise HTTPException(status_code=422, detail="Filename is required")
    clean = posixpath.basename(filename.replace("\\", "/"))
    if not clean or clean.startswith("."):
        raise HTTPException(status_code=422, detail="Invalid filename")
    return clean


def _validate_file_type(filename: str, content_type: str) -> None:
    ext = posixpath.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422, detail=f"File type '{ext}' is not allowed"
        )
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            status_code=422, detail=f"Content type '{content_type}' is not allowed"
        )


@file_router.get(
    "/files/list",
    response_model=list[FileInfoResponse],
    status_code=status.HTTP_200_OK,
)
async def list_files(
    prefix: str = "",
    recursive: bool = True,
    use_case: ListFilesUseCase = Depends(get_list_files_use_case),
) -> list[FileInfoResponse]:
    files = await use_case.execute(prefix=prefix, recursive=recursive)
    return [FileInfoResponse(**asdict(f)) for f in files]


@file_router.post(
    "/files/read",
    response_model=FileContentResponse,
    status_code=status.HTTP_200_OK,
)
async def read_file(
    request: ReadFileRequest,
    use_case: ReadFileUseCase = Depends(get_read_file_use_case),
) -> FileContentResponse:
    try:
        result = await use_case.execute(file_path=request.file_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from None
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )


@file_router.post(
    "/files/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
    prefix: str = Form(default=""),
    use_case: UploadFileUseCase = Depends(get_upload_file_use_case),
):
    normalized = posixpath.normpath(prefix.replace("\\", "/"))
    if normalized == ".":
        normalized = ""
    if normalized.startswith("..") or posixpath.isabs(normalized):
        raise HTTPException(
            status_code=422, detail="prefix must be a relative path"
        )
    if prefix.endswith("/") and not normalized.endswith("/"):
        normalized += "/"

    safe_name = _sanitize_filename(file.filename)
    content_type = file.content_type or "application/octet-stream"
    _validate_file_type(safe_name, content_type)

    file_data = await file.read()
    if len(file_data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
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
