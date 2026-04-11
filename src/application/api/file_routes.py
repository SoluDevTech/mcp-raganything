from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from application.requests.file_request import ReadFileRequest
from application.responses.file_response import FileContentResponse, FileInfoResponse
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from dependencies import (
    get_list_files_use_case,
    get_list_folders_use_case,
    get_read_file_use_case,
)

file_router = APIRouter(tags=["Files"])


@file_router.get(
    "/files/list",
    status_code=status.HTTP_200_OK,
)
async def list_files(
    prefix: str = "",
    recursive: bool = True,
    use_case: ListFilesUseCase = Depends(get_list_files_use_case),
) -> list[FileInfoResponse]:
    files = await use_case.execute(prefix=prefix, recursive=recursive)
    return [FileInfoResponse(**asdict(f)) for f in files]


@file_router.get(
    "/files/folders",
    status_code=status.HTTP_200_OK,
)
async def list_folders(
    prefix: str = "",
    use_case: ListFoldersUseCase = Depends(get_list_folders_use_case),
) -> list[str]:
    try:
        return await use_case.execute(prefix=prefix)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None


@file_router.post(
    "/files/read",
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
