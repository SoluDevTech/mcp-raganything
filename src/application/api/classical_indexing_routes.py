import asyncio
import logging

from fastapi import APIRouter, Depends, status

from application.requests.classical_indexing_request import (
    ClassicalIndexFileRequest,
    ClassicalIndexFolderRequest,
)
from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from dependencies import (
    get_classical_index_file_use_case,
    get_classical_index_folder_use_case,
)

logger = logging.getLogger(__name__)

classical_indexing_router = APIRouter(tags=["Classical Indexing"])

_background_tasks: set[asyncio.Task] = set()


async def _run_in_background(coro, label: str) -> None:
    try:
        await coro
    except Exception:
        logger.exception("Background %s failed", label)


@classical_indexing_router.post(
    "/classical/file/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def classical_index_file(
    request: ClassicalIndexFileRequest,
    use_case: ClassicalIndexFileUseCase = Depends(get_classical_index_file_use_case),
):
    task = asyncio.create_task(
        _run_in_background(
            use_case.execute(
                file_name=request.file_name,
                working_dir=request.working_dir,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            ),
            label=f"classical file indexing {request.file_name}",
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "accepted", "message": "File indexing started in background"}


@classical_indexing_router.post(
    "/classical/folder/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def classical_index_folder(
    request: ClassicalIndexFolderRequest,
    use_case: ClassicalIndexFolderUseCase = Depends(
        get_classical_index_folder_use_case
    ),
):
    task = asyncio.create_task(
        _run_in_background(
            use_case.execute(
                working_dir=request.working_dir,
                recursive=request.recursive,
                file_extensions=request.file_extensions,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            ),
            label=f"classical folder indexing {request.working_dir}",
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "accepted", "message": "Folder indexing started in background"}
