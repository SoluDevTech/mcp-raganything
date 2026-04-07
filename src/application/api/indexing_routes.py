import asyncio
import logging

from fastapi import APIRouter, Depends, status

from application.requests.indexing_request import IndexFileRequest, IndexFolderRequest
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from dependencies import get_index_file_use_case, get_index_folder_use_case

logger = logging.getLogger(__name__)

indexing_router = APIRouter(tags=["Multimodal Indexing"])

_background_tasks: set[asyncio.Task] = set()


async def _run_in_background(coro, label: str) -> None:
    try:
        await coro
    except Exception:
        logger.exception("Background %s failed", label)


@indexing_router.post(
    "/file/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_file(
    request: IndexFileRequest,
    use_case: IndexFileUseCase = Depends(get_index_file_use_case),
):
    task = asyncio.create_task(
        _run_in_background(
            use_case.execute(
                file_name=request.file_name, working_dir=request.working_dir
            ),
            label=f"file indexing {request.file_name}",
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "accepted", "message": "File indexing started in background"}


@indexing_router.post(
    "/folder/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_folder(
    request: IndexFolderRequest,
    use_case: IndexFolderUseCase = Depends(get_index_folder_use_case),
):
    task = asyncio.create_task(
        _run_in_background(
            use_case.execute(request=request),
            label=f"folder indexing {request.working_dir}",
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "accepted", "message": "Folder indexing started in background"}
