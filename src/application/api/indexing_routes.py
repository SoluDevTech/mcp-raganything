from fastapi import APIRouter, BackgroundTasks, Depends, status

from application.requests.indexing_request import IndexFileRequest, IndexFolderRequest
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from dependencies import get_index_file_use_case, get_index_folder_use_case

indexing_router = APIRouter(tags=["Multimodal Indexing"])


@indexing_router.post(
    "/file/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_file(
    request: IndexFileRequest,
    background_tasks: BackgroundTasks,
    use_case: IndexFileUseCase = Depends(get_index_file_use_case),
):
    background_tasks.add_task(
        use_case.execute,
        file_name=request.file_name,
        working_dir=request.working_dir,
    )
    return {"status": "accepted", "message": "File indexing started in background"}


@indexing_router.post(
    "/folder/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_folder(
    request: IndexFolderRequest,
    background_tasks: BackgroundTasks,
    use_case: IndexFolderUseCase = Depends(get_index_folder_use_case),
):
    background_tasks.add_task(use_case.execute, request=request)
    return {"status": "accepted", "message": "Folder indexing started in background"}
