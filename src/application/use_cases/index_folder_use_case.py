import asyncio
import logging
import os

import aiofiles

from application.requests.indexing_request import IndexFolderRequest
from domain.entities.indexing_result import FolderIndexingResult
from domain.ports.rag_engine import RAGEnginePort
from domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


class IndexFolderUseCase:
    """Use case for indexing a folder of documents downloaded from MinIO."""

    def __init__(
        self,
        rag_engine: RAGEnginePort,
        storage: StoragePort,
        bucket: str,
        output_dir: str,
    ) -> None:
        self.rag_engine = rag_engine
        self.storage = storage
        self.bucket = bucket
        self.output_dir = output_dir

    async def execute(self, request: IndexFolderRequest) -> FolderIndexingResult:
        local_folder = os.path.join(self.output_dir, request.working_dir)
        
        os.makedirs(local_folder, exist_ok=True)

        files = await self.storage.list_objects(
            self.bucket, prefix=request.working_dir, recursive=request.recursive
        )

        if request.file_extensions:
            exts = set(request.file_extensions)
            files = [f for f in files if any(f.endswith(ext) for ext in exts)]

        semaphore = asyncio.Semaphore(10)

        async def _download(file_name: str) -> None:
            async with semaphore:
                data = await self.storage.get_object(self.bucket, file_name)
                local_name = os.path.basename(file_name)
                async with aiofiles.open(os.path.join(local_folder, local_name), "wb") as f:
                    await f.write(data)

        await asyncio.gather(*[_download(f) for f in files])

        self.rag_engine.init_project(request.working_dir)
        
        result = await self.rag_engine.index_folder(
            folder_path=local_folder,
            output_dir=self.output_dir,
            recursive=request.recursive,
            file_extensions=request.file_extensions,
            working_dir=request.working_dir,
        )

        logger.info(f"Folder indexation finished: {result.model_dump()}")
        return result
