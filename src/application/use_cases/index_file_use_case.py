import logging
import os

import aiofiles

from domain.entities.indexing_result import FileIndexingResult
from domain.ports.rag_engine import RAGEnginePort
from domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


class IndexFileUseCase:
    """Use case for indexing a single file downloaded from MinIO."""

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

    async def execute(self, file_name: str, working_dir: str) -> FileIndexingResult:
        os.makedirs(self.output_dir, exist_ok=True)

        data = await self.storage.get_object(self.bucket, file_name)
        file_path = os.path.join(self.output_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)

        self.rag_engine.init_project(working_dir)

        result = await self.rag_engine.index_document(
            file_path=file_path,
            file_name=file_name,
            output_dir=self.output_dir,
            working_dir=working_dir,
        )

        logger.info(f"Indexation finished: {result.model_dump()}")
        
        return result
