import contextlib
import os

import aiofiles
from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file

from domain.services.classical_helpers import (
    build_documents_from_extraction,
    validate_path,
)
from domain.entities.indexing_result import FileIndexingResult, IndexingStatus
from domain.ports.storage_port import StoragePort
from domain.ports.vector_store_port import VectorStorePort


class ClassicalIndexFileUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        storage: StoragePort,
        bucket: str,
        output_dir: str,
    ) -> None:
        self.vector_store = vector_store
        self.storage = storage
        self.bucket = bucket
        self.output_dir = output_dir

    async def execute(
        self,
        file_name: str,
        working_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FileIndexingResult:
        file_path = None
        try:
            data = await self.storage.get_object(self.bucket, file_name)

            file_path = validate_path(self.output_dir, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            await self.vector_store.ensure_table(working_dir)

            config = ExtractionConfig(
                chunking=ChunkingConfig(max_chars=chunk_size, max_overlap=chunk_overlap)
            )
            result = await extract_file(file_path, config=config)

            documents = build_documents_from_extraction(result, file_name)

            if documents:
                await self.vector_store.add_documents(
                    working_dir=working_dir, documents=documents
                )

            return FileIndexingResult(
                status=IndexingStatus.SUCCESS,
                message="File indexed successfully",
                file_path=file_path,
                file_name=file_name,
            )
        except Exception as e:
            return FileIndexingResult(
                status=IndexingStatus.FAILED,
                message="File indexing failed",
                file_path=file_path or file_name,
                file_name=file_name,
                error=str(e),
            )
        finally:
            if file_path and os.path.exists(file_path):
                with contextlib.suppress(OSError):
                    os.remove(file_path)
