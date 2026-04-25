import contextlib
import os
from pathlib import Path

import aiofiles
from kreuzberg import extract_file

from domain.entities.indexing_result import (
    FileProcessingDetail,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)
from domain.ports.storage_port import StoragePort
from domain.ports.vector_store_port import VectorStorePort
from domain.services.classical_helpers import (
    build_documents_from_extraction,
    validate_path,
)
from infrastructure.document_reader.kreuzberg_adapter import make_extraction_config


class ClassicalIndexFolderUseCase:
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
        working_dir: str,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FolderIndexingResult:
        await self.vector_store.ensure_table(working_dir)

        prefix = working_dir if working_dir.endswith("/") else f"{working_dir}/"

        files = await self.storage.list_objects(
            self.bucket, prefix=prefix, recursive=recursive
        )

        if file_extensions:
            exts = set(file_extensions)
            files = [f for f in files if Path(f).suffix in exts]

        processed = 0
        failed = 0
        file_results = []
        config = make_extraction_config(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for file_name in files:
            local_path = None
            try:
                data = await self.storage.get_object(self.bucket, file_name)

                local_path = validate_path(self.output_dir, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                async with aiofiles.open(local_path, "wb") as f:
                    await f.write(data)

                result = await extract_file(local_path, config=config)

                documents = build_documents_from_extraction(result, file_name)

                if documents:
                    await self.vector_store.add_documents(
                        working_dir=working_dir, documents=documents
                    )

                processed += 1
                file_results.append(
                    FileProcessingDetail(
                        file_path=file_name,
                        file_name=os.path.basename(file_name),
                        status=IndexingStatus.SUCCESS,
                    )
                )
            except Exception as exc:
                failed += 1
                file_results.append(
                    FileProcessingDetail(
                        file_path=file_name,
                        file_name=os.path.basename(file_name),
                        status=IndexingStatus.FAILED,
                        error=str(exc),
                    )
                )
            finally:
                if local_path and os.path.exists(local_path):
                    with contextlib.suppress(OSError):
                        os.remove(local_path)

        if failed > 0 and processed > 0:
            status = IndexingStatus.PARTIAL
        elif failed > 0:
            status = IndexingStatus.FAILED
        else:
            status = IndexingStatus.SUCCESS

        return FolderIndexingResult(
            status=status,
            message="Folder indexing completed",
            folder_path=working_dir,
            recursive=recursive,
            stats=FolderIndexingStats(
                total_files=len(files),
                files_processed=processed,
                files_failed=failed,
            ),
            file_results=file_results,
        )
