import contextlib
import os
import tempfile

import aiofiles

from domain.ports.document_reader_port import DocumentContent, DocumentReaderPort
from domain.ports.storage_port import StoragePort


class ReadFileUseCase:
    def __init__(
        self,
        storage: StoragePort,
        document_reader: DocumentReaderPort,
        bucket: str,
        output_dir: str,
    ) -> None:
        self.storage = storage
        self.document_reader = document_reader
        self.bucket = bucket
        self.output_dir = output_dir

    async def execute(self, file_path: str) -> DocumentContent:
        data = await self.storage.get_object(self.bucket, file_path)

        os.makedirs(self.output_dir, exist_ok=True)
        suffix = os.path.splitext(file_path)[1] or ".bin"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=self.output_dir)
        try:
            os.close(fd)
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(data)

            return await self.document_reader.extract_content(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
