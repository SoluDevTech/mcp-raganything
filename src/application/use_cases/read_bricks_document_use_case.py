import contextlib
import os
import tempfile

import aiofiles

from domain.ports.bricks_api_port import BricksApiPort
from domain.ports.document_reader_port import DocumentContent, DocumentReaderPort


class ReadBricksDocumentUseCase:
    def __init__(
        self,
        bricks_api: BricksApiPort,
        document_reader: DocumentReaderPort,
        output_dir: str,
    ) -> None:
        self.bricks_api = bricks_api
        self.document_reader = document_reader
        self.output_dir = output_dir

    async def execute(
        self,
        document_id: str,
        project_id: str,
    ) -> DocumentContent:
        data, filename, mime_type = await self.bricks_api.download_document(
            document_id=document_id, project_id=project_id
        )

        os.makedirs(self.output_dir, exist_ok=True)
        suffix = os.path.splitext(filename)[1] or ".bin"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=self.output_dir)
        try:
            os.close(fd)
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(data)

            return await self.document_reader.extract_content(tmp_path, mime_type=mime_type)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
