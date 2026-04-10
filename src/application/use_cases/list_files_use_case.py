from domain.ports.storage_port import FileInfo, StoragePort


class ListFilesUseCase:
    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(self, prefix: str = "", recursive: bool = True) -> list[FileInfo]:
        return await self.storage.list_files_metadata(self.bucket, prefix, recursive)
