from domain.ports.storage_port import StoragePort


class ListFoldersUseCase:
    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(self, prefix: str = "") -> list[str]:
        return await self.storage.list_folders(self.bucket, prefix)
