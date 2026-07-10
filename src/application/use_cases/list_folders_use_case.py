from domain.ports.storage_port import StoragePort


class ListFoldersUseCase:
    """List virtual folder prefixes in the object storage bucket."""

    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(self, prefix: str = "") -> list[str]:
        """List folder names under an optional prefix.

        Args:
            prefix: Object key prefix to scope the listing (empty = root).

        Returns:
            List of folder prefix strings.
        """
        return await self.storage.list_folders(self.bucket, prefix)
