from domain.ports.storage_port import FileInfo, StoragePort


class ListFilesUseCase:
    """List files stored in the object storage bucket."""

    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(self, prefix: str = "", recursive: bool = True) -> list[FileInfo]:
        """List file metadata under an optional prefix.

        Args:
            prefix: Object key prefix to filter by (empty = entire bucket).
            recursive: Whether to include objects in sub-folders.

        Returns:
            List of FileInfo metadata objects.
        """
        return await self.storage.list_files_metadata(self.bucket, prefix, recursive)
