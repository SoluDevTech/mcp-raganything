from domain.ports.storage_port import StoragePort


class CreateFolderUseCase:
    """Create a folder (0-byte object with trailing slash) in the storage bucket."""

    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(self, prefix: str) -> str:
        """Create a folder marker under the given prefix.

        Args:
            prefix: Folder prefix (auto-appended with ``/`` if missing).

        Returns:
            The normalized prefix ending with a trailing slash.
        """
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        await self.storage.put_object(
            self.bucket, prefix, b"", "application/octet-stream"
        )
        return prefix
