from domain.ports.storage_port import FileInfo, StoragePort


class UploadFileUseCase:
    """Upload a file to the object storage bucket."""

    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(
        self, file_data: bytes, file_name: str, prefix: str, content_type: str
    ) -> FileInfo:
        """Store a file in the bucket under the given prefix.

        Args:
            file_data: Raw file bytes.
            file_name: Destination file name.
            prefix: Folder prefix (auto-appended with ``/`` if missing).
            content_type: MIME type for the stored object.

        Returns:
            FileInfo with the final object path and size.
        """
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        object_path = prefix + file_name
        await self.storage.put_object(self.bucket, object_path, file_data, content_type)
        return FileInfo(object_name=object_path, size=len(file_data))
