from domain.ports.storage_port import FileInfo, StoragePort


class UploadFileUseCase:
    def __init__(self, storage: StoragePort, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket

    async def execute(
        self, file_data: bytes, file_name: str, prefix: str, content_type: str
    ) -> FileInfo:
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        object_path = prefix + file_name
        await self.storage.put_object(self.bucket, object_path, file_data, content_type)
        return FileInfo(object_name=object_path, size=len(file_data))
