import os

from pydantic import BaseModel, Field, field_validator


class ReadFileRequest(BaseModel):
    file_path: str = Field(..., description="File path in MinIO bucket")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        normalized = os.path.normpath(v).replace("\\", "/")
        if normalized.startswith("..") or os.path.isabs(normalized):
            raise ValueError("file_path must be a relative path within the bucket")
        return normalized
