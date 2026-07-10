import os

from pydantic import BaseModel, Field, field_validator

from domain.errors.messages import ErrorMessage


class ReadFileRequest(BaseModel):
    file_path: str = Field(..., description="File path in MinIO bucket")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(str(ErrorMessage.FILE_PATH_EMPTY))
        normalized = os.path.normpath(v).replace("\\", "/")
        if normalized in (".", "..") or normalized.startswith("../"):
            raise ValueError(str(ErrorMessage.FILE_PATH_EMPTY))
        if os.path.isabs(normalized):
            raise ValueError("file_path must be a relative path within the bucket")
        return normalized
