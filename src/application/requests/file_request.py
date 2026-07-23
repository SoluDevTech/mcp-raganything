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


class CreateFolderRequest(BaseModel):
    """Request DTO for creating a folder in the storage bucket."""

    prefix: str = Field(..., description="Folder prefix to create")

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(str(ErrorMessage.PREFIX_MUST_BE_RELATIVE))
        normalized = os.path.normpath(v).replace("\\", "/")
        if (
            normalized in (".", "..")
            or normalized.startswith("../")
            or "/.." in normalized
        ):
            raise ValueError(str(ErrorMessage.PREFIX_MUST_BE_RELATIVE))
        if os.path.isabs(normalized):
            raise ValueError(str(ErrorMessage.PREFIX_MUST_BE_RELATIVE))
        if v.endswith("/") and not normalized.endswith("/"):
            normalized += "/"
        return normalized
