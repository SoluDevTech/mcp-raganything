from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_file_extensions(v: str | list[str] | None) -> list[str] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return [v]
    return v


class IndexFileRequest(BaseModel):
    file_name: str = Field(..., description="File name/path in MinIO bucket")
    working_dir: str = Field(
        ..., description="RAG workspace directory for this project"
    )


class IndexFolderRequest(BaseModel):
    working_dir: str = Field(
        ...,
        description="RAG workspace directory — also used as MinIO prefix",
    )
    recursive: bool = Field(
        default=True, description="Process subdirectories recursively"
    )
    file_extensions: Annotated[
        list[str] | None, BeforeValidator(_coerce_file_extensions)
    ] = Field(default=None, description="File extensions to filter")
