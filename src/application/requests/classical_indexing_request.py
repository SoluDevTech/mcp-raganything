"""Request models for classical RAG indexing endpoints."""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_file_extensions(v: str | list[str] | None) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return [ext.strip() for ext in v.split(",") if ext.strip()]
    return v


class ClassicalIndexFileRequest(BaseModel):
    file_name: str = Field(..., description="Object path in the MinIO bucket")
    working_dir: str = Field(
        ..., description="RAG workspace directory (project isolation)"
    )
    chunk_size: int = Field(
        default=1000, description="Max chars per chunk", ge=100, le=10000
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks", ge=0, le=2000
    )


class ClassicalIndexFolderRequest(BaseModel):
    working_dir: str = Field(
        ..., description="RAG workspace directory, also used as MinIO prefix"
    )
    recursive: bool = Field(
        default=True, description="Process subdirectories recursively"
    )
    file_extensions: Annotated[
        list[str] | None, BeforeValidator(_coerce_file_extensions)
    ] = Field(default=None, description="Filter by extensions, e.g. ['.pdf', '.docx']")
    chunk_size: int = Field(
        default=1000, description="Max chars per chunk", ge=100, le=10000
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks", ge=0, le=2000
    )
