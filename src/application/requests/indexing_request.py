from pydantic import BaseModel, Field


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
    file_extensions: list[str] | None = Field(
        default=None, description="File extensions to filter"
    )
