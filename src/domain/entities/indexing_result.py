from enum import Enum

from pydantic import BaseModel, Field

ERROR_MESSAGE_IF_FAILED = "Error message if failed"


class IndexingStatus(str, Enum):
    """Status of an indexing operation."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class FileIndexingResult(BaseModel):
    """Result of indexing a single file."""

    status: IndexingStatus = Field(description="Indexing status")
    message: str = Field(description="Status message")
    file_path: str = Field(description="Path to the indexed file")
    file_name: str = Field(description="Name of the indexed file")
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )
    error: str | None = Field(default=None, description=ERROR_MESSAGE_IF_FAILED)


class FileProcessingDetail(BaseModel):
    """Details of a processed file in folder indexing."""

    file_path: str = Field(description="Path to the file")
    file_name: str = Field(description="Name of the file")
    status: IndexingStatus = Field(description="Processing status")
    error: str | None = Field(default=None, description=ERROR_MESSAGE_IF_FAILED)


class FolderIndexingStats(BaseModel):
    """Statistics for folder indexing operation."""

    total_files: int = Field(default=0, description="Total files found")
    files_processed: int = Field(default=0, description="Files successfully processed")
    files_failed: int = Field(default=0, description="Files that failed processing")
    files_skipped: int = Field(
        default=0, description="Files skipped (already indexed or filtered)"
    )


class FolderIndexingResult(BaseModel):
    """Result of indexing a folder of documents."""

    status: IndexingStatus = Field(description="Overall indexing status")
    message: str = Field(description="Status message")
    folder_path: str = Field(description="Path to the indexed folder")
    recursive: bool = Field(description="Whether subdirectories were processed")
    stats: FolderIndexingStats = Field(
        default_factory=FolderIndexingStats, description="Processing statistics"
    )
    processing_time_ms: float | None = Field(
        default=None, description="Total processing time in milliseconds"
    )
    file_results: list[FileProcessingDetail] | None = Field(
        default=None, description="Individual file results"
    )
    error: str | None = Field(default=None, description="Error message if failed")
