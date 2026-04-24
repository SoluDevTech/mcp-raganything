"""Response models for classical RAG query endpoint."""

from pydantic import BaseModel, Field


class ClassicalChunkResponse(BaseModel):
    chunk_id: str = Field(description="Unique identifier for the chunk")
    content: str = Field(description="The text content of the chunk")
    file_path: str = Field(description="Source file path in MinIO")
    relevance_score: float = Field(description="LLM judge relevance score (0-10)")
    metadata: dict[str, str | int | float | None] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ClassicalQueryResponse(BaseModel):
    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Optional message")
    queries: list[str] = Field(default_factory=list, description="Query variations used")
    chunks: list[ClassicalChunkResponse] = Field(
        default_factory=list, description="Filtered and ranked chunks"
    )
