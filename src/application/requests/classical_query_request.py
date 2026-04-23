"""Request model for classical RAG query endpoint."""

from pydantic import BaseModel, Field


class ClassicalQueryRequest(BaseModel):
    working_dir: str = Field(
        ..., description="RAG workspace directory for this project"
    )
    query: str = Field(..., description="The search query")
    top_k: int = Field(
        default=10,
        description="Maximum number of chunks to retrieve per query variation",
        ge=1,
        le=100,
    )
    num_variations: int = Field(
        default=3,
        description="Number of query variations to generate (multi-query)",
        ge=1,
        le=10,
    )
    relevance_threshold: float = Field(
        default=5.0,
        description="Minimum relevance score (0-10) from LLM judge to include a chunk",
        ge=0.0,
        le=10.0,
    )
