"""Request model for classical RAG query endpoint."""

from typing import Literal

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
    vector_distance_threshold: float | None = Field(
        default=None,
        description="Maximum cosine distance for vector store filtering. Lower = more similar. None disables filtering.",
        ge=0.0,
        le=2.0,
    )
    enable_llm_judge: bool = Field(
        default=True,
        description="Enable LLM-as-judge scoring. When disabled, relevance_score = cosine similarity (1 - distance).",
    )
    mode: Literal["vector", "hybrid"] = Field(
        default="vector",
        description="Query mode: 'vector' for vector-only search, 'hybrid' for BM25+vector combined search.",
    )
