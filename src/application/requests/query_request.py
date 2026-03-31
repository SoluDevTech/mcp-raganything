from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    working_dir: str = Field(
        ..., description="RAG workspace directory for this project"
    )
    query: str = Field(
        ...,
        description="The user's question or search query (e.g., 'What are the main findings?')",
    )
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="naive",
        description=(
            "Search mode - 'naive' (default, recommended), 'local' (context-aware), "
            "'global' (document-level), or 'hybrid' (comprehensive) or 'mix' (automatic strategy). "
        ),
    )
    top_k: int = Field(
        default=10,
        description=(
            "Number of chunks to retrieve (default 10, use 20 for broader search). "
            "Use 10 for fast, focused results; use 20 for comprehensive search."
        ),
    )
