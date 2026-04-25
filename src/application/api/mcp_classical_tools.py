from typing import Literal

from fastmcp import FastMCP

from dependencies import (
    get_classical_index_file_use_case,
    get_classical_index_folder_use_case,
    get_classical_query_use_case,
)

mcp_classical = FastMCP("RAGAnythingClassical")


@mcp_classical.tool()
async def classical_query(
    working_dir: str,
    query: str,
    top_k: int = 10,
    num_variations: int = 3,
    relevance_threshold: float = 5.0,
    vector_distance_threshold: float = 0.5,
    enable_llm_judge: bool = True,
    mode: Literal["vector", "hybrid"] = "vector",
):
    """Query the classical RAG knowledge base.

    Args:
        working_dir: RAG workspace directory for this project.
        query: The search query.
        top_k: Maximum number of chunks to retrieve.
        num_variations: Number of query variations (multi-query).
        relevance_threshold: Minimum relevance score (0-10) from LLM judge.
        vector_distance_threshold: Maximum cosine distance for vector filtering.
        enable_llm_judge: Enable LLM-as-judge scoring.
        mode: Query mode - 'vector' for vector-only, 'hybrid' for BM25+vector combined.
    """
    use_case = get_classical_query_use_case()
    response = await use_case.execute(
        working_dir=working_dir,
        query=query,
        top_k=top_k,
        num_variations=num_variations,
        relevance_threshold=relevance_threshold,
        vector_distance_threshold=vector_distance_threshold,
        enable_llm_judge=enable_llm_judge,
        mode=mode,
    )
    return response.chunks
