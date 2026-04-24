from fastmcp import FastMCP

from dependencies import (
    get_classical_index_file_use_case,
    get_classical_index_folder_use_case,
    get_classical_query_use_case,
)

mcp_classical = FastMCP("RAGAnythingClassical")


@mcp_classical.tool()
async def classical_index_file(
    file_name: str, working_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200
):
    use_case = get_classical_index_file_use_case()
    return await use_case.execute(
        file_name=file_name,
        working_dir=working_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@mcp_classical.tool()
async def classical_index_folder(
    working_dir: str,
    recursive: bool = True,
    file_extensions: list[str] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    use_case = get_classical_index_folder_use_case()
    return await use_case.execute(
        working_dir=working_dir,
        recursive=recursive,
        file_extensions=file_extensions,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@mcp_classical.tool()
async def classical_query(
    working_dir: str,
    query: str,
    top_k: int = 10,
    num_variations: int = 3,
    relevance_threshold: float = 5.0,
    vector_distance_threshold: float | None = None,
    enable_llm_judge: bool = True,
):
    use_case = get_classical_query_use_case()
    return await use_case.execute(
        working_dir=working_dir,
        query=query,
        top_k=top_k,
        num_variations=num_variations,
        relevance_threshold=relevance_threshold,
        vector_distance_threshold=vector_distance_threshold,
        enable_llm_judge=enable_llm_judge,
    )
