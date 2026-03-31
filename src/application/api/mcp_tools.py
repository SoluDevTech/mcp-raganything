"""MCP tools for RAGAnything.

These tools are registered with FastMCP for Claude Desktop integration.
"""

from fastmcp import FastMCP

from dependencies import get_query_use_case

mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(
    working_dir: str, query: str, mode: str = "naive", top_k: int = 10
) -> dict:
    """Search the RAGAnything knowledge base for relevant document chunks.

    Args:
        working_dir: RAG workspace directory for this project
        query: The user's question or search query
        mode: Search mode - "naive" (default), "local", "global", "hybrid", "mix"
        top_k: Number of chunks to retrieve (default 10)

    Returns:
        Query response from LightRAG
    """
    use_case = get_query_use_case()
    return await use_case.execute(
        working_dir=working_dir, query=query, mode=mode, top_k=top_k
    )
