"""MCP tools for RAGAnything.

These tools are registered with FastMCP for Claude Desktop integration.
"""

from fastmcp import FastMCP

from application.requests.query_request import MultimodalContentItem
from application.responses.query_response import ChunkResponse, QueryResponse
from dependencies import get_multimodal_query_use_case, get_query_use_case

mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(
    working_dir: str, query: str, mode: str = "hybrid", top_k: int = 5
) -> list[ChunkResponse]:
    """Search the RAGAnything knowledge base for relevant document chunks.

    Args:
        working_dir: RAG workspace directory for this project
        query: The user's question or search query
        mode: Search mode - "naive" (default), "local", "global", "hybrid", "mix"
        top_k: Number of chunks to retrieve (default 5)

    Returns:
        Query response from LightRAG
    """
    use_case = get_query_use_case()
    response = QueryResponse(
        **await use_case.execute(
            working_dir=working_dir, query=query, mode=mode, top_k=top_k
        )
    )
    return response.data.chunks


@mcp.tool()
async def query_knowledge_base_multimodal(
    working_dir: str,
    query: str,
    multimodal_content: list[MultimodalContentItem],
    mode: str = "hybrid",
    top_k: int = 5,
) -> dict:
    """Query the knowledge base with multimodal content (images, tables, equations).

    Use this tool when your query involves visual or structured data.
    Each item in multimodal_content must have a "type" field ("image", "table", or "equation")
    plus type-specific fields:
      - image: img_path (file path) or image_data (base64)
      - table: table_data (CSV format), optional table_caption
      - equation: latex (LaTeX string), optional equation_caption

    Args:
        working_dir: RAG workspace directory for this project
        query: The user's question or search query
        multimodal_content: List of multimodal content items
        mode: Search mode - "hybrid" (recommended), "naive", "local", "global", "mix"
        top_k: Number of chunks to retrieve (default 10)

    Returns:
        Query response with multimodal analysis
    """
    use_case = get_multimodal_query_use_case()
    return await use_case.execute(
        working_dir=working_dir,
        query=query,
        multimodal_content=multimodal_content,
        mode=mode,
        top_k=top_k,
    )
