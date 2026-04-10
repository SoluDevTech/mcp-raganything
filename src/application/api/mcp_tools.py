"""MCP tools for RAGAnything.

These tools are registered with FastMCP for Claude Desktop integration.
"""

import logging
from dataclasses import asdict

from fastmcp import FastMCP

from application.requests.query_request import MultimodalContentItem
from application.responses.file_response import FileContentResponse, FileInfoResponse
from application.responses.query_response import ChunkResponse, QueryResponse
from dependencies import (
    get_list_files_use_case,
    get_multimodal_query_use_case,
    get_query_use_case,
    get_read_file_use_case,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(
    working_dir: str, query: str, mode: str = "hybrid", top_k: int = 5
) -> list[ChunkResponse]:
    """Search the RAGAnything knowledge base for relevant document chunks.

    Args:
        working_dir: RAG workspace directory for this project
        query: The user's question or search query
        mode: Search mode
            - "naive": Vector search only
            - "local": Local knowledge graph search
            - "global": Global knowledge graph search
            - "hybrid": Local + global knowledge graph
            - "hybrid+": BM25 + vector search (parallel)
            - "mix": Knowledge graph + vector chunks
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


@mcp.tool()
async def list_files(
    prefix: str = "", recursive: bool = True
) -> list[FileInfoResponse]:
    """List files in MinIO storage under a given prefix.

    Args:
        prefix: MinIO prefix/path to filter files by (e.g. 'documents/')
        recursive: Whether to list files in subdirectories (default True)

    Returns:
        List of file objects with object_name, size, and last_modified
    """
    use_case = get_list_files_use_case()
    files = await use_case.execute(prefix=prefix, recursive=recursive)
    return [FileInfoResponse(**asdict(f)) for f in files]


@mcp.tool()
async def read_file(file_path: str) -> FileContentResponse:
    """Read and extract text content from a file stored in MinIO.

    Supports 91 file formats including PDF, Office documents, images, HTML, etc.
    Uses Kreuzberg for document intelligence extraction.

    Args:
        file_path: Path to the file in MinIO bucket (e.g. 'documents/report.pdf')

    Returns:
        Extracted text content with metadata and any detected tables
    """
    use_case = get_read_file_use_case()
    try:
        result = await use_case.execute(file_path=file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}") from None
    except Exception:
        logger.exception("Unexpected error reading file: %s", file_path)
        raise RuntimeError("Failed to read file") from None
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )
