"""MCP file tools for RAGAnything.

These tools are registered with FastMCP for Claude Desktop integration.
"""

import logging
from dataclasses import asdict

from fastmcp import FastMCP

from application.responses.file_response import FileContentResponse, FileInfoResponse
from dependencies import (
    get_list_files_use_case,
    get_read_file_use_case,
)

logger = logging.getLogger(__name__)

mcp_files = FastMCP("RAGAnythingFiles")


@mcp_files.tool()
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


@mcp_files.tool()
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
