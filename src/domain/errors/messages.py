"""Centralized error message templates.

The message text raised with every domain error is declared here so error
wording is discoverable, consistent and never duplicated. Callers format the
template with the runtime values::

    raise StorageNotFoundError(ErrorMessage.OBJECT_NOT_FOUND.format(bucket=b, path=p))
"""

from enum import StrEnum


class ErrorMessage(StrEnum):
    """Catalog of centralized error message templates (``str.format`` style)."""

    # --- Configuration / dependencies ---
    CLASSICAL_RAG_UNAVAILABLE_VECTOR = (
        "Classical RAG unavailable: vector store not initialized"
    )
    CLASSICAL_RAG_UNAVAILABLE_VECTOR_LLM = (
        "Classical RAG unavailable: vector store or LLM not initialized"
    )

    # --- Storage (MinIO) ---
    OBJECT_NOT_FOUND = "Object not found: bucket={bucket}, path={path}"
    BUCKET_NOT_FOUND = "Bucket not found: {bucket}"

    # --- Vector store (pgvector) ---
    VECTOR_STORE_NOT_INITIALIZED = (
        "Vector store not initialized for working_dir: {working_dir}"
    )

    # --- Document reader (Kreuzberg) ---
    UNSUPPORTED_FILE_FORMAT = "Unsupported file format: {error}"
    INVALID_FILE = "Invalid file: {error}"
    KREUZBERG_EXTRACTION_FAILED = "Kreuzberg extraction failed: {error}"
    KREUZBERG_EXTRACTION_TIMED_OUT = (
        "Kreuzberg extraction timed out after {timeout}s"
    )

    # --- RAG engine (LightRAG) ---
    UNKNOWN_DOCUMENT_PARSER = (
        "Unknown document parser: {parser!r}. Choose from: kreuzberg, {choices}"
    )
    RAG_ENGINE_NOT_INITIALIZED = (
        "RAG engine not initialized for '{working_dir}'. Call init_project() first."
    )

    # --- Classical RAG helpers / BM25 ---
    FILE_NAME_ESCAPES_OUTPUT_DIR = (
        "file_name escapes output directory: {file_name}"
    )
    INVALID_TABLE_PREFIX = "Invalid table_prefix: {table_prefix!r}"
    INVALID_TEXT_CONFIG = "Invalid text_config: {text_config!r}"

    # --- Bricks API ---
    BRICKS_AUTH_FAILED = "Bricks API authentication failed (HTTP {code})"
    BRICKS_PROJECT_NOT_FOUND = "Bricks project not found: {project_id}"
    BRICKS_API_ERROR = "Bricks API error (HTTP {code})"
    BRICKS_CONNECTION_FAILED = "Bricks API connection failed: {reason}"
    BRICKS_REQUEST_TIMED_OUT = "Bricks API request timed out: {error}"
    BRICKS_DOCUMENT_NOT_FOUND = (
        "Document {document_id} not found in project {project_id}"
    )
    DOCUMENT_DOWNLOAD_AUTH_FAILED = (
        "Document download authentication failed (HTTP {code})"
    )
    DOCUMENT_NOT_FOUND = "Document {document_id} not found (project {project_id})"
    DOCUMENT_DOWNLOAD_FAILED = "Failed to download document (HTTP {code})"
    DOCUMENT_DOWNLOAD_CONNECTION_FAILED = (
        "Document download connection failed: {reason}"
    )
    DOCUMENT_DOWNLOAD_TIMED_OUT = "Document download timed out: {error}"
    PUBLISH_AUTH_FAILED = "Publish authentication failed (HTTP {code})"
    PUBLISH_FAILED = "Publish failed (HTTP {code})"
    PUBLISH_CONNECTION_FAILED = "Publish connection failed: {reason}"
    PUBLISH_TIMED_OUT = "Publish request timed out: {error}"

    # --- File routes ---
    FILENAME_REQUIRED = "Filename is required"
    INVALID_FILENAME = "Invalid filename"
    FILE_TYPE_NOT_ALLOWED = "File type '{ext}' is not allowed"
    CONTENT_TYPE_NOT_ALLOWED = "Content type '{content_type}' is not allowed"
    PREFIX_MUST_BE_RELATIVE = "prefix must be a relative path within the bucket"
    PREFIX_MUST_BE_RELATIVE_SHORT = "prefix must be a relative path"
    FILE_TOO_LARGE = (
        "File exceeds maximum allowed size of {max_mb} MB"
    )

    # --- Indexing / query ---
    VECTOR_SEARCH_FAILED = "Vector search failed in hybrid+ mode: {error}"

    # --- MCP tool generic failures ---
    MCP_LIST_DOCS_FAILED_GENERIC = "Failed to list bricks documents: {error}"
    MCP_READ_DOC_FAILED_GENERIC = "Failed to read bricks document: {error}"
    MCP_PUBLISH_FAILED_GENERIC = "Failed to publish section version: {error}"
