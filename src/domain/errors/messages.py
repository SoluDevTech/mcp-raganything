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
    KREUZBERG_EXTRACTION_TIMED_OUT = "Kreuzberg extraction timed out after {timeout}s"

    # --- Classical RAG helpers / BM25 ---
    FILE_NAME_ESCAPES_OUTPUT_DIR = "file_name escapes output directory: {file_name}"
    INVALID_TABLE_PREFIX = "Invalid table_prefix: {table_prefix!r}"
    INVALID_TEXT_CONFIG = "Invalid text_config: {text_config!r}"

    # --- Security ---
    API_KEY_UNAUTHORIZED = "The Api Key you provided is unauthorized"
    API_KEY_EMPTY = "The Api Key is empty"
    API_KEY_DISABLED = "Auth by Api key is disabled"

    # --- File routes ---
    FILENAME_REQUIRED = "Filename is required"
    INVALID_FILENAME = "Invalid filename"
    FILE_TYPE_NOT_ALLOWED = "File type '{ext}' is not allowed"
    CONTENT_TYPE_NOT_ALLOWED = "Content type '{content_type}' is not allowed"
    PREFIX_MUST_BE_RELATIVE = "prefix must be a relative path within the bucket"
    PREFIX_MUST_BE_RELATIVE_SHORT = "prefix must be a relative path"
    FILE_TOO_LARGE = "File exceeds maximum allowed size of {max_mb} MB"
    FILE_PATH_EMPTY = "file_path must not be empty"
    MINIO_REMOVE_PARTIAL_FAILED = "Failed to delete {count} object(s) from storage"

    # --- Indexing / query ---
    VECTOR_SEARCH_FAILED = "Vector search failed in hybrid+ mode: {error}"
