"""Centralized log message templates.

All log message strings used across the application are declared here so they
are discoverable, non-duplicated, and easy to audit. Callers reference these
constants instead of inlining message text, e.g.::

    logger.info(LogMessage.APP_SHUTDOWN_INITIATED)

Each template may contain ``%s``/``%d`` style placeholders consumed by the
stdlib logging lazy interpolation.
"""

from enum import StrEnum


class LogMessage(StrEnum):
    """Catalog of centralized log message templates (stdlib ``%`` style)."""

    # --- Application lifecycle (main.py) ---
    APP_SHUTDOWN_INITIATED = "Application shutdown initiated"
    APP_SHUTDOWN_COMPLETE = "Application shutdown complete"
    APP_BM25_CLOSE_FAILED = "Failed to close BM25 adapter"
    APP_CLASSICAL_VS_CLOSE_FAILED = "Failed to close classical vector store"
    APP_MIGRATIONS_RUNNING = "Running database migrations..."
    APP_MIGRATIONS_DONE = "Database migrations completed"

    # --- Storage (MinIO) ---
    MINIO_OBJECT_NOT_FOUND = "Object not found: bucket=%s, path=%s"
    MINIO_RETRIEVE_ERROR = "MinIO error retrieving object: %s"
    MINIO_BUCKET_NOT_FOUND = "Bucket not found: %s"
    MINIO_UPLOAD_ERROR = "MinIO error uploading object: %s"
    MINIO_LIST_ERROR = "MinIO error listing objects: %s"
    MINIO_HEALTH_CHECK_FAILED = "MinIO health check failed"

    # --- Database health (asyncpg) ---
    POSTGRES_HEALTH_CHECK_FAILED = "PostgreSQL health check failed"

    # --- Document reader (Kreuzberg) ---
    KREUZBERG_EXTRACTION_FAILED_FOR = "Kreuzberg extraction failed for %s: %s"
    KREUZBERG_EXTRACTION_DURATION = "Kreuzberg extraction done: file=%s elapsed_ms=%.2f"
    KREUZBERG_SERIALIZATION_DURATION = (
        "Kreuzberg serialization done: file=%s elapsed_ms=%.2f"
    )

    # --- Indexing use cases ---
    INDEXATION_FINISHED = "Indexation finished: %s"
    FOLDER_INDEXATION_FINISHED = "Folder indexation finished: %s"
    PAYLOAD_PREVIEW = "Payload : %s"

    # --- RAG engine (LightRAG) ---
    LIGHTRAG_INDEX_DOCUMENT_FAILED = "Failed to index document %s: %s"
    LIGHTRAG_INDEXED_FILE = "Indexed %s (%d/%d)"
    LIGHTRAG_INDEX_FILE_FAILED = "Failed to index %s: %s"

    # --- Classical query use case ---
    CLASSICAL_BM25_UNAVAILABLE_FALLBACK = (
        "BM25 unavailable, falling back to vector mode"
    )
    CLASSICAL_BM25_SEARCH_FAILED_HYBRID = "BM25 search failed in hybrid mode: %s"
    CLASSICAL_VECTOR_SEARCH_FAILED_HYBRID = "Vector search failed in hybrid mode: %s"

    # --- Query use case (hybrid+) ---
    QUERY_BM25_SEARCH_FAILED = "BM25 search failed in hybrid+ mode: %s"
    QUERY_VECTOR_SEARCH_FAILED = "Vector search failed in hybrid+ mode: %s"

    # --- Background task helpers (routes) ---
    BACKGROUND_TASK_FAILED = "Background %s failed"

    # --- pg_textsearch adapter ---
    PG_TEXTSEARCH_NOT_INSTALLED = (
        "pg_textsearch extension not installed. "
        "BM25 ranking <@> operator will not work. "
        "Run: CREATE EXTENSION pg_textsearch;"
    )
    PG_TEXTSEARCH_CHECK_FAILED = "Could not check pg_textsearch extension: %s"
    PG_TEXTSEARCH_INDEX_EXISTS = "BM25 index '%s' already exists for text_config='%s'"
    PG_TEXTSEARCH_INDEX_CREATED = "Created BM25 index '%s' with text_config='%s'"
    PG_TEXTSEARCH_ENSURE_INDEX_FAILED = "Failed to ensure BM25 index: %s"
    PG_TEXTSEARCH_UPDATING_TRIGGER = (
        "Updating trigger function from old text_config to '%s'"
    )
    PG_TEXTSEARCH_REBUILT_TSV = "Rebuilt content_tsv: %s with text_config='%s'"
    PG_TEXTSEARCH_TRIGGER_CHECK_FAILED = "Could not check/rebuild trigger function: %s"
    PG_TEXTSEARCH_SEARCH_FAILED = "BM25 search failed: %s"
    PG_TEXTSEARCH_INDEX_CREATION_FAILED = "BM25 index creation failed: %s"
    PG_TEXTSEARCH_INDEX_DROP_FAILED = "BM25 index drop failed: %s"

    # --- Classical BM25 adapter ---
    CLASSICAL_PG_TEXTSEARCH_NOT_INSTALLED = (
        "pg_textsearch extension not installed. "
        "BM25 ranking will not work. "
        "Run: CREATE EXTENSION pg_textsearch;"
    )
    CLASSICAL_BM25_TABLE_MISSING = "Table %s does not exist yet, skipping BM25 index"
    CLASSICAL_BM25_INDEX_CREATED = "Created BM25 index '%s' on %s with text_config='%s'"
    CLASSICAL_BM25_ENSURE_INDEX_FAILED = "Failed to ensure BM25 index on %s: %s"
    CLASSICAL_BM25_SEARCH_FAILED = "BM25 search failed on %s: %s"
    CLASSICAL_BM25_INDEX_DROP_FAILED = "BM25 index drop failed on %s: %s"

    # --- Bricks API adapter ---
    BRICKS_GET = "GET %s"
    BRICKS_GET_BYTES = "GET %s -> %d bytes (status=%s)"
    BRICKS_GET_HTTP_ERROR = "GET %s -> HTTP %d: %s"
    BRICKS_GET_ERROR = "GET %s -> error: %s"
    BRICKS_POST = "POST %s (body=%d bytes)"
    BRICKS_POST_BYTES = "POST %s -> %d bytes (status=%s)"
    BRICKS_POST_HTTP_ERROR = "POST %s -> HTTP %d: %s"
    BRICKS_POST_ERROR = "POST %s -> error: %s"
    BRICKS_LISTING_DOCUMENTS = "Listing Bricks documents for project %s"
    BRICKS_FOUND_DOCUMENTS = "Found %d Bricks documents for project %s"
    BRICKS_DOWNLOADING_DOCUMENT = "Downloading Bricks document %s from project %s"
    BRICKS_DOWNLOADED_DOCUMENT = (
        "Downloaded Bricks document %s (%d bytes, mime=%s, filename=%s)"
    )
    BRICKS_PUBLISHING_VERSION = (
        "Publishing section version: project=%s section=%s workflow=%s"
    )
    BRICKS_PUBLISH_PAYLOAD = "Publish payload: %s"
    BRICKS_PUBLISHED_VERSION = "Published section version successfully: %s"
    BRICKS_FILENAME_QUOTED = "Filename from Content-Disposition (quoted): %s"
    BRICKS_FILENAME_UNQUOTED = "Filename from Content-Disposition (unquoted): %s"
    BRICKS_FILENAME_FROM_URL = "Filename from URL path: %s (url=%s)"
    BRICKS_FILENAME_FALLBACK = (
        "Could not extract filename, falling back to document.bin (url=%s)"
    )
    BRICKS_NORMALIZED_EXTENSION = "Normalized file extension: %s -> %s (filename=%s)"

    # --- MCP bricks tools ---
    MCP_DOCUMENTS_FOUND = "Documents found : %s"
    MCP_LIST_DOCS_AUTH_FAILED = "List documents auth failed for project %s: %s"
    MCP_LIST_DOCS_NETWORK_ERROR = "List documents network error for project %s: %s"
    MCP_LIST_DOCS_FAILED = "Failed to list bricks documents for project %s: %s"
    MCP_READ_DOC_RESULT = "Result for read document use case : %s"
    MCP_READ_DOC_AUTH_FAILED = "Read document auth failed for %s: %s"
    MCP_READ_DOC_NETWORK_ERROR = "Read document network error for %s: %s"
    MCP_READ_DOC_FAILED = "Failed to read bricks document: %s in project %s: %s"
    MCP_PUBLISH_AUTH_FAILED = "Publish auth failed for project %s: %s"
    MCP_PUBLISH_NETWORK_ERROR = "Publish network error for project %s: %s"
    MCP_PUBLISH_FAILED = "Failed to publish section version for project %s: %s"

    # --- MCP file tools ---
    MCP_READ_FILE_UNEXPECTED_ERROR = "Unexpected error reading file: %s"

    # --- Exception handler log lines (main.py) ---
    LOG_STORAGE_ERROR = "Storage error: %s"
    LOG_BRICKS_NOT_FOUND = "Bricks not found: %s"
    LOG_BRICKS_PERMISSION = "Bricks permission error: %s"
    LOG_BRICKS_CONNECTION = "Bricks connection error: %s"
    LOG_BRICKS_TIMEOUT = "Bricks timeout: %s"
    LOG_BRICKS_API_ERROR = "Bricks API error: %s"
    LOG_RAG_ERROR = "RAG error: %s"
    LOG_RAG_UNAVAILABLE = "RAG unavailable: %s"
    LOG_VECTOR_STORE_ERROR = "Vector store error: %s"
    LOG_DOCUMENT_ERROR = "Document error: %s"
    LOG_CLASSICAL_ERROR = "Classical config error: %s"
    LOG_INDEXING_ERROR = "Indexing error: %s"
    LOG_CONFIG_ERROR = "Config error: %s"
    LOG_FILE_ERROR = "File error: %s"
    LOG_UNHANDLED_DOMAIN_ERROR = "Unhandled domain error: %s"
