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
    MINIO_REMOVE_OBJECT_ERROR = "MinIO error removing object: %s"
    MINIO_REMOVE_PREFIX_ERROR = "MinIO error removing prefix %s: %s"
    MINIO_REMOVE_OBJECT_PARTIAL_ERROR = "MinIO partial delete error for object %s: %s"

    # --- Delete cascade (MinIO → pgvector) ---
    DELETE_FILE_STARTED = "Delete file started: bucket=%s object=%s working_dir=%s"
    DELETE_FILE_MINIO_DONE = "MinIO object deleted: bucket=%s object=%s"
    DELETE_FILE_VECTORS_DONE = (
        "Vectors deleted for file: working_dir=%s file_path=%s count=%d"
    )
    DELETE_FILE_VECTORS_SKIPPED = (
        "Vector deletion skipped (no vector store configured): object=%s"
    )
    DELETE_FOLDER_STARTED = "Delete folder started: bucket=%s prefix=%s"
    DELETE_FOLDER_MINIO_DONE = "MinIO prefix deleted: bucket=%s prefix=%s"
    DELETE_FOLDER_VECTORS_DONE = (
        "Vectors deleted for folder: working_dir=%s prefix=%s count=%d"
    )
    DELETE_FOLDER_VECTORS_SKIPPED = (
        "Vector deletion skipped (no vector store configured): prefix=%s"
    )

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
    CLASSICAL_FILE_INDEX_STARTED = (
        "Classical file indexing started: file=%s working_dir=%s"
    )
    CLASSICAL_FILE_INDEX_DONE = "Classical file indexing done: file=%s status=%s"
    CLASSICAL_FILE_INDEX_FAILED = "Classical file indexing failed: file=%s error=%s"
    CLASSICAL_FOLDER_INDEX_STARTED = (
        "Classical folder indexing started: working_dir=%s recursive=%s"
    )
    CLASSICAL_FOLDER_INDEX_LISTING = (
        "Classical folder indexing: found %d file(s) under prefix=%s"
    )
    CLASSICAL_FOLDER_INDEX_NO_FILES = (
        "Classical folder indexing: no files found under prefix=%s"
    )
    CLASSICAL_FOLDER_INDEX_FILE_START = "Indexing file %d/%d: %s"
    CLASSICAL_FOLDER_INDEX_FILE_DONE = "Indexed file %d/%d: %s (%d chunks)"
    CLASSICAL_FOLDER_INDEX_FILE_FAILED = "Failed to index file %d/%d: %s error=%s"
    CLASSICAL_FOLDER_INDEX_DONE = "Classical folder indexing done: working_dir=%s total=%d processed=%d failed=%d status=%s"

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

    # --- MCP file tools ---
    MCP_READ_FILE_UNEXPECTED_ERROR = "Unexpected error reading file: %s"

    # --- Exception handler log lines (main.py) ---
    LOG_STORAGE_ERROR = "Storage error: %s"
    LOG_RAG_ERROR = "RAG error: %s"
    LOG_RAG_UNAVAILABLE = "RAG unavailable: %s"
    LOG_VECTOR_STORE_ERROR = "Vector store error: %s"
    LOG_DOCUMENT_ERROR = "Document error: %s"
    LOG_CLASSICAL_ERROR = "Classical config error: %s"
    LOG_INDEXING_ERROR = "Indexing error: %s"
    LOG_CONFIG_ERROR = "Config error: %s"
    LOG_FILE_ERROR = "File error: %s"
    LOG_UNHANDLED_DOMAIN_ERROR = "Unhandled domain error: %s"
    LOG_INVALID_API_KEY = "Invalid API key: %s"
    LOG_MCP_ERROR = "MCP error: %s"
    MCP_CONNECT_FAILED = "MCP connection to %s failed: %s"
    MCP_TOOLS_LOAD_FAILED = "MCP tool loading from %s failed: %s"

    # --- MCP server registry (use cases / store / factory / runner) ---
    MCP_SERVER_CREATED_UC = "MCP server created (use case): name=%s tool_count=%d"
    MCP_SERVER_UPDATED_UC = "MCP server updated (use case): name=%s tool_count=%d"
    MCP_SERVER_DELETED_UC = "MCP server deleted (use case): name=%s"
    MCP_SERVER_LISTED_UC = "MCP servers listed (use case): count=%d"
    MCP_SERVER_GET_UC = "MCP server retrieved (use case): name=%s"
    MCP_SERVER_VALIDATED_UC = "MCP server validated (use case): name=%s tool_count=%d"
    MCP_SERVER_SAVE_REPOSITORY = "MCP server saved to registry: name=%s"
    MCP_SERVER_DELETE_REPOSITORY = "MCP server deleted from registry: name=%s"
    MCP_OPENAPI_SPEC_FETCHED = "OpenAPI spec fetched: url=%s bytes=%d"
    MCP_OPENAPI_SPEC_FETCH_FAILED = "OpenAPI spec fetch failed: url=%s error=%s"
    MCP_OPENAPI_SPEC_INVALID = "OpenAPI spec invalid: url=%s reason=%s"
    MCP_OPENAPI_MCP_BUILT = "OpenAPI MCP server built: name=%s tool_count=%d"
    MCP_RUNNER_MOUNTED = "Generated MCP server mounted: name=%s url=%s tool_count=%d"
    MCP_RUNNER_UNMOUNTED = "Generated MCP server unmounted: name=%s"
    MCP_RUNNER_MOUNT_FAILED = "Generated MCP server mount failed: name=%s error=%s"
    MCP_REHYDRATION_STARTED = "Rehydrating generated MCP servers from registry..."
    MCP_REHYDRATION_DONE = "Rehydration complete: mounted=%d skipped=%d"
    MCP_REHYDRATION_ENTRY_FAILED = (
        "Rehydration: failed to remount generated MCP server %s: %s"
    )
    MCP_REHYDRATION_SKIPPED_SUMMARY = (
        "Rehydration skipped %d generated MCP server(s) due to mount failures; "
        "they will be retried on the next startup."
    )
