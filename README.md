# MCP-RAGAnything

Multi-modal RAG service exposing a REST API and MCP server for document indexing and knowledge-base querying, powered by [RAGAnything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG). Two retrieval pathways are available: a **graph-based LightRAG pipeline** and a **classical RAG pipeline** using multi-query generation with LLM-as-judge scoring. Files are retrieved from MinIO object storage and indexed into a PostgreSQL-backed knowledge graph. Each project is isolated via its own `working_dir`.

## Architecture

```
                            Clients
                     (REST / MCP / Claude)
                               |
                 +-------------+-------------+
                 |          FastAPI App        |
                 +-------------+-------------+
                               |
               +---------------+---------------+
               |                               |
       Application Layer            MCP Servers (FastMCP)
       +------------------------------+       |
       | api/                         |   +---+--------+  +--+-----------+  +--+-------------+
       |   indexing_routes.py         |   | RAGAnything |  | RAGAnything |  | RAGAnything    |
       |   query_routes.py            |   | Query       |  | Files       |  | Classical      |
       |   file_routes.py             |   |  /rag/mcp   |  |  /files/mcp |  |  /classical/mcp|
       |   health_routes.py           |   +---+--------+  +--+-----------+  +--+-------------+
       |   classical_indexing_routes   |       |               |                 |
       |   classical_query_routes      |       |               |         classical_index_file
       | use_cases/                   |       |               |         classical_index_folder
       |   IndexFileUseCase           |       |               |         classical_query
       |   IndexFolderUseCase         |
       |   QueryUseCase               |
       |   ClassicalIndexFileUseCase   |
       |   ClassicalIndexFolderUseCase |
       |   ClassicalQueryUseCase       |
       |   ListFilesUseCase           |
       |   ListFoldersUseCase         |
       |   ReadFileUseCase            |
       | requests/ responses/         |
       +------------------------------+
                |         |          |
                v         v          v
     Domain Layer (ports)
     +----------------------------------------------------------+
     | RAGEnginePort  StoragePort  BM25EnginePort              |
     | DocumentReaderPort  VectorStorePort  LLMPort            |
     +----------------------------------------------------------+
              |         |          |            |      |
              v         v          v            v      v
     Infrastructure Layer (adapters)
     +----------------------------------------------------------+
     | LightRAGAdapter       MinioAdapter                        |
     | (RAGAnything)         (minio-py)                          |
     |                                                            |
     | PostgresBM25Adapter       RRFCombiner                      |
     | (pg_textsearch)            (hybrid+ fusion)                |
     |                                                            |
     | KreuzbergAdapter          LangchainPgvectorAdapter         |
     | (kreuzberg - 91 formats) (langchain-postgres PGVector)    |
     |                                                            |
     | LangchainOpenAIAdapter                                     |
     | (langchain-openai ChatOpenAI)                             |
     +----------------------------------------------------------+
              |         |          |            |      |
              v         v          v            v      v
        PostgreSQL        MinIO       Kreuzberg    OpenAI-compatible
        (pgvector +     (object     (document     (LLM API)
         Apache AGE      storage)    extraction)
         pg_textsearch)
```

## Prerequisites

- Python 3.13+
- Docker and Docker Compose
- An [OpenRouter](https://openrouter.ai/) API key (or any OpenAI-compatible provider)
- The `soludev-compose-apps/bricks/` stack for production deployment (provides PostgreSQL, MinIO, and this service)

## Quick Start

Production runs from the shared compose stack at `soludev-compose-apps/bricks/`. The `docker-compose.yml` in this repository is for local development only.

### Local development

```bash
# 1. Install dependencies
uv sync

# 2. Start PostgreSQL and MinIO (docker-compose.yml provides Postgres;
#    MinIO must be available separately or added to the compose file)
docker compose up -d postgres

# 3. Configure environment
cp .env.example .env
# Edit .env: set OPEN_ROUTER_API_KEY and adjust MINIO_HOST / POSTGRES_HOST

# 4. Run the server
uv run python src/main.py
```

The API is available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

### Production (soludev-compose-apps)

```bash
cd soludev-compose-apps/bricks/
docker compose up -d
```

This starts all brick services including `raganything-api`, `postgres`, and `minio`.

## API Reference

Base path: `/api/v1`

### Health

```bash
# Health check
curl http://localhost:8000/api/v1/health
```

Response:

```json
{"message": "RAG Anything API is running"}
```

### Indexing

Both indexing endpoints accept JSON bodies and run processing in the background. Files are downloaded from MinIO, not uploaded directly.

#### Index a single file

Downloads the file identified by `file_name` from the configured MinIO bucket, then indexes it into the RAG knowledge graph scoped to `working_dir`.

```bash
curl -X POST http://localhost:8000/api/v1/file/index \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "project-alpha/report.pdf",
    "working_dir": "project-alpha"
  }'
```

Response (`202 Accepted`):

```json
{"status": "accepted", "message": "File indexing started in background"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_name` | string | yes | Object path in the MinIO bucket |
| `working_dir` | string | yes | RAG workspace directory (project isolation) |

#### Index a folder

Lists all objects under the `working_dir` prefix in MinIO, downloads them, then indexes the entire folder.

```bash
curl -X POST http://localhost:8000/api/v1/folder/index \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "recursive": true,
    "file_extensions": [".pdf", ".docx", ".txt"]
  }'
```

Response (`202 Accepted`):

```json
{"status": "accepted", "message": "Folder indexing started in background"}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `working_dir` | string | yes | -- | RAG workspace directory, also used as the MinIO prefix |
| `recursive` | boolean | no | `true` | Process subdirectories recursively |
| `file_extensions` | list[string] | no | `null` (all files) | Filter by extensions, e.g. `[".pdf", ".docx", ".txt"]` |

## Supported Document Formats

The service automatically detects and processes the following document formats through the RAGAnything parser:

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Includes OCR support (English + French via Tesseract) |
| Microsoft Word | `.docx` | |
| Microsoft PowerPoint | `.pptx` | |
| Microsoft Excel | `.xlsx` | |
| HTML | `.html`, `.htm` | |
| Plain Text | `.txt`, `.text`, `.md` | UTF-8, UTF-16, ASCII supported; converted to PDF via ReportLab |
| Quarto Markdown | `.qmd` | Quarto documents |
| R Markdown | `.Rmd`, `.rmd` | R Markdown files |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif` | Vision model processing (if enabled) |

**Note:** File format detection is automatic. No configuration is required to specify the document type. The service will process any supported format when indexed. All document and image formats are supported out-of-the-box when installed with `raganything[all]`.

## File Browsing & Reading

Browse and read files directly from MinIO without indexing them into the RAG knowledge base. Powered by [Kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) for document text extraction (91 file formats).

### List files

```bash
# List all files in the bucket
curl http://localhost:8000/api/v1/files/list

# List files under a specific prefix
curl "http://localhost:8000/api/v1/files/list?prefix=documents/&recursive=true"
```

Response (`200 OK`):

```json
[
  {"object_name": "documents/report.pdf", "size": 1024, "last_modified": "2026-01-01 00:00:00+00:00"},
  {"object_name": "documents/notes.txt", "size": 512, "last_modified": "2026-01-02 00:00:00+00:00"}
]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | string | `""` | MinIO prefix to filter files by |
| `recursive` | boolean | `true` | List files in subdirectories |

### Upload a file

Uploads a file directly to the MinIO bucket. The file is stored at `{prefix}{filename}`. This endpoint does **not** index the file — use the `POST /file/index` endpoint after uploading to add it to the RAG knowledge base.

**Allowed file types:** `.pdf`, `.txt`, `.docx`, `.xlsx`, `.pptx`, `.md`, `.csv`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.svg`, `.bmp`, `.html`, `.xml`, `.json`, `.rtf`, `.odt`, `.ods`
**Maximum file size:** 50 MB

```bash
curl -X POST http://localhost:8000/api/v1/files/upload \
  -F "file=@report.pdf" \
  -F "prefix=documents/"
```

Response (`201 Created`):

```json
{"object_name": "documents/report.pdf", "size": 2048, "message": "File uploaded successfully"}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | -- | The file to upload (multipart form) |
| `prefix` | string | no | `""` | MinIO prefix (folder path). Must be a relative path |

Error responses:

| Status | Condition |
|--------|-----------|
| `413` | File exceeds 50 MB limit |
| `422` | Invalid prefix (path traversal/absolute), disallowed file type, or missing file |

### Read a file

Downloads the file from MinIO, extracts its text content using Kreuzberg, and returns the result. Supports 91 file formats including PDF, Office documents, images, and HTML.

```bash
curl -X POST http://localhost:8000/api/v1/files/read \
  -H "Content-Type: application/json" \
  -d '{"file_path": "documents/report.pdf"}'
```

Response (`200 OK`):

```json
{
  "content": "Extracted text from the document...",
  "metadata": {"format_type": "pdf", "mime_type": "application/pdf"},
  "tables": [{"markdown": "| Header | Value |\n|---|---|\n| A | 1 |"}]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | string | **Required.** File path in the MinIO bucket (relative, no `..` or absolute paths) |

Error responses:

| Status | Condition |
|--------|-----------|
| `404` | File not found in MinIO |
| `422` | Unsupported file format or invalid path (path traversal, absolute path) |

### Query

Query the indexed knowledge base. The RAG engine is initialized for the given `working_dir` before executing the query.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "query": "What are the main findings of the report?",
    "mode": "naive",
    "top_k": 10
  }'
```

Response (`200 OK`):

```json
{
  "status": "success",
  "message": "",
  "data": {
    "entities": [],
    "relationships": [],
    "chunks": [
      {
        "reference_id": "...",
        "content": "...",
        "file_path": "...",
        "chunk_id": "..."
      }
    ],
    "references": []
  },
  "metadata": {
    "query_mode": "naive",
    "keywords": null,
    "processing_info": null
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `working_dir` | string | yes | -- | RAG workspace directory for this project |
| `query` | string | yes | -- | The search query |
| `mode` | string | no | `"naive"` | Search mode: `naive`, `local`, `global`, `hybrid`, `hybrid+`, `mix`, `bm25`, `bypass` |

#### BM25 query mode

Returns results ranked by PostgreSQL full-text search using `pg_textsearch`. Each chunk includes a `score` field with the BM25 relevance score.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "query": "quarterly revenue growth",
    "mode": "bm25",
    "top_k": 10
  }'
```

Response (`200 OK`):

```json
{
  "status": "success",
  "message": "",
  "data": {
    "entities": [],
    "relationships": [],
    "chunks": [
      {
        "chunk_id": "abc123",
        "content": "Quarterly revenue grew 12% year-over-year...",
        "file_path": "reports/financials-q4.pdf",
        "score": 3.456,
        "metadata": {}
      }
    ],
    "references": []
  },
  "metadata": {
    "query_mode": "bm25",
    "total_results": 10
  }
}
```

#### Hybrid+ query mode

Runs BM25 and vector search in parallel, then merges results using Reciprocal Rank Fusion (RRF). Each chunk includes `bm25_rank`, `vector_rank`, and `combined_score` fields.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "query": "quarterly revenue growth",
    "mode": "hybrid+",
    "top_k": 10
  }'
```

Response (`200 OK`):

```json
{
  "status": "success",
  "message": "",
  "data": {
    "entities": [],
    "relationships": [],
    "chunks": [
      {
        "chunk_id": "abc123",
        "content": "Quarterly revenue grew 12% year-over-year...",
        "file_path": "reports/financials-q4.pdf",
        "score": 0.0328,
        "bm25_rank": 1,
        "vector_rank": 3,
        "combined_score": 0.0328,
        "metadata": {}
      }
    ],
    "references": []
  },
  "metadata": {
    "query_mode": "hybrid+",
    "total_results": 10,
    "rrf_k": 60
  }
}
```

The `combined_score` is the sum of `bm25_score` and `vector_score`, each computed as `1 / (k + rank)`. Results are sorted by `combined_score` descending. A chunk that appears in both result sets will have a higher combined score than one that appears in only one.

---

## Classical RAG Pipeline

A second retrieval pathway alongside the graph-based LightRAG. Classical RAG uses a straightforward chunk → embed → retrieve flow with two quality-enhancing techniques: **multi-query generation** and **LLM-as-judge relevance scoring**. It stores chunks in dedicated PGVector tables (one per `working_dir`) and does not build a knowledge graph.

### How it works

1. **Indexing** — A file is downloaded from MinIO, text is extracted via Kreuzberg (with chunking), and each chunk is embedded and stored in a PGVector table.
2. **Querying** — The LLM generates N alternative phrasings of the user query (multi-query), similarity search runs for each variation, results are deduplicated by `chunk_id`, then an LLM judge scores each chunk's relevance on a 0–10 scale. Chunks below the relevance threshold are discarded; the rest are returned sorted by score.

### Classical Indexing

Both classical indexing endpoints accept JSON bodies and run processing in the background.

#### Index a single file (classical)

Downloads the file from MinIO, extracts text with Kreuzberg chunking, and embeds the chunks into a PGVector table scoped to `working_dir`.

```bash
curl -X POST http://localhost:8000/api/v1/classical/file/index \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "project-alpha/report.pdf",
    "working_dir": "project-alpha",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

Response (`202 Accepted`):

```json
{"status": "accepted", "message": "File indexing started in background"}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_name` | string | yes | -- | Object path in the MinIO bucket |
| `working_dir` | string | yes | -- | RAG workspace directory (project isolation) |
| `chunk_size` | integer | no | `1000` | Max characters per chunk (100–10000) |
| `chunk_overlap` | integer | no | `200` | Overlap characters between chunks (0–2000) |

#### Index a folder (classical)

Lists all objects under the `working_dir` prefix in MinIO, downloads them, and indexes each file into the PGVector table.

```bash
curl -X POST http://localhost:8000/api/v1/classical/folder/index \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "recursive": true,
    "file_extensions": [".pdf", ".docx", ".txt"],
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

Response (`202 Accepted`):

```json
{"status": "accepted", "message": "Folder indexing started in background"}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `working_dir` | string | yes | -- | RAG workspace directory, also used as the MinIO prefix |
| `recursive` | boolean | no | `true` | Process subdirectories recursively |
| `file_extensions` | list[string] | no | `null` (all files) | Filter by extensions, e.g. `[".pdf", ".docx", ".txt"]` |
| `chunk_size` | integer | no | `1000` | Max characters per chunk (100–10000) |
| `chunk_overlap` | integer | no | `200` | Overlap characters between chunks (0–2000) |

### Classical Query

Query the classical RAG pipeline. Supports two modes: **vector** (default) and **hybrid** (BM25 + vector via Reciprocal Rank Fusion).

#### Vector mode (default)

The LLM generates query variations, runs vector similarity search for each, deduplicates results, then scores and filters them with an LLM judge.

```bash
curl -X POST http://localhost:8000/api/v1/classical/query \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "query": "What are the main findings of the report?",
    "top_k": 10,
    "num_variations": 3,
    "relevance_threshold": 5.0,
    "mode": "vector"
  }'
```

#### Hybrid mode

Runs BM25 full-text search and multi-query vector search in parallel, merges results using Reciprocal Rank Fusion (RRF), then scores with an LLM judge. Chunks include `bm25_score`, `vector_score`, and `combined_score` fields.

```bash
curl -X POST http://localhost:8000/api/v1/classical/query \
  -H "Content-Type: application/json" \
  -d '{
    "working_dir": "project-alpha",
    "query": "What are the main findings of the report?",
    "mode": "hybrid"
  }'
```

Response (`200 OK`):

```json
{
  "status": "success",
  "message": "",
  "queries": [
    "What are the main findings of the report?",
    "What key results does the report present?",
    "Summarize the primary conclusions from the report"
  ],
  "chunks": [
    {
      "chunk_id": "a1b2c3d4-...",
      "content": "The primary finding indicates that...",
      "file_path": "project-alpha/report.pdf",
      "relevance_score": 8.5,
      "metadata": {"chunk_index": 0},
      "bm25_score": 0.0164,
      "vector_score": 0.0164,
      "combined_score": 0.0328
    }
  ],
  "mode": "hybrid"
}
```

If BM25 is unavailable (`BM25_ENABLED=false` or pg_textsearch extension missing), hybrid mode falls back to vector mode and logs a warning.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `working_dir` | string | yes | -- | RAG workspace directory for this project |
| `query` | string | yes | -- | The search query |
| `top_k` | integer | no | `10` | Maximum chunks to retrieve per query variation (1–100) |
| `num_variations` | integer | no | `3` | Number of LLM-generated query variations (1–10) |
| `relevance_threshold` | float | no | `5.0` | Minimum LLM judge score (0–10) to include a chunk |
| `mode` | string | no | `"vector"` | Query mode: `vector` (vector-only) or `hybrid` (BM25+vector RRF) |

### LightRAG vs Classical RAG

| Aspect | LightRAG (graph-based) | Classical RAG |
|--------|----------------------|---------------|
| Storage | Apache AGE knowledge graph + pgvector | PGVector tables only |
| Indexing | Builds entity/relationship graph | Chunk + embed only |
| Query modes | `naive`, `local`, `global`, `hybrid`, `hybrid+`, `mix`, `bm25`, `bypass` | `vector` (multi-query + LLM judge), `hybrid` (BM25+vector RRF) |
| Project isolation | Shared graph per `working_dir` | Separate PG table per `working_dir` |
| Best for | Complex reasoning, relationship traversal | Straightforward document Q&A, simpler setup |

---

## MCP Servers

The service exposes **three MCP servers**, all using streamable HTTP transport:

### RAGAnythingQuery — `/rag/mcp`

Query-focused tools for searching the indexed knowledge base.

#### Tool: `query_knowledge_base`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | string | required | RAG workspace directory for this project |
| `query` | string | required | The search query |
| `mode` | string | `"hybrid"` | Search mode: `naive`, `local`, `global`, `hybrid`, `hybrid+`, `mix`, `bm25`, `bypass` |
| `top_k` | integer | `5` | Number of chunks to retrieve |

#### Tool: `query_knowledge_base_multimodal`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | string | required | RAG workspace directory for this project |
| `query` | string | required | The search query |
| `multimodal_content` | list | required | List of multimodal content items |
| `mode` | string | `"hybrid"` | Search mode |
| `top_k` | integer | `5` | Number of chunks to retrieve |

### RAGAnythingFiles — `/files/mcp`

File browsing tools for listing and reading files from MinIO storage.

#### Tool: `list_files`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | string | `""` | MinIO prefix to filter files by |
| `recursive` | boolean | `true` | List files in subdirectories |

#### Tool: `read_file`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | required | File path in MinIO bucket (e.g. `documents/report.pdf`) |

Downloads the file from MinIO, extracts its text content using Kreuzberg, and returns the extracted text along with metadata and any detected tables.

### RAGAnythingClassical — `/classical/mcp`

Classical RAG tools for indexing and querying without a knowledge graph.

#### Tool: `classical_index_file`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_name` | string | required | Object path in the MinIO bucket |
| `working_dir` | string | required | RAG workspace directory (project isolation) |
| `chunk_size` | integer | `1000` | Max characters per chunk (100–10000) |
| `chunk_overlap` | integer | `200` | Overlap characters between chunks (0–2000) |

#### Tool: `classical_index_folder`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | string | required | RAG workspace directory, also used as MinIO prefix |
| `recursive` | boolean | `true` | Process subdirectories recursively |
| `file_extensions` | list[string] | `null` (all files) | Filter by extensions, e.g. `[".pdf", ".docx"]` |
| `chunk_size` | integer | `1000` | Max characters per chunk (100–10000) |
| `chunk_overlap` | integer | `200` | Overlap characters between chunks (0–2000) |

#### Tool: `classical_query`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | string | required | RAG workspace directory for this project |
| `query` | string | required | The search query |
| `top_k` | integer | `10` | Maximum chunks to retrieve per query variation |
| `num_variations` | integer | `3` | Number of LLM-generated query variations (1–10) |
| `relevance_threshold` | float | `5.0` | Minimum LLM judge score (0–10) to include a chunk |
| `mode` | string | `"vector"` | Query mode: `vector` (vector-only) or `hybrid` (BM25+vector RRF) |

### Transport

All MCP servers use **streamable HTTP** transport exclusively. Connect MCP clients to the mount paths:

```
http://localhost:8000/rag/mcp          # RAGAnythingQuery
http://localhost:8000/files/mcp        # RAGAnythingFiles
http://localhost:8000/classical/mcp    # RAGAnythingClassical
```

## Configuration

All configuration is via environment variables, loaded through Pydantic Settings. See `.env.example` for a complete reference.

### Application (`AppConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins |
| `OUTPUT_DIR` | system temp | Temporary directory for downloaded files |
| `UVICORN_LOG_LEVEL` | `critical` | Uvicorn log level |

### Database (`DatabaseConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `raganything` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `raganything` | PostgreSQL password |
| `POSTGRES_DATABASE` | `raganything` | PostgreSQL database name |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |

### LLM (`LLMConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPEN_ROUTER_API_KEY` | -- | **Required.** OpenRouter API key |
| `OPEN_ROUTER_API_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `BASE_URL` | -- | Override base URL (takes precedence over `OPEN_ROUTER_API_URL`) |
| `CHAT_MODEL` | `openai/gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding vector dimension |
| `MAX_TOKEN_SIZE` | `8192` | Max token size for embeddings |
| `VISION_MODEL` | `openai/gpt-4o` | Vision model for image processing |

### RAG (`RAGConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_STORAGE_TYPE` | `postgres` | Storage backend: `postgres` or `local` |
| `COSINE_THRESHOLD` | `0.2` | Similarity threshold for vector search (0.0-1.0) |
| `MAX_CONCURRENT_FILES` | `1` | Concurrent file processing limit |
| `MAX_WORKERS` | `3` | Workers for folder processing |
| `ENABLE_IMAGE_PROCESSING` | `true` | Process images during indexing |
| `ENABLE_TABLE_PROCESSING` | `true` | Process tables during indexing |
| `ENABLE_EQUATION_PROCESSING` | `true` | Process equations during indexing |

### BM25 (`BM25Config`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BM25_ENABLED` | `true` | Enable BM25 full-text search |
| `BM25_TEXT_CONFIG` | `english` | PostgreSQL text search configuration |
| `BM25_RRF_K` | `60` | RRF constant K for hybrid search (must be >= 1) |

When `BM25_ENABLED` is `false` or the pg_textsearch extension is not available, `hybrid+` mode falls back to `naive` (vector-only) and `bm25` mode returns an error.

### Classical RAG (`ClassicalRAGConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLASSICAL_CHUNK_SIZE` | `1000` | Max characters per chunk (Kreuzberg `ChunkingConfig`) |
| `CLASSICAL_CHUNK_OVERLAP` | `200` | Overlap characters between chunks |
| `CLASSICAL_NUM_QUERY_VARIATIONS` | `3` | Number of multi-query variations the LLM generates (1–10) |
| `CLASSICAL_RELEVANCE_THRESHOLD` | `5.0` | Minimum LLM judge score (0–10) for a chunk to be included in results |
| `CLASSICAL_TABLE_PREFIX` | `classical_rag_` | Prefix for PGVectorStore table names. Full name: `{prefix}{sha256(working_dir)[:16]}` |
| `CLASSICAL_LLM_TEMPERATURE` | `0.0` | Temperature for LLM calls (multi-query generation + judge scoring) |
| `CLASSICAL_RRF_K` | `60` | RRF constant K for hybrid BM25+vector search (must be >= 1) |

The classical RAG adapters share the same `OPEN_ROUTER_API_KEY`, `OPEN_ROUTER_API_URL`/`BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIM` settings from the LLM config. If initialization fails (e.g. missing API key), the classical endpoints return `503 Service Unavailable` with a descriptive error.

### MinIO (`MinioConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_HOST` | `localhost:9000` | MinIO endpoint (host:port) |
| `MINIO_ACCESS` | `minioadmin` | MinIO access key |
| `MINIO_SECRET` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `raganything` | Default bucket name |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |

## Query Modes

| Mode | Description |
|------|-------------|
| `naive` | Vector search only -- fast, recommended default |
| `local` | Entity-focused search using the knowledge graph |
| `global` | Relationship-focused search across the knowledge graph |
| `hybrid` | Combines local + global strategies |
| `hybrid+` | Parallel BM25 + vector search using Reciprocal Rank Fusion (RRF). Best of both worlds |
| `mix` | Knowledge graph + vector chunks combined |
| `bm25` | BM25 full-text search only. PostgreSQL pg_textsearch |
| `bypass` | Direct LLM query without retrieval |

## Development

```bash
uv sync                          # Install all dependencies (including dev)
uv run python src/main.py        # Run the server locally
uv run pytest                    # Run tests with coverage
uv run ruff check src/           # Lint
uv run ruff format src/          # Format
uv run mypy src/                 # Type checking
```

### Docker (local)

```bash
docker compose up -d             # Start Postgres + API
docker compose logs -f raganything-api   # Follow API logs
docker compose down -v           # Stop and remove volumes
```

## Database Migrations

Alembic migrations run automatically at startup via the `db_lifespan` context manager in `main.py`. The migration state is tracked in the `raganything_alembic_version` table, which is separate from the `composable-agents` Alembic table to avoid conflicts.

The initial migration (`001_add_bm25_support`) creates the `chunks` table with a `tsvector` column for full-text search, GIN and BM25 indexes, and an auto-update trigger.

### Production requirements

The PostgreSQL server must have the `pg_textsearch` extension installed and loaded. In production, this requires:

1. **Dockerfile.db** builds a custom PostgreSQL image that compiles `pg_textsearch` from source (along with `pgvector` and `Apache AGE`).

2. **docker-compose.yml** must configure `shared_preload_libraries=pg_textsearch` for the `bricks-db` service. The local dev `docker-compose.yml` in this repository includes this by default.

3. The Alembic migration `001_add_bm25_support` will fail if `pg_textsearch` is not available. Ensure the database image is built from `Dockerfile.db` and the shared library is preloaded.

## Project Structure

```
src/
  main.py                           -- FastAPI app, triple MCP mounts, entry point
  config.py                         -- Pydantic Settings config classes
  dependencies.py                   -- Dependency injection wiring
  domain/
    entities/
      indexing_result.py             -- FileIndexingResult, FolderIndexingResult
    ports/
      rag_engine.py                  -- RAGEnginePort (abstract)
      storage_port.py                -- StoragePort (abstract) + FileInfo
      bm25_engine.py                 -- BM25EnginePort (abstract)
      document_reader_port.py        -- DocumentReaderPort (abstract) + DocumentContent
      vector_store_port.py          -- VectorStorePort (abstract) + SearchResult
      llm_port.py                   -- LLMPort (abstract)
  application/
    api/
      health_routes.py               -- GET /health
      indexing_routes.py              -- POST /file/index, /folder/index
      query_routes.py                 -- POST /query
      file_routes.py                  -- GET /files/list, GET /files/folders, POST /files/read
      classical_indexing_routes.py   -- POST /classical/file/index, /classical/folder/index
      classical_query_routes.py      -- POST /classical/query
      mcp_query_tools.py              -- MCP tools: query_knowledge_base, query_knowledge_base_multimodal
      mcp_file_tools.py               -- MCP tools: list_files, read_file
      mcp_classical_tools.py          -- MCP tools: classical_index_file, classical_index_folder, classical_query
    requests/
      indexing_request.py            -- IndexFileRequest, IndexFolderRequest
      classical_indexing_request.py  -- ClassicalIndexFileRequest, ClassicalIndexFolderRequest
      classical_query_request.py     -- ClassicalQueryRequest
      query_request.py                -- QueryRequest, MultimodalQueryRequest
      file_request.py                 -- ListFilesRequest, ReadFileRequest
    responses/
      query_response.py              -- QueryResponse, QueryDataResponse
      classical_query_response.py    -- ClassicalQueryResponse, ClassicalChunkResponse
      file_response.py                -- FileInfoResponse, FileContentResponse
    use_cases/
      index_file_use_case.py         -- Downloads from MinIO, indexes single file (LightRAG)
      index_folder_use_case.py       -- Downloads from MinIO, indexes folder (LightRAG)
      query_use_case.py              -- Query with bm25/hybrid+ support
      classical_index_file_use_case.py  -- Classical: download → Kreuzberg chunk → PGVector
      classical_index_folder_use_case.py -- Classical: folder batch index
      classical_query_use_case.py    -- Classical: multi-query + LLM judge + hybrid BM25
      _classical_helpers.py          -- validate_path, build_documents_from_extraction
      list_files_use_case.py          -- Lists files with metadata from MinIO
      list_folders_use_case.py        -- Lists folder prefixes from MinIO
      read_file_use_case.py           -- Reads file from MinIO, extracts content via Kreuzberg
      upload_file_use_case.py          -- Uploads file to MinIO storage
  infrastructure/
    rag/
      lightrag_adapter.py            -- LightRAGAdapter (RAGAnything/LightRAG)
    storage/
      minio_adapter.py               -- MinioAdapter (minio-py client)
    document_reader/
      kreuzberg_adapter.py            -- KreuzbergAdapter (kreuzberg, 91 formats)
    bm25/
      pg_textsearch_adapter.py        -- PostgresBM25Adapter (pg_textsearch, LightRAG tables)
      classical_bm25_adapter.py        -- ClassicalBM25Adapter (pg_textsearch, classical_rag_* tables)
    hybrid/
      rrf_combiner.py                 -- RRFCombiner (Reciprocal Rank Fusion)
    vector_store/
      langchain_pgvector_adapter.py -- LangchainPgvectorAdapter (langchain-postgres PGVectorStore)
    llm/
      langchain_openai_adapter.py    -- LangchainOpenAIAdapter (langchain-openai ChatOpenAI)
  alembic/
    env.py                            -- Alembic migration environment (async)
    versions/
      001_add_bm25_support.py          -- BM25 table, indexes, triggers
```

## License

MIT
