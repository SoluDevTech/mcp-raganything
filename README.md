# MCP-RAGAnything

Multi-modal RAG service exposing a REST API and MCP server for document indexing and knowledge-base querying, powered by [RAGAnything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG). Two retrieval pathways are available: a **graph-based LightRAG pipeline** and a **classical RAG pipeline** using multi-query generation with LLM-as-judge scoring. Files are retrieved from MinIO object storage and indexed into a PostgreSQL-backed knowledge graph. Each project is isolated via its own `working_dir`.

The service also hosts the **MCP server registry** — a CRUD API for registering external MCP servers and generating new MCP servers on the fly from OpenAPI/Swagger documents. The registry is persisted in PostgreSQL and rehydrated at startup, so [composable-agents](https://github.com/soludev/composable-agents) and other clients can discover all available MCP servers from a single endpoint. See [MCP Server Registry](#mcp-server-registry).

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
        | api/                         |   +---+--------+  +--+-----------+  +--+-------------+  +--+----------+
        |   indexing_routes.py         |   | RAGAnything |  | RAGAnything |  | RAGAnything    |  | RAGAnything |
        |   query_routes.py            |   | Query       |  | Files       |  | Classical      |  | Bricks     |
        |   file_routes.py             |   |  /rag/mcp   |  |  /files/mcp |  |  /classical/mcp|  | /bricks/mcp|
        |   health_routes.py           |   +---+--------+  +--+-----------+  +--+-------------+  +--+----------+
        |   classical_indexing_routes   |       |               |                 |                  |
        |   classical_query_routes      |       |               |         classical_index_file   list_bricks_documents
        | use_cases/                   |       |               |         classical_index_folder  read_bricks_document
        |   IndexFileUseCase           |       |               |         classical_query         publish_section_version
        |   IndexFolderUseCase         |
        |   QueryUseCase               |
        |   ClassicalIndexFileUseCase   |
        |   ClassicalIndexFolderUseCase |
        |   ClassicalQueryUseCase       |
         |   ListFilesUseCase           |
         |   ListFoldersUseCase         |
         |   ReadFileUseCase            |
         |   UploadFileUseCase          |
         |   CreateFolderUseCase        |
         |   DeleteFileUseCase          |
         |   DeleteFolderUseCase        |
         |   ListBricksDocumentsUseCase |
        |   ReadBricksDocumentUseCase  |
        |   PublishSectionVersionUseCase|
        | requests/ responses/         |
        +------------------------------+
                 |         |          |
                 v         v          v
      Domain Layer (ports)
      +----------------------------------------------------------+
      | RAGEnginePort  StoragePort  BM25EnginePort              |
      | DocumentReaderPort  VectorStorePort  LLMPort            |
      | BricksApiPort                                            |
      +----------------------------------------------------------+
               |         |          |            |      |
               v         v          v            v      v
      Infrastructure Layer (adapters)
      +----------------------------------------------------------+
       | LightRAGAdapter       MinioAdapter                        |
       | (RAGAnything/         (minio-py)                          |
       |  KreuzbergParser)                                         |
      |                                                            |
      | PostgresBM25Adapter       RRFCombiner                      |
      | (pg_textsearch)            (hybrid+ fusion)                |
      |                                                            |
      | KreuzbergAdapter          LangchainPgvectorAdapter         |
      | (kreuzberg - 91 formats) (langchain-postgres PGVector)    |
      |                                                            |
      | LangchainOpenAIAdapter    BricksApiAdapter                 |
      | (langchain-openai ChatOpenAI)  (httpx, Bricks REST API)   |
      +----------------------------------------------------------+
               |         |          |            |      |
               v         v          v            v      v
         PostgreSQL        MinIO       Kreuzberg    OpenAI-compatible    Bricks API
         (pgvector +     (object     (document     (LLM API)          (analyse.bricks.co
          Apache AGE      storage)    extraction)                      + section-versions)
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

## API Key Authentication

The service supports optional API key authentication for both REST and MCP endpoints. A single `API_KEY` environment variable controls all layers.

### Enabling authentication

Set `API_KEY` to a non-empty value in `.env`:

```env
API_KEY=your-secret-key-here
```

When enabled, all REST endpoints (except health) and all MCP tool calls require the `X-API-Key` header. Requests without a valid key receive `401 Unauthorized` (REST) or a `ToolError` (MCP).

### Disabling authentication

Leave `API_KEY` empty (default) to disable authentication — useful for local development and testing:

```env
API_KEY=
```

### REST usage

Include the `X-API-Key` header in every request:

```bash
curl -H "X-API-Key: your-secret-key-here" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/api/v1/classical/query \
     -d '{"working_dir": "project-alpha", "query": "test"}'
```

Health endpoints (`/api/v1/health`, `/api/v1/health/live`) remain public regardless of the `API_KEY` setting.

### MCP usage

MCP clients must send the `X-API-Key` header in their HTTP transport configuration. For [composable-agents](https://github.com/soludev/composable-agents), add it to the `headers` field of each MCP server config:

```yaml
mcp_servers:
  - name: bricks
    transport: http
    url: https://raganything.soludev.tech/bricks/mcp
    headers:
      X-API-Key: "${MCP_RAGANYTHING_API_KEY}"
  - name: files
    transport: http
    url: https://raganything.soludev.tech/files/mcp
    headers:
      X-API-Key: "${MCP_RAGANYTHING_API_KEY}"
  - name: classical
    transport: http
    url: https://raganything.soludev.tech/classical/mcp
    headers:
      X-API-Key: "${MCP_RAGANYTHING_API_KEY}"
```

The `${MCP_RAGANYTHING_API_KEY}` placeholder is resolved from the environment variable of the same name. Set it to the same value as the server's `API_KEY`.

### Protected endpoints

| Layer | Protected | Public |
|-------|-----------|--------|
| REST | `/api/v1/files/*`, `/api/v1/files`, `/api/v1/classical/*`, `/api/v1/file/*`, `/api/v1/folder/*`, `/api/v1/query` | `/api/v1/health`, `/api/v1/health/live` |
| MCP | `tools/call`, `tools/list` (all 3 servers) | `initialize` |

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

### Create a folder

Creates a folder marker in the MinIO bucket by writing a 0-byte object whose object name ends with a trailing `/`. The folder is purely a prefix — MinIO does not have a real folder concept. This endpoint does **not** index anything.

```bash
curl -X POST http://localhost:8000/api/v1/files/folders \
  -H "Content-Type: application/json" \
  -d '{"prefix": "documents/reports/"}'
```

Response (`201 Created`):

```json
{"message": "Folder created", "prefix": "documents/reports/"}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prefix` | string | yes | -- | Folder prefix to create. Must be a relative path; a trailing `/` is preserved |

Error responses:

| Status | Condition |
|--------|-----------|
| `422` | Missing, absolute, or path-traversing `prefix` |

### Delete a file (cascade)

Deletes a single object from the MinIO bucket **and** removes its corresponding vectors from the pgvector store. The `object_name` and `working_dir` are both required query parameters.

Deletion is performed in two steps, strictly in order:

1. **MinIO first** — the object is removed from the storage bucket.
2. **pgvector second** — vectors associated with the file (scoped to `working_dir`) are deleted from the vector store.

If the MinIO deletion fails, the pgvector store is **not** touched, ensuring no orphaned vectors are created from a failed storage operation.

```bash
curl -X DELETE "http://localhost:8000/api/v1/files?object_name=documents/report.pdf&working_dir=project-alpha"
```

Response (`200 OK`):

```json
{"message": "File deleted", "object_name": "documents/report.pdf"}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `object_name` | string | yes | -- | Object path to delete. Must be a relative path within the bucket |
| `working_dir` | string | yes | -- | RAG workspace directory (project isolation). Used to scope the pgvector deletion |

Error responses:

| Status | Condition |
|--------|-----------|
| `422` | Missing, absolute, or path-traversing `object_name` or `working_dir` |

### Delete a folder (recursive, cascade)

Deletes all objects whose object names start with `prefix` from the MinIO bucket **and** removes all corresponding vectors from the pgvector store. The deletion is recursive and permanent — there is no confirmation beyond the API call. A trailing `/` is preserved so that a prefix like `documents/` does not match `documents-archive/`.

The `prefix` is automatically used as the `working_dir` for the pgvector deletion — no separate `working_dir` parameter is needed on this endpoint.

Deletion is performed in two steps, strictly in order:

1. **MinIO first** — all objects under the prefix are removed from the storage bucket.
2. **pgvector second** — all vectors matching the prefix are deleted from the vector store.

If the MinIO deletion fails, the pgvector store is **not** touched.

```bash
curl -X DELETE "http://localhost:8000/api/v1/files/folders?prefix=documents/reports/"
```

Response (`200 OK`):

```json
{"message": "Folder deleted", "prefix": "documents/reports/"}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | yes | -- | Folder prefix to delete. Must be a relative path; trailing `/` is preserved. Also used as the `working_dir` for pgvector cleanup |

Error responses:

| Status | Condition |
|--------|-----------|
| `422` | Missing, absolute, or path-traversing `prefix` |

### Path traversal protection

All delete routes (`DELETE /files`, `DELETE /files/folders`) and the create-folder route (`POST /files/folders`) validate the supplied path before touching storage. The validation rules are shared with the existing `read`/`upload`/`list` endpoints:

- The value must not be empty.
- It must be a relative path (no leading `/`, no drive prefixes).
- After normalization it must not resolve to `.`, `..`, start with `../`, or contain a `/../` segment.

Any violation returns `422 Unprocessable Entity` with a `FileValidationError` body. The `object_name` / `prefix` is normalized (backslashes converted to `/`, trailing slashes preserved) before being forwarded to the `StoragePort`.

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

The service exposes **four MCP servers**, all using streamable HTTP transport:

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

### RAGAnythingBricks — `/bricks/mcp`

Bricks integration tools for accessing project documents from the Bricks platform and publishing structured section versions.

#### Tool: `list_bricks_documents`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_unique_id` | string | required | Bricks project unique identifier |

Returns a list of documents for the specified Bricks project, including metadata like file name, MIME type, size, status, and presigned download URLs.

#### Tool: `read_bricks_document`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_url` | string | required | Presigned S3 URL from `list_bricks_documents` |

Downloads the document from the presigned S3 URL, extracts its text content using Kreuzberg, and returns the extracted text, metadata, and detected tables. No authentication is required — the URL is already signed.

#### Tool: `publish_section_version`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_unique_id` | string | required | Bricks project unique identifier |
| `section_key` | string | required | Section key to publish (e.g. `"summary"`, `"analysis"`) |
| `content` | dict | required | Structured content for the section |
| `workflow_id` | string | `"agent-haiku-files-v1"` | Workflow identifier |
| `workflow_name` | string | `"haiku-files"` | Workflow display name |
| `workflow_metadata` | dict | `null` | Additional workflow metadata |

Publishes a structured section version back to the Bricks platform. When `BRICKS_PUBLISH_DRY_RUN=true` (default), the tool returns a preview of the payload without making an API call. Set `BRICKS_PUBLISH_DRY_RUN=false` to enable real publishing.

**Dry-run response example:**

```json
{
  "success": true,
  "message": "DRY RUN — no API call made",
  "dry_run": true,
  "payload_preview": {
    "project_unique_id": "abc-123",
    "section_key": "summary",
    "content": {"title": "Analysis Summary"},
    "workflow_id": "agent-haiku-files-v1",
    "workflow_name": "haiku-files",
    "workflow_metadata": {}
  }
}
```

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
http://localhost:8000/bricks/mcp       # RAGAnythingBricks
```

---

## MCP Server Registry

In addition to the four built-in MCP servers above, mcp-raganything **owns the MCP server registry** — a CRUD service that lets you register external MCP servers (by URL) or generate new MCP servers on the fly from any OpenAPI/Swagger document. Registered servers are persisted in the `mcp_servers` PostgreSQL table (shared with [composable-agents](https://github.com/soludev/composable-agents)) and rehydrated at startup, so composable-agents (and other clients) only need to know the registry URL to discover and connect to every available MCP server.

This registry was previously hosted in the `composable-agents` brick; it has been migrated here so that the RAG service is the single owner of MCP server lifecycle (registration, generation, mounting, crash recovery).

> **Schema ownership:** mcp-raganything owns the registry *service*, but the `mcp_servers` **table schema** is created and evolved by [composable-agents](https://github.com/soludev/composable-agents)' Alembic migrations (`008_create_mcp_servers_table`, `009_alter_mcp_servers_add_openapi`). This service is a consumer of the shared table and assumes it already exists — on a fresh database, composable-agents' migrations must run before the registry is used. mcp-raganything no longer self-creates the table at startup.

### What it does

- **Register external MCP servers** — store a name, transport URL, optional headers and an encrypted auth token. Composable-agents reads these entries and connects to the URL at agent build time.
- **Generate MCP servers from OpenAPI specs** — point the registry at any OpenAPI 3.0 (or Swagger 2.0) document URL; the service fetches the spec, builds an in-process [FastMCP](https://github.com/jlowin/fastmcp) server exposing one tool per operation, and mounts it under `/generated/{name}/mcp`.
- **Encrypt secrets at rest** — auth tokens and sensitive header values are encrypted with Fernet using the shared `SECRET_ENCRYPTION_KEY` before being written to `mcp_servers`. They are only decrypted on the `/reveal` endpoint or when the server is mounted.
- **Crash recovery** — on startup the service reads every `openapi` row from `mcp_servers`, re-fetches the spec, rebuilds the FastMCP server, and remounts it. External `http` servers do not need rehydration (the client connects to them lazily), so only `openapi` rows are rebuilt.

### Endpoints

All registry endpoints live under `/api/v1/mcp/servers` and are protected by the global `API_KEY` middleware when `API_KEY` is set.

| Method | Path | Description | Success Status |
|---|---|---|---|
| `POST` | `/api/v1/mcp/servers` | Create a registered MCP server (external or openapi). Returns the masked entry (secrets hidden). | `201` |
| `POST` | `/api/v1/mcp/servers/validate` | Dry-run validation of a server config without persisting anything. Returns the parsed/mounted result. | `200` |
| `GET` | `/api/v1/mcp/servers` | List all registered servers (masked). | `200` |
| `GET` | `/api/v1/mcp/servers/{name}` | Get a single server (masked). | `200` |
| `GET` | `/api/v1/mcp/servers/{name}/reveal` | Get a single server **with plaintext secrets**. Use with care. | `200` |
| `PUT` | `/api/v1/mcp/servers/{name}` | Update a server (URL, headers, auth token, openapi spec). Re-mounts openapi servers. | `200` |
| `DELETE` | `/api/v1/mcp/servers/{name}` | Delete a server and unmount it if openapi. | `204` |

The request body for `POST` and `PUT` extends the `McpServerConfig` shape with two fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Unique server name (1–100 chars). |
| `url` | string | `null` | Server URL. Required for `source_type="external"`, omitted for `source_type="openapi"` (the mounted URL is generated). |
| `headers` | dict | `{}` | HTTP headers sent to the server (upstream auth for openapi). |
| `env` | dict | `{}` | Environment variables for `stdio` transport. |
| `auth_token` | string | `null` | Bearer auth token (external only). Encrypted at rest. |
| `source_type` | `"external"` \| `"openapi"` | `"external"` | Origin of the server. `openapi` triggers spec fetch + FastMCP generation. |
| `openapi_url` | string | `null` | URL of the OpenAPI document. Required when `source_type="openapi"`. |

### Swagger 2.0 → OpenAPI 3.0 conversion

When `source_type="openapi"` and the fetched document is a **Swagger 2.0** spec (`swagger: "2.0"`), the service converts it to OpenAPI 3.0 **in-process, in pure Python, with no external service call** (offline). The converter handles:

- `swagger: "2.0"` → `openapi: "3.0.x"`
- `host` + `basePath` + `schemes` → `servers[].url`
- `definitions` → `components.schemas`
- `responses` / `parameters` / `securityDefinitions` → `components.*`
- `produces` / `consumes` → per-operation `requestBody.content` / `responses.*.content`
- `x-...` vendor extensions are preserved

After conversion, the resulting OpenAPI 3.0 document is fed to FastMCP to build the generated server. If the document is already OpenAPI 3.0, no conversion is performed. Validation errors (malformed spec, unreachable URL, unsupported version) are returned as `422` responses on `/validate` and `POST`.

### Generated server mounting & lifecycle

For `source_type="openapi"` servers, the service:

1. Fetches the OpenAPI/Swagger document from `openapi_url` (with optional `headers` for upstream auth).
2. Converts Swagger 2.0 → OpenAPI 3.0 if needed (see above).
3. Builds a FastMCP server exposing one MCP tool per OpenAPI operation (operationId or method+path as the tool name).
4. Mounts the server at **`/generated/{name}/mcp`** using streamable HTTP transport, with proper lifespan management via an `AsyncExitStack` so that HTTP clients and sessions opened by FastMCP are closed cleanly on shutdown or unmount.
5. Persists the entry in `mcp_servers` with `url = {GENERATED_MCP_BASE_URL}/generated/{name}/mcp`.

The returned `url` is built from `GENERATED_MCP_BASE_URL` so that clients outside the container can reach it. Connect an MCP client to:

```
{GENERATED_MCP_BASE_URL}/generated/{name}/mcp
```

For local development the default is `http://localhost:8020/generated/{name}/mcp`. In Docker the default is `http://raganything-api:8000/generated/{name}/mcp` (set `GENERATED_MCP_BASE_URL` to the public/ingress URL if clients are external).

### Startup rehydration (crash recovery)

On startup, the FastAPI lifespan reads all rows from `mcp_servers` and, for each row with `source_type="openapi"`, re-fetches the OpenAPI spec, rebuilds the FastMCP server, and remounts it at `/generated/{name}/mcp`. External `http`/`stdio` servers are **not** rebuilt (the client connects to them lazily on first tool call). If `SECRET_ENCRYPTION_KEY` is not set, the registry is disabled at startup and a warning is logged — the rest of the RAG service still starts.

### Example: register an external server

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "name": "weather",
    "source_type": "external",
    "url": "https://weather.example.com/mcp",
    "auth_token": "super-secret"
  }'
```

Response (`201 Created`, secrets masked):

```json
{
  "name": "weather",
  "source_type": "external",
  "url": "https://weather.example.com/mcp",
  "auth_token": "***",
  "headers": {}
}
```

### Example: generate a server from an OpenAPI spec

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "name": "petstore",
    "source_type": "openapi",
    "openapi_url": "https://petstore.swagger.io/v2/swagger.json",
    "headers": {"Authorization": "Bearer upstream-token"}
  }'
```

Response (`201 Created`) — note the generated `url`:

```json
{
  "name": "petstore",
  "source_type": "openapi",
  "url": "http://localhost:8020/generated/petstore/mcp",
  "openapi_url": "https://petstore.swagger.io/v2/swagger.json",
  "headers": {"Authorization": "***"}
}
```

The mounted server is immediately reachable at `http://localhost:8020/generated/petstore/mcp` and will be rehydrated automatically on the next startup.

---

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
| `API_KEY` | `""` (disabled) | Shared secret key for REST + MCP authentication. When non-empty, all requests must include `X-API-Key` header. Leave empty to disable authentication |
| `SECRET_ENCRYPTION_KEY` | `""` (disabled) | Fernet key used to encrypt MCP registry secrets (auth tokens, sensitive headers) at rest in the `mcp_servers` table. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Must be **stable across restarts** to decrypt previously encrypted secrets. When empty, the MCP server registry is disabled at startup (the rest of the service still runs) |
| `GENERATED_MCP_BASE_URL` | `http://localhost:8020` | Absolute base URL of this service, used to build the public URL returned for generated (`openapi`) MCP servers mounted under `/generated/{name}/mcp`. Set to `http://raganything-api:8000` for Docker, or to your public ingress URL for external clients |

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
| `DOCUMENT_PARSER` | `kreuzberg` | Document parser for LightRAG pipeline: `kreuzberg` (VLM OCR via OpenRouter), `mineru`, or `paddleocr` |
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

### Bricks API (`BricksConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BRICKS_API_BASE_URL` | `https://analyse.bricks.co` | Bricks platform base URL |
| `BRICKS_API_KEY` | -- | X-API-Key for Bricks API authentication (publish) |
| `BRICKS_BEARER_TOKEN` | -- | Bearer token for Bricks API authentication (list documents) |
| `BRICKS_PUBLISH_DRY_RUN` | `true` | When `true`, `publish_section_version` returns a payload preview without making an API call |

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

## Database Schema

mcp-raganything does **not** use Alembic. Schemas are provisioned by three mechanisms:

1. **`mcp_servers` table (shared with composable-agents)** — created and owned by composable-agents' Alembic migrations `008_create_mcp_servers_table` and `009_alter_mcp_servers_add_openapi`. This service consumes the table and assumes it already exists. Run composable-agents' migrations first on a fresh database.

2. **Classical RAG tables (`classical_rag_*`)** — created at runtime by `LangchainPgvectorAdapter` (langchain-postgres `PGVectorStore`) the first time a collection is indexed. No migration is needed.

3. **BM25 index** — created on demand by `ClassicalBM25Adapter._ensure_bm25_index` the first time a `classical_rag_*` table is queried, using `CREATE INDEX ... USING bm25(content)`.

### Production requirements

The PostgreSQL server must have the `pg_textsearch` extension installed and loaded. In production, this requires:

1. **Dockerfile.db** builds a custom PostgreSQL image that compiles `pg_textsearch` from source (along with `pgvector` and `Apache AGE`).

2. **docker-compose.yml** must configure `shared_preload_libraries=pg_textsearch` for the `bricks-db` service. The local dev `docker-compose.yml` in this repository includes this by default.

3. `ClassicalBM25Adapter` checks for the extension at pool-creation time and logs a warning if it is missing; BM25 queries will then fail at runtime. Ensure the database image is built from `Dockerfile.db` and the shared library is preloaded.

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
       storage_port.py                -- StoragePort (abstract, methods: list, read, upload, create_folder, remove_object, remove_prefix) + FileInfo
      bm25_engine.py                 -- BM25EnginePort (abstract)
      document_reader_port.py        -- DocumentReaderPort (abstract) + DocumentContent
      vector_store_port.py          -- VectorStorePort (abstract) + SearchResult
      llm_port.py                   -- LLMPort (abstract)
      bricks_api_port.py             -- BricksApiPort (abstract) + BricksDocumentInfo + SectionVersionResult
  application/
    api/
      health_routes.py               -- GET /health
      indexing_routes.py              -- POST /file/index, /folder/index
      query_routes.py                 -- POST /query
      file_routes.py                  -- GET /files/list, GET /files/folders, POST /files/read, POST /files/upload, POST /files/folders, DELETE /files, DELETE /files/folders
      classical_indexing_routes.py   -- POST /classical/file/index, /classical/folder/index
      classical_query_routes.py      -- POST /classical/query
      mcp_query_tools.py              -- MCP tools: query_knowledge_base, query_knowledge_base_multimodal
      mcp_file_tools.py               -- MCP tools: list_files, read_file
      mcp_classical_tools.py          -- MCP tools: classical_index_file, classical_index_folder, classical_query
      mcp_bricks_tools.py             -- MCP tools: list_bricks_documents, read_bricks_document, publish_section_version
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
       create_folder_use_case.py        -- Creates a folder marker (0-byte trailing-slash object) in MinIO
       delete_file_use_case.py          -- Deletes a single object from MinIO storage
       delete_folder_use_case.py        -- Recursively deletes all objects under a prefix in MinIO storage
       list_bricks_documents_use_case.py   -- Lists documents from Bricks API
      read_bricks_document_use_case.py    -- Downloads Bricks document via presigned URL, extracts via Kreuzberg
      publish_section_version_use_case.py -- Publishes section version (dry-run aware)
  infrastructure/
    rag/
      lightrag_adapter.py            -- LightRAGAdapter (RAGAnything/LightRAG)
      kreuzberg_raganything_parser.py -- KreuzbergRAGAnythingParser (kreuzberg, custom parser for RAGAnything)
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
    bricks/
      bricks_api_adapter.py           -- BricksApiAdapter (httpx, Bricks REST API + section-versions)
```

## Deployment on Railway

[Railway](https://railway.app/) is a deployment platform that supports Docker-based services with managed PostgreSQL. The service requires PostgreSQL with `pgvector` and `pg_textsearch` extensions.

### Architecture on Railway

```
Railway project
├── PostgreSQL (Railway managed plugin — requires pgvector + pg_textsearch)
├── mcp-raganything (Dockerfile deploy)
└── (MinIO — use Railway's external S3-compatible storage or a separate container)
```

### Step-by-step

1. **Create a Railway project** and add a PostgreSQL plugin.

2. **Provision pgvector + pg_textsearch**:
   - Railway's managed PostgreSQL is based on the standard `postgres` image without `pgvector` or `pg_textsearch`.
   - You need a custom PostgreSQL image. Use the `Dockerfile.db` from this repo to build a `pgvector/pgvector:pg17`-based image with `pg_textsearch` compiled in.
   - Alternatively, deploy the custom PostgreSQL as a Railway service using `Dockerfile.db`, and disable the managed plugin.

3. **Deploy mcp-raganything**:
   - New Service → GitHub Repo → select this repository.
   - Railway detects the `Dockerfile` automatically.
   - Set the port to `8000` (Railway auto-detects `EXPOSE 8000`).

4. **Configure environment variables** in the Railway dashboard:

   | Variable | Example | Notes |
   |----------|---------|-------|
   | `POSTGRES_HOST` | `roundhouse.proxy.rlwy.net` | Railway PostgreSQL host |
   | `POSTGRES_PORT` | `33019` | Railway PostgreSQL port (not 5432) |
   | `POSTGRES_USER` | `postgres` | Railway PostgreSQL user |
   | `POSTGRES_PASSWORD` | `********` | Railway PostgreSQL password |
   | `POSTGRES_DATABASE` | `railway` | Railway PostgreSQL database name |
   | `OPEN_ROUTER_API_KEY` | `sk-or-v1-...` | OpenRouter API key |
   | `OPEN_ROUTER_API_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
   | `CHAT_MODEL` | `openai/gpt-4o-mini` | Chat model |
   | `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
   | `EMBEDDING_DIM` | `1536` | Embedding dimensions |
   | `MINIO_HOST` | `minio.xxx.railway.app:9000` | MinIO host (external service) |
   | `MINIO_ACCESS` | `minioadmin` | MinIO access key |
   | `MINIO_SECRET` | `********` | MinIO secret key |
   | `MINIO_BUCKET` | `raganything` | MinIO bucket name |
   | `MINIO_SECURE` | `true` | Use HTTPS in production |
   | `API_KEY` | `your-shared-secret` | API key for REST + MCP auth (empty = disabled) |
   | `ALLOWED_ORIGINS` | `["https://composable-agents.xxx.railway.app"]` | CORS origins |

5. **MinIO**: Railway does not offer managed MinIO. Options:
   - Deploy MinIO as a separate Railway service (Docker image `minio/minio`).
   - Use an external S3-compatible service (AWS S3, Cloudflare R2, etc.) and adapt the MinIO config accordingly.
   - Use the `soludev-compose-apps` stack if you need all services together.

6. **Verify deployment**:
   ```bash
   curl https://mcp-raganything.xxx.up.railway.app/api/v1/health
   ```

7. **Connect composable-agents** to this Railway deployment by setting `MCP_RAGANYTHING_API_KEY` to the same value as `API_KEY` and pointing agent MCP URLs to the Railway domain.

### Notes

- Railway automatically generates a public domain (e.g. `mcp-raganything-production.up.railway.app`).
- The `pg_textsearch` extension must be available on the PostgreSQL server. If using Railway's managed PostgreSQL, you will need to replace it with a custom image built from `Dockerfile.db`.
- Alembic migrations run automatically on startup — no manual migration step is needed.

## License

MIT
