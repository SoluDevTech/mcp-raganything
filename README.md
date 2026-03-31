# MCP-RAGAnything

Multi-modal RAG service exposing a REST API and MCP server for document indexing and knowledge-base querying, powered by [RAGAnything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG). Files are retrieved from MinIO object storage and indexed into a PostgreSQL-backed knowledge graph. Each project is isolated via its own `working_dir`.

## Architecture

```
                        Clients
                 (REST / MCP / Claude)
                          |
               +-----------------------+
               |      FastAPI App      |
               +-----------+-----------+
                           |
           +---------------+---------------+
           |                               |
  Application Layer                  MCP Tools
  +------------------------------+   (FastMCP)
  | api/                         |       |
  |   indexing_routes.py         |       |
  |   query_routes.py           |       |
  |   health_routes.py          |       |
  | use_cases/                   |       |
  |   IndexFileUseCase           |       |
  |   IndexFolderUseCase         |       |
  | requests/ responses/         |       |
  +------------------------------+       |
           |              |              |
           v              v              v
  Domain Layer (ports)
  +--------------------------------------+
  | RAGEnginePort        StoragePort     |
  +--------------------------------------+
           |              |
           v              v
  Infrastructure Layer (adapters)
  +--------------------------------------+
  | LightRAGAdapter      MinioAdapter    |
  | (RAGAnything)        (minio-py)      |
  +--------------------------------------+
           |              |
           v              v
     PostgreSQL        MinIO
     (pgvector +     (object
      Apache AGE)    storage)
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
    "file_extensions": [".pdf", ".docx"]
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
| `file_extensions` | list[string] | no | `null` (all files) | Filter by extensions, e.g. `[".pdf", ".docx"]` |

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
| `mode` | string | no | `"naive"` | Search mode (see Query Modes below) |
| `top_k` | integer | no | `10` | Number of chunks to retrieve |

## MCP Server

The MCP server is mounted at `/mcp` and exposes a single tool: `query_knowledge_base`.

### Tool: `query_knowledge_base`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | string | required | RAG workspace directory for this project |
| `query` | string | required | The search query |
| `mode` | string | `"naive"` | Search mode: `naive`, `local`, `global`, `hybrid`, `mix`, `bypass` |
| `top_k` | integer | `10` | Number of chunks to retrieve |

### Transport modes

The `MCP_TRANSPORT` environment variable controls how the MCP server is exposed:

| Value | Behavior |
|-------|----------|
| `stdio` | MCP runs over stdin/stdout; FastAPI runs in a background thread |
| `sse` | MCP mounted at `/mcp` as SSE endpoint |
| `streamable` | MCP mounted at `/mcp` as streamable HTTP endpoint |

### Claude Desktop configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "raganything": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/mcp-raganything",
        "python",
        "-m",
        "src.main"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

## Configuration

All configuration is via environment variables, loaded through Pydantic Settings. See `.env.example` for a complete reference.

### Application (`AppConfig`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio`, `sse`, `streamable` |
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
| `mix` | Knowledge graph + vector chunks combined |
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

## Project Structure

```
src/
  main.py                           -- FastAPI app, MCP mount, entry point
  config.py                         -- Pydantic Settings config classes
  dependencies.py                   -- Dependency injection wiring
  domain/
    entities/
      indexing_result.py             -- FileIndexingResult, FolderIndexingResult
    ports/
      rag_engine.py                  -- RAGEnginePort (abstract)
      storage_port.py                -- StoragePort (abstract)
  application/
    api/
      health_routes.py               -- GET /health
      indexing_routes.py              -- POST /file/index, /folder/index
      query_routes.py                -- POST /query
      mcp_tools.py                   -- MCP tool: query_knowledge_base
    requests/
      indexing_request.py            -- IndexFileRequest, IndexFolderRequest
      query_request.py               -- QueryRequest
    responses/
      query_response.py              -- QueryResponse, QueryDataResponse
    use_cases/
      index_file_use_case.py         -- Downloads from MinIO, indexes single file
      index_folder_use_case.py       -- Downloads from MinIO, indexes folder
  infrastructure/
    rag/
      lightrag_adapter.py            -- LightRAGAdapter (RAGAnything/LightRAG)
    storage/
      minio_adapter.py               -- MinioAdapter (minio-py client)
```

## License

MIT
