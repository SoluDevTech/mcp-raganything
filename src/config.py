import os
import tempfile

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class AppConfig(BaseSettings):
    ALLOWED_ORIGINS: list[str] = Field(
        default=["*"], description="CORS allowed origins"
    )
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    UVICORN_LOG_LEVEL: str = Field(default="critical", description="Uvicorn log level")
    OUTPUT_DIR: str = Field(
        default=os.path.join(tempfile.gettempdir(), "output"),
        description="Directory for temporary output file storage",
    )


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    POSTGRES_USER: str = Field(default="raganything")
    POSTGRES_PASSWORD: str = Field(default="raganything")
    POSTGRES_DATABASE: str = Field(default="raganything")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")

    @property
    def DATABASE_URL(self) -> str:
        """Construct async PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""

    OPEN_ROUTER_API_KEY: str | None = Field(default=None)
    OPENROUTER_API_KEY: str | None = Field(default=None)
    OPEN_ROUTER_API_URL: str = Field(default="https://openrouter.ai/api/v1")
    BASE_URL: str | None = Field(default=None)

    CHAT_MODEL: str = Field(
        default="openai/gpt-4o-mini", description="Model name for chat completions"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="Model name for embeddings"
    )
    EMBEDDING_DIM: int = Field(
        default=1536, description="Dimension of the embedding vectors"
    )
    MAX_TOKEN_SIZE: int = Field(
        default=8192, description="Maximum token size for the embedding model"
    )
    VISION_MODEL: str = Field(
        default="openai/gpt-4o", description="Model name for vision tasks"
    )

    @property
    def api_key(self) -> str:
        """Get API key with fallback."""
        key = self.OPEN_ROUTER_API_KEY or self.OPENROUTER_API_KEY
        if not key:
            print("WARNING: OPENROUTER_API_KEY not set. API calls will fail.")
        return key or ""

    @property
    def api_base_url(self) -> str:
        """Get API base URL with fallback."""
        return self.BASE_URL or self.OPEN_ROUTER_API_URL


class RAGConfig(BaseSettings):
    """RAG-specific configuration for LightRAG."""

    COSINE_THRESHOLD: float = Field(
        default=0.2, description="Similarity threshold for vector search (0.0-1.0)"
    )
    MAX_CONCURRENT_FILES: int = Field(
        default=1, description="Number of files to process concurrently"
    )
    ENABLE_IMAGE_PROCESSING: bool = Field(
        default=True, description="Enable image processing during indexing"
    )
    ENABLE_TABLE_PROCESSING: bool = Field(
        default=True, description="Enable table processing during indexing"
    )
    ENABLE_EQUATION_PROCESSING: bool = Field(
        default=True, description="Enable equation processing during indexing"
    )
    MAX_WORKERS: int = Field(
        default=3, description="Number of workers for folder processing"
    )
    RAG_STORAGE_TYPE: str = Field(
        default="postgres", description="Storage type for RAG system"
    )


class BM25Config(BaseSettings):
    """BM25 search configuration."""

    BM25_ENABLED: bool = Field(default=True, description="Enable BM25 full-text search")
    BM25_TEXT_CONFIG: str = Field(
        default="english", description="PostgreSQL text search configuration"
    )
    BM25_RRF_K: int = Field(
        default=60, ge=1, description="RRF constant K for hybrid search"
    )


class ClassicalRAGConfig(BaseSettings):
    """Configuration for the classical RAG pathway."""

    CLASSICAL_CHUNK_SIZE: int = Field(
        default=1000, description="Max characters per chunk (Kreuzberg ChunkingConfig)"
    )
    CLASSICAL_CHUNK_OVERLAP: int = Field(
        default=200, description="Overlap characters between chunks"
    )
    CLASSICAL_NUM_QUERY_VARIATIONS: int = Field(
        default=3,
        description="Number of multi-query variations to generate",
        ge=1,
        le=10,
    )
    CLASSICAL_RELEVANCE_THRESHOLD: float = Field(
        default=5.0,
        description="Minimum LLM judge score (0-10) to include a chunk",
        ge=0.0,
        le=10.0,
    )
    CLASSICAL_TABLE_PREFIX: str = Field(
        default="classical_rag_", description="Prefix for PGVectorStore table names"
    )
    CLASSICAL_LLM_TEMPERATURE: float = Field(
        default=0.0,
        description="Temperature for LLM calls (multi-query + judge)",
        ge=0.0,
        le=2.0,
    )
    CLASSICAL_RRF_K: int = Field(
        default=60,
        ge=1,
        description="RRF constant K for hybrid BM25+vector search",
    )


class MinioConfig(BaseSettings):
    """MinIO object storage configuration."""

    MINIO_HOST: str = Field(default="localhost:9000")
    MINIO_ACCESS: str = Field(default="minioadmin")
    MINIO_SECRET: str = Field(default="minioadmin")
    MINIO_BUCKET: str = Field(default="raganything")
    MINIO_SECURE: bool = Field(default=False)
