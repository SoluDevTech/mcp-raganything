"""Dependency injection — module-level singletons following the pickpro pattern."""

import os

from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.multimodal_query_use_case import MultimodalQueryUseCase
from application.use_cases.query_use_case import QueryUseCase
from config import (
    AppConfig,
    BM25Config,
    DatabaseConfig,
    LLMConfig,
    MinioConfig,
    RAGConfig,
)
from domain.ports.bm25_engine import BM25EnginePort
from infrastructure.bm25.pg_textsearch_adapter import PostgresBM25Adapter
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from infrastructure.storage.minio_adapter import MinioAdapter

app_config = AppConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
rag_config = RAGConfig()  # type: ignore
minio_config = MinioConfig()  # type: ignore
bm25_config = BM25Config()  # type: ignore
db_config = DatabaseConfig()  # type: ignore

os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

rag_adapter = LightRAGAdapter(llm_config, rag_config)
minio_adapter = MinioAdapter(
    host=minio_config.MINIO_HOST,
    access=minio_config.MINIO_ACCESS,
    secret=minio_config.MINIO_SECRET,
    secure=minio_config.MINIO_SECURE,
)

bm25_adapter: BM25EnginePort | None = None
if bm25_config.BM25_ENABLED:
    try:
        bm25_adapter = PostgresBM25Adapter(
            db_url=db_config.DATABASE_URL.replace("+asyncpg", ""),
            text_config=bm25_config.BM25_TEXT_CONFIG,
        )
    except Exception as e:
        print(f"WARNING: BM25 adapter initialization failed: {e}")
        bm25_adapter = None


def get_index_file_use_case() -> IndexFileUseCase:
    return IndexFileUseCase(
        rag_adapter, minio_adapter, minio_config.MINIO_BUCKET, app_config.OUTPUT_DIR
    )


def get_index_folder_use_case() -> IndexFolderUseCase:
    return IndexFolderUseCase(
        rag_adapter, minio_adapter, minio_config.MINIO_BUCKET, app_config.OUTPUT_DIR
    )


def get_query_use_case() -> QueryUseCase:
    return QueryUseCase(
        rag_engine=rag_adapter,
        bm25_engine=bm25_adapter,
        rrf_k=bm25_config.BM25_RRF_K,
    )


def get_multimodal_query_use_case() -> MultimodalQueryUseCase:
    return MultimodalQueryUseCase(rag_adapter)
