"""Dependency injection — module-level singletons following the pickpro pattern."""

import os

from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.query_use_case import QueryUseCase
from config import AppConfig, LLMConfig, MinioConfig, RAGConfig
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from infrastructure.storage.minio_adapter import MinioAdapter

# ============= CONFIG =============

app_config = AppConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
rag_config = RAGConfig()  # type: ignore
minio_config = MinioConfig()  # type: ignore

os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

# ============= ADAPTERS =============

rag_adapter = LightRAGAdapter(llm_config, rag_config)
minio_adapter = MinioAdapter(
    host=minio_config.MINIO_HOST,
    access=minio_config.MINIO_ACCESS,
    secret=minio_config.MINIO_SECRET,
    secure=minio_config.MINIO_SECURE,
)

# ============= USE CASE PROVIDERS =============


def get_index_file_use_case() -> IndexFileUseCase:
    return IndexFileUseCase(
        rag_adapter, minio_adapter, minio_config.MINIO_BUCKET, app_config.OUTPUT_DIR
    )


def get_index_folder_use_case() -> IndexFolderUseCase:
    return IndexFolderUseCase(
        rag_adapter, minio_adapter, minio_config.MINIO_BUCKET, app_config.OUTPUT_DIR
    )


def get_query_use_case() -> QueryUseCase:
    return QueryUseCase(rag_adapter)
