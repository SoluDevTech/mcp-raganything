"""Dependency injection — module-level singletons following the pickpro pattern."""

import os

from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from application.use_cases.classical_query_use_case import ClassicalQueryUseCase
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.liveness_check_use_case import LivenessCheckUseCase
from application.use_cases.multimodal_query_use_case import MultimodalQueryUseCase
from application.use_cases.query_use_case import QueryUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from config import (
    AppConfig,
    BM25Config,
    ClassicalRAGConfig,
    DatabaseConfig,
    LLMConfig,
    MinioConfig,
    RAGConfig,
)
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.llm_port import LLMPort
from domain.ports.vector_store_port import VectorStorePort
from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter
from infrastructure.document_reader.kreuzberg_adapter import KreuzbergAdapter
from infrastructure.rag.classical_bm25_adapter import ClassicalBM25Adapter
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from infrastructure.rag.pg_textsearch_adapter import PostgresBM25Adapter
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

kreuzberg_adapter = KreuzbergAdapter()
postgres_health_adapter = AsyncpgHealthAdapter(db_config)

classical_rag_config = ClassicalRAGConfig()

classical_bm25_adapter: BM25EnginePort | None = None
if bm25_config.BM25_ENABLED:
    try:
        classical_bm25_adapter = ClassicalBM25Adapter(
            db_url=db_config.DATABASE_URL.replace("+asyncpg", ""),
            table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
            text_config=bm25_config.BM25_TEXT_CONFIG,
        )
    except Exception as e:
        print(f"WARNING: Classical BM25 adapter initialization failed: {e}")
        classical_bm25_adapter = None

classical_vector_store: VectorStorePort | None = None
classical_llm: LLMPort | None = None
try:
    from langchain_openai import OpenAIEmbeddings

    _embedding = OpenAIEmbeddings(
        model=llm_config.EMBEDDING_MODEL,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
    )
    from infrastructure.vector_store.langchain_pgvector_adapter import (
        LangchainPgvectorAdapter,
    )

    classical_vector_store = LangchainPgvectorAdapter(
        connection_string=db_config.DATABASE_URL,
        table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
        embedding_dimension=llm_config.EMBEDDING_DIM,
        embedding_service=_embedding,
    )
    from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

    classical_llm = LangchainOpenAIAdapter(
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        model=llm_config.CHAT_MODEL,
        temperature=classical_rag_config.CLASSICAL_LLM_TEMPERATURE,
    )
except Exception as e:
    print(f"WARNING: Classical RAG adapter initialization failed: {e}")


def get_classical_index_file_use_case() -> ClassicalIndexFileUseCase:
    if classical_vector_store is None:
        raise RuntimeError("Classical RAG unavailable: vector store not initialized")
    return ClassicalIndexFileUseCase(
        vector_store=classical_vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_classical_index_folder_use_case() -> ClassicalIndexFolderUseCase:
    if classical_vector_store is None:
        raise RuntimeError("Classical RAG unavailable: vector store not initialized")
    return ClassicalIndexFolderUseCase(
        vector_store=classical_vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_classical_query_use_case() -> ClassicalQueryUseCase:
    if classical_vector_store is None or classical_llm is None:
        raise RuntimeError(
            "Classical RAG unavailable: vector store or LLM not initialized"
        )
    return ClassicalQueryUseCase(
        vector_store=classical_vector_store,
        llm=classical_llm,
        config=classical_rag_config,
        bm25_engine=classical_bm25_adapter,
        rrf_k=classical_rag_config.CLASSICAL_RRF_K,
    )


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


def get_list_files_use_case() -> ListFilesUseCase:
    return ListFilesUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_list_folders_use_case() -> ListFoldersUseCase:
    return ListFoldersUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_read_file_use_case() -> ReadFileUseCase:
    return ReadFileUseCase(
        storage=minio_adapter,
        document_reader=kreuzberg_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_upload_file_use_case() -> UploadFileUseCase:
    return UploadFileUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_liveness_check_use_case() -> LivenessCheckUseCase:
    return LivenessCheckUseCase(
        storage=minio_adapter,
        postgres_health=postgres_health_adapter,
        bucket=minio_config.MINIO_BUCKET,
    )
