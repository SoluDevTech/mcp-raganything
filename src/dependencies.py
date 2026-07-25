"""Dependency injection — module-level singletons following the pickpro pattern."""

import logging
import os

from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from application.use_cases.classical_query_use_case import ClassicalQueryUseCase
from application.use_cases.create_folder_use_case import CreateFolderUseCase
from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_file_use_case import DeleteFileUseCase
from application.use_cases.delete_folder_use_case import DeleteFolderUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.liveness_check_use_case import LivenessCheckUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from config import (
    AppConfig,
    BM25Config,
    ClassicalRAGConfig,
    DatabaseConfig,
    LLMConfig,
    MinioConfig,
)
from domain.errors.config import DependencyNotInitializedError
from domain.errors.messages import ErrorMessage
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.llm_port import LLMPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory
from domain.ports.vector_store_port import VectorStorePort
from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter
from infrastructure.document_reader.kreuzberg_adapter import KreuzbergAdapter
from infrastructure.mcp.tool_loader import FastMcpToolLoader
from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory
from infrastructure.rag.classical_bm25_adapter import ClassicalBM25Adapter
from infrastructure.security.crypto import FernetSecretCipher
from infrastructure.storage.minio_adapter import MinioAdapter

logger = logging.getLogger(__name__)

app_config = AppConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
minio_config = MinioConfig()  # type: ignore
bm25_config = BM25Config()  # type: ignore
db_config = DatabaseConfig()  # type: ignore

os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

minio_adapter = MinioAdapter(
    host=minio_config.MINIO_HOST,
    access=minio_config.MINIO_ACCESS,
    secret=minio_config.MINIO_SECRET,
    secure=minio_config.MINIO_SECURE,
)

kreuzberg_adapter = KreuzbergAdapter()
postgres_health_adapter = AsyncpgHealthAdapter(db_config)

classical_rag_config = ClassicalRAGConfig()

classical_bm25_adapter: BM25EnginePort | None = None
if bm25_config.BM25_ENABLED:
    try:
        classical_bm25_adapter = ClassicalBM25Adapter(
            db_url=db_config.asyncpg_url,
            table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
            text_config=bm25_config.BM25_TEXT_CONFIG,
        )
    except Exception as e:
        logger.warning("Classical BM25 adapter initialization failed: %s", e)
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
    logger.warning("Classical RAG adapter initialization failed: %s", e)


def get_classical_index_file_use_case() -> ClassicalIndexFileUseCase:
    if classical_vector_store is None:
        raise DependencyNotInitializedError(
            ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
        )
    return ClassicalIndexFileUseCase(
        vector_store=classical_vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_classical_index_folder_use_case() -> ClassicalIndexFolderUseCase:
    if classical_vector_store is None:
        raise DependencyNotInitializedError(
            ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
        )
    return ClassicalIndexFolderUseCase(
        vector_store=classical_vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_classical_query_use_case() -> ClassicalQueryUseCase:
    if classical_vector_store is None or classical_llm is None:
        raise DependencyNotInitializedError(
            ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR_LLM
        )
    return ClassicalQueryUseCase(
        vector_store=classical_vector_store,
        llm=classical_llm,
        config=classical_rag_config,
        bm25_engine=classical_bm25_adapter,
        rrf_k=classical_rag_config.CLASSICAL_RRF_K,
    )


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


def get_create_folder_use_case() -> CreateFolderUseCase:
    return CreateFolderUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_delete_file_use_case() -> DeleteFileUseCase:
    return DeleteFileUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=classical_vector_store,
    )


def get_delete_folder_use_case() -> DeleteFolderUseCase:
    return DeleteFolderUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=classical_vector_store,
    )


def get_liveness_check_use_case() -> LivenessCheckUseCase:
    return LivenessCheckUseCase(
        storage=minio_adapter,
        postgres_health=postgres_health_adapter,
        bucket=minio_config.MINIO_BUCKET,
    )


# --- MCP server registry -----------------------------------------------------
#
# The store and runner are wired lazily because they depend on resources that
# are only available after the application starts:
# - the store needs an asyncpg pool (created in ``main.lifespan``);
# - the runner needs the FastAPI ``app`` instance (created in ``main`` after
#   importing this module).
# ``main.py`` assigns these module-level singletons once the resources exist.
# The cipher is created lazily from ``SECRET_ENCRYPTION_KEY`` so importing this
# module never fails when the key is unset (tests/dev override the providers).

mcp_openapi_factory: OpenApiMcpFactory = FastMcpOpenApiFactory(
    max_spec_bytes=app_config.OPENAPI_MAX_SPEC_BYTES
)
mcp_secret_cipher: FernetSecretCipher | None = None
mcp_registry_store: McpServerRegistryStore | None = None
generated_mcp_runner: GeneratedMcpRunnerPort | None = None


def get_mcp_secret_cipher() -> FernetSecretCipher:
    """Return the Fernet secret cipher singleton.

    Created lazily from ``SECRET_ENCRYPTION_KEY``. Raises if the key is unset.

    Returns:
        The FernetSecretCipher instance.

    Raises:
        DependencyNotInitializedError: If ``SECRET_ENCRYPTION_KEY`` is empty.
    """
    global mcp_secret_cipher
    if mcp_secret_cipher is None:
        if not app_config.SECRET_ENCRYPTION_KEY:
            raise DependencyNotInitializedError(
                "SECRET_ENCRYPTION_KEY is not set; MCP registry secrets cannot "
                "be encrypted. Set SECRET_ENCRYPTION_KEY in the environment."
            )
        mcp_secret_cipher = FernetSecretCipher(app_config.SECRET_ENCRYPTION_KEY)
    return mcp_secret_cipher


def get_mcp_openapi_factory() -> OpenApiMcpFactory:
    """Return the OpenAPI MCP factory singleton."""
    return mcp_openapi_factory


def get_mcp_registry_store() -> McpServerRegistryStore:
    """Return the MCP registry store singleton.

    Returns:
        The McpRegistryStore instance (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the store has not been wired yet.
    """
    if mcp_registry_store is None:
        raise DependencyNotInitializedError(
            "MCP registry store is not initialized; the application lifespan "
            "must create the asyncpg pool before serving requests."
        )
    return mcp_registry_store


def get_generated_mcp_runner() -> GeneratedMcpRunnerPort:
    """Return the generated MCP runner singleton.

    Returns:
        The GeneratedMcpRunner instance (wired by ``main`` after app creation).

    Raises:
        DependencyNotInitializedError: If the runner has not been wired yet.
    """
    if generated_mcp_runner is None:
        raise DependencyNotInitializedError(
            "Generated MCP runner is not initialized; the application must "
            "create it after the FastAPI app is built."
        )
    return generated_mcp_runner


def get_mcp_tool_loader() -> McpToolLoader:
    """Return the concrete fastmcp-based MCP tool loader.

    Returns:
        The FastMcpToolLoader instance configured with ``MCP_TOOL_TIMEOUT``.
    """
    return FastMcpToolLoader(tool_timeout=app_config.MCP_TOOL_TIMEOUT)


def get_create_mcp_server_use_case() -> CreateMcpServerUseCase:
    return CreateMcpServerUseCase(
        store=get_mcp_registry_store(),
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
        runner=get_generated_mcp_runner(),
    )


def get_list_mcp_servers_use_case() -> ListMcpServersUseCase:
    return ListMcpServersUseCase(store=get_mcp_registry_store())


def get_get_mcp_server_use_case() -> GetMcpServerUseCase:
    return GetMcpServerUseCase(store=get_mcp_registry_store())


def get_update_mcp_server_use_case() -> UpdateMcpServerUseCase:
    return UpdateMcpServerUseCase(
        store=get_mcp_registry_store(),
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
        runner=get_generated_mcp_runner(),
    )


def get_delete_mcp_server_use_case() -> DeleteMcpServerUseCase:
    return DeleteMcpServerUseCase(
        store=get_mcp_registry_store(),
        runner=get_generated_mcp_runner(),
    )


def get_validate_mcp_server_use_case() -> ValidateMcpServerUseCase:
    return ValidateMcpServerUseCase(
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
    )
