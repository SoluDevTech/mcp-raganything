"""Dependency injection — module-level singletons following the pickpro pattern.

Per-user LLM credentials (chat + embeddings) are resolved at request time from
the ``current_user_id`` RLS contextvar (set by ``verify_credentials`` after a
successful dual-auth). When the contextvar is set, the factories
``get_embedding_for_user`` / ``get_chat_llm_for_user`` / ``get_vector_store_for_user``
resolve the user's ``(base_url, api_key)`` from the shared ``user_llm_settings``
table (read-only, owned by composable-agents) via the
:class:`UserLlmCredentialResolver`. When the user has no configured provider,
the factories raise :class:`LlmNotConfiguredError` (422). When the contextvar
is None (legacy/tests, no auth wired), the factories fall back to the
:class:`LLMConfig` env-based credentials — the original behaviour — so all
existing tests keep passing.
"""

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
from domain.errors.llm import LlmNotConfiguredError
from domain.errors.messages import ErrorMessage
from domain.ports.auth.api_key_repository import ApiKeyRepositoryPort
from domain.ports.auth.jwt_service import JwtServicePort
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.llm_port import LLMPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory
from domain.ports.user_llm_settings_repository import UserLlmSettingsRepositoryPort
from domain.ports.vector_store_port import VectorStorePort
from domain.services.auth.auth_service import AuthService
from domain.services.llm.credential_resolver import UserLlmCredentialResolver
from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter
from infrastructure.database.rls_context import current_user_id
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

# --- Classical RAG: legacy singletons (env-based, used when no auth wired) ---
#
# These are kept as fallbacks for the legacy/tests path (``current_user_id``
# is None). The per-request factories below (``get_embedding_for_user`` etc.)
# use them when the RLS contextvar is not set, so existing tests and the dev
# path keep working unchanged. When the contextvar IS set, the factories
# resolve per-user credentials and build fresh per-user instances.
classical_vector_store: VectorStorePort | None = None
_legacy_embedding = None
_legacy_llm: LLMPort | None = None
try:
    from langchain_openai import OpenAIEmbeddings

    _legacy_embedding = OpenAIEmbeddings(
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
        embedding_service=_legacy_embedding,
    )
    from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

    _legacy_llm = LangchainOpenAIAdapter(
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        model=llm_config.CHAT_MODEL,
        temperature=classical_rag_config.CLASSICAL_LLM_TEMPERATURE,
    )
except Exception as e:
    logger.warning("Classical RAG adapter initialization failed: %s", e)


# --- Per-user LLM credential resolution ----------------------------------------
#
# The reader and resolver are wired lazily because they depend on the asyncpg
# pool (created in ``main.lifespan``). ``main.py`` assigns ``user_llm_reader``
# once the pool exists, then the resolver is built on first access.
user_llm_reader: UserLlmSettingsRepositoryPort | None = None


def get_user_llm_reader() -> UserLlmSettingsRepositoryPort:
    """Return the user LLM settings reader singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the reader has not been wired yet.
    """
    if user_llm_reader is None:
        raise DependencyNotInitializedError(
            "User LLM settings reader is not initialized; the application "
            "lifespan must create the asyncpg pool before serving requests."
        )
    return user_llm_reader


def get_user_llm_resolver() -> UserLlmCredentialResolver:
    """Return the per-user LLM credential resolver (built from the reader)."""
    return UserLlmCredentialResolver(repo=get_user_llm_reader())


# --- Per-user LLM/embeddings/vector-store factories ---------------------------
#
# These async factories replace the module-level singletons for the per-user
# path. They read ``current_user_id`` from the RLS contextvar:
# - set + user has settings → build per-user instance with the user's creds;
# - set + no settings → raise ``LlmNotConfiguredError`` (422);
# - None → fall back to the legacy env-based singletons (tests/dev).


async def get_embedding_for_user():
    """Build an ``OpenAIEmbeddings`` for the current user (or legacy fallback).

    Returns:
        An ``OpenAIEmbeddings`` instance bound to the user's credentials or
        the env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based embedding singleton was not built at import time.
    """
    from langchain_openai import OpenAIEmbeddings

    user_id = current_user_id.get()
    if user_id is None:
        if _legacy_embedding is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
            )
        return _legacy_embedding

    resolver = get_user_llm_resolver()
    creds = await resolver.resolve(user_id)
    if creds is None:
        raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
    base_url, api_key = creds
    return OpenAIEmbeddings(
        model=llm_config.EMBEDDING_MODEL,
        api_key=api_key,
        base_url=base_url,
    )


async def get_chat_llm_for_user() -> LLMPort:
    """Build a ``LangchainOpenAIAdapter`` for the current user (or legacy fallback).

    Returns:
        A ``LangchainOpenAIAdapter`` wrapping ``ChatOpenAI`` bound to the
        user's credentials or the env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based LLM singleton was not built at import time.
    """
    from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

    user_id = current_user_id.get()
    if user_id is None:
        if _legacy_llm is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR_LLM
            )
        return _legacy_llm

    resolver = get_user_llm_resolver()
    creds = await resolver.resolve(user_id)
    if creds is None:
        raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
    base_url, api_key = creds
    return LangchainOpenAIAdapter(
        api_key=api_key,
        base_url=base_url,
        model=llm_config.CHAT_MODEL,
        temperature=classical_rag_config.CLASSICAL_LLM_TEMPERATURE,
    )


async def get_vector_store_for_user() -> VectorStorePort:
    """Build a ``LangchainPgvectorAdapter`` for the current user (or legacy fallback).

    The per-user adapter reuses the legacy singleton's ``PGEngine`` connection
    pool (so we don't create a pool per request) but binds a per-user
    ``OpenAIEmbeddings`` to the ``PGVectorStore`` created for each working dir.

    Returns:
        A ``LangchainPgvectorAdapter`` bound to the user's embeddings or the
        env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based vector store singleton was not built at import time.
    """
    from infrastructure.vector_store.langchain_pgvector_adapter import (
        LangchainPgvectorAdapter,
    )

    user_id = current_user_id.get()
    if user_id is None:
        if classical_vector_store is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
            )
        return classical_vector_store

    # Per-user path: reuse the shared PGEngine from the legacy singleton so we
    # don't create a connection pool per request. The embeddings are per-user.
    embedding = await get_embedding_for_user()
    shared_engine = (
        classical_vector_store._engine if classical_vector_store is not None else None
    )
    return LangchainPgvectorAdapter(
        connection_string=db_config.DATABASE_URL,
        table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
        embedding_dimension=llm_config.EMBEDDING_DIM,
        embedding_service=embedding,
        engine=shared_engine,
    )


async def get_classical_index_file_use_case() -> ClassicalIndexFileUseCase:
    vector_store = await get_vector_store_for_user()
    return ClassicalIndexFileUseCase(
        vector_store=vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


async def get_classical_index_folder_use_case() -> ClassicalIndexFolderUseCase:
    vector_store = await get_vector_store_for_user()
    return ClassicalIndexFolderUseCase(
        vector_store=vector_store,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


async def get_classical_query_use_case() -> ClassicalQueryUseCase:
    vector_store = await get_vector_store_for_user()
    llm = await get_chat_llm_for_user()
    return ClassicalQueryUseCase(
        vector_store=vector_store,
        llm=llm,
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


async def get_delete_file_use_case() -> DeleteFileUseCase:
    """Build the delete-file use case with an OPTIONAL per-user vector store.

    The vector store is only used to cascade-delete pgvector chunks for the
    file. When the authenticated user has no configured LLM provider
    (``user_llm_settings`` row absent), the per-user vector store cannot be
    built (it needs the user's embeddings). In that case we pass
    ``vector_store=None`` so the MinIO delete still proceeds — the pgvector
    cascade is skipped (the chunks, if any, are orphaned, but the user can
    still manage their files). This avoids a 422 on a simple MinIO delete for
    users without LLM credentials.
    """
    vector_store = None
    try:
        vector_store = await get_vector_store_for_user()
    except LlmNotConfiguredError:
        logger.warning(
            "Per-user vector store unavailable for delete-file (no LLM "
            "configured); pgvector cascade will be skipped."
        )
    return DeleteFileUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=vector_store,
    )


async def get_delete_folder_use_case() -> DeleteFolderUseCase:
    """Build the delete-folder use case with an OPTIONAL per-user vector store.

    See ``get_delete_file_use_case`` for the rationale (cascade-delete is
    best-effort; MinIO delete must still work without LLM credentials).
    """
    vector_store = None
    try:
        vector_store = await get_vector_store_for_user()
    except LlmNotConfiguredError:
        logger.warning(
            "Per-user vector store unavailable for delete-folder (no LLM "
            "configured); pgvector cascade will be skipped."
        )
    return DeleteFolderUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=vector_store,
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


# --- Dual auth (JWT + per-user API key) ---------------------------------------
#
# The JWT adapter (JWKS fetch over httpx) and the API-key reader (asyncpg read
# of the shared ``api_keys`` table owned by composable-agents) are wired
# lazily because they depend on resources that are only available after the
# application starts:
# - the JwtAdapter needs no external resource (it creates its own httpx client
#   and caches the JWKS), but its JWKS URL is derived from ``LOGTO_URL``;
# - the AsyncpgApiKeyReader needs the asyncpg pool (created in ``main.lifespan``).
# ``main.py`` assigns these module-level singletons once the resources exist,
# then calls ``security.set_auth_service(AuthService(jwt, reader))`` so the
# dual-auth dependency is engaged on protected routers and the MCP middleware.

jwt_adapter: JwtServicePort | None = None
api_key_reader: ApiKeyRepositoryPort | None = None
auth_service: AuthService | None = None


def get_jwt_adapter() -> JwtServicePort:
    """Return the JWT adapter singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the adapter has not been wired yet.
    """
    if jwt_adapter is None:
        raise DependencyNotInitializedError(
            "JWT adapter is not initialized; the application lifespan must "
            "create it before serving requests."
        )
    return jwt_adapter


def get_api_key_reader() -> ApiKeyRepositoryPort:
    """Return the API-key reader singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the reader has not been wired yet.
    """
    if api_key_reader is None:
        raise DependencyNotInitializedError(
            "API key reader is not initialized; the application lifespan must "
            "create the asyncpg pool before serving requests."
        )
    return api_key_reader


def get_auth_service() -> AuthService:
    """Return the dual-auth :class:`AuthService` singleton.

    Raises:
        DependencyNotInitializedError: If the service has not been wired yet.
    """
    if auth_service is None:
        raise DependencyNotInitializedError(
            "Auth service is not initialized; the application lifespan must "
            "wire the JwtAdapter and AsyncpgApiKeyReader before serving requests."
        )
    return auth_service
