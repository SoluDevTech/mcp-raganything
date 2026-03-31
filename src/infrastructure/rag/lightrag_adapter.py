import hashlib
import os
import time
from typing import Literal, cast

from fastapi.logger import logger
from lightrag import QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

from config import LLMConfig, RAGConfig
from domain.entities.indexing_result import (
    FileIndexingResult,
    FileProcessingDetail,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)
from domain.ports.rag_engine import RAGEnginePort

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

_POSTGRES_STORAGE = {
    "kv_storage": "PGKVStorage",
    "vector_storage": "PGVectorStorage",
    "graph_storage": "PGGraphStorage",
    "doc_status_storage": "PGDocStatusStorage",
}

_LOCAL_STORAGE = {
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "JsonDocStatusStorage",
}


class LightRAGAdapter(RAGEnginePort):
    """Adapter for RAGAnything/LightRAG implementing RAGEnginePort."""

    def __init__(self, llm_config: LLMConfig, rag_config: RAGConfig) -> None:
        self._llm_config = llm_config
        self._rag_config = rag_config
        self.rag: dict[str,RAGAnything] = {}

    @staticmethod
    def _make_workspace(working_dir: str) -> str:
        """Create a short, AGE-safe workspace name from the working_dir.

        Apache AGE graph names must be valid PostgreSQL identifiers
        (alphanumeric + underscore, max 63 chars). We use a truncated
        SHA-256 hash prefixed with 'ws_' to guarantee uniqueness and
        compliance.
        """
        digest = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"ws_{digest}"

    def init_project(self, working_dir: str) -> RAGAnything:
        if working_dir in self.rag:
            return self.rag[working_dir]
        workspace = self._make_workspace(working_dir)
        self.rag[working_dir] = RAGAnything(
            config=RAGAnythingConfig(
                working_dir=working_dir,
                parser="docling",
                parse_method="txt",
                enable_image_processing=self._rag_config.ENABLE_IMAGE_PROCESSING,
                enable_table_processing=self._rag_config.ENABLE_TABLE_PROCESSING,
                enable_equation_processing=self._rag_config.ENABLE_EQUATION_PROCESSING,
                max_concurrent_files=self._rag_config.MAX_CONCURRENT_FILES,
            ),
            llm_model_func=self._llm_call,
            vision_model_func=self._vision_call,
            embedding_func=EmbeddingFunc(
                embedding_dim=self._llm_config.EMBEDDING_DIM,
                max_token_size=self._llm_config.MAX_TOKEN_SIZE,
                func=lambda texts: openai_embed(
                    texts,
                    model=self._llm_config.EMBEDDING_MODEL,
                    api_key=self._llm_config.api_key,
                    base_url=self._llm_config.api_base_url,
                ),
            ),
            lightrag_kwargs={
                **(
                    _POSTGRES_STORAGE
                    if self._rag_config.RAG_STORAGE_TYPE == "postgres"
                    else _LOCAL_STORAGE
                ),
                "cosine_threshold": self._rag_config.COSINE_THRESHOLD,
                "workspace": workspace,
            },
        )
        return self.rag[working_dir]
    # ------------------------------------------------------------------
    # LLM callables (passed directly to RAGAnything)
    # ------------------------------------------------------------------

    async def _llm_call(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        if history_messages is None:
            history_messages = []
        return await openai_complete_if_cache(
            self._llm_config.CHAT_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self._llm_config.api_key,
            base_url=self._llm_config.api_base_url,
            **kwargs,
        )

    async def _vision_call(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        messages = _build_vision_messages(
            system_prompt, history_messages, prompt, image_data
        )
        return await openai_complete_if_cache(
            self._llm_config.VISION_MODEL,
            "Image Description Task",
            system_prompt=None,
            history_messages=messages,
            api_key=self._llm_config.api_key,
            base_url=self._llm_config.api_base_url,
            messages=messages,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Port implementation — indexing
    # ------------------------------------------------------------------

    def _ensure_initialized(self, working_dir: str) -> RAGAnything:
        if self.rag[working_dir] is None:
            raise RuntimeError("RAG engine not initialized. Call init_project() first.")
        return self.rag[working_dir]

    async def index_document(
        self, file_path: str, file_name: str, output_dir: str, working_dir: str = ""
    ) -> FileIndexingResult:
        start_time = time.time()
        rag = self._ensure_initialized(working_dir)
        await rag._ensure_lightrag_initialized()
        try:
            await rag.process_document_complete(
                file_path=file_path, output_dir=output_dir, parse_method="txt"
            )
            processing_time_ms = (time.time() - start_time) * 1000
            return FileIndexingResult(
                status=IndexingStatus.SUCCESS,
                message=f"File '{file_name}' indexed successfully",
                file_path=file_path,
                file_name=file_name,
                processing_time_ms=round(processing_time_ms, 2),
            )
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to index document {file_path}: {e}", exc_info=True)
            return FileIndexingResult(
                status=IndexingStatus.FAILED,
                message=f"Failed to index file '{file_name}'",
                file_path=file_path,
                file_name=file_name,
                processing_time_ms=round(processing_time_ms, 2),
                error=str(e),
            )

    async def index_folder(
        self,
        folder_path: str,
        output_dir: str,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        working_dir: str = "",
    ) -> FolderIndexingResult:
        start_time = time.time()
        rag = self._ensure_initialized(working_dir)
        await rag._ensure_lightrag_initialized()
        try:
            result = await rag.process_folder_complete(
                folder_path=folder_path,
                output_dir=output_dir,
                parse_method="txt",
                file_extensions=file_extensions,
                recursive=recursive,
                display_stats=True,
                max_workers=self._rag_config.MAX_WORKERS,
            )
            processing_time_ms = (time.time() - start_time) * 1000
            return self._build_folder_result(
                result, folder_path, recursive, processing_time_ms
            )
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to index folder {folder_path}: {e}", exc_info=True)
            return FolderIndexingResult(
                status=IndexingStatus.FAILED,
                message=f"Failed to index folder '{folder_path}'",
                folder_path=folder_path,
                recursive=recursive,
                stats=FolderIndexingStats(),
                processing_time_ms=round(processing_time_ms, 2),
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Port implementation — query
    # ------------------------------------------------------------------

    async def query(self, query: str, mode: str = "naive", top_k: int = 10, working_dir: str = "") -> dict:
        rag = self._ensure_initialized(working_dir)
        await rag._ensure_lightrag_initialized()
        if rag.lightrag is None:
            return {
                "status": "failure",
                "message": "RAG engine not initialized",
                "data": {},
            }
        param = QueryParam(mode=cast(QueryMode, mode), top_k=top_k, chunk_top_k=top_k)
        return await rag.lightrag.aquery_data(query=query, param=param)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_folder_result(
        result, folder_path: str, recursive: bool, processing_time_ms: float
    ) -> FolderIndexingResult:
        result_dict = result if isinstance(result, dict) else {}
        stats = FolderIndexingStats(
            total_files=result_dict.get("total_files", 0),
            files_processed=result_dict.get("successful_files", 0),
            files_failed=result_dict.get("failed_files", 0),
            files_skipped=result_dict.get("skipped_files", 0),
        )

        file_results = _parse_file_details(result_dict)

        if stats.files_failed == 0 and stats.files_processed > 0:
            status = IndexingStatus.SUCCESS
            message = f"Successfully indexed {stats.files_processed} file(s) from '{folder_path}'"
        elif stats.files_processed > 0 and stats.files_failed > 0:
            status = IndexingStatus.PARTIAL
            message = f"Partially indexed folder '{folder_path}': {stats.files_processed} succeeded, {stats.files_failed} failed"
        elif stats.files_processed == 0 and stats.total_files > 0:
            status = IndexingStatus.FAILED
            message = f"Failed to index any files from '{folder_path}'"
        else:
            status = IndexingStatus.SUCCESS
            message = f"No files found to index in '{folder_path}'"

        return FolderIndexingResult(
            status=status,
            message=message,
            folder_path=folder_path,
            recursive=recursive,
            stats=stats,
            processing_time_ms=round(processing_time_ms, 2),
            file_results=file_results,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_vision_messages(
    system_prompt: str | None,
    history_messages: list,
    prompt: str,
    image_data,
) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_data:
        images = image_data if isinstance(image_data, list) else [image_data]
        for img in images:
            url = (
                img
                if isinstance(img, str) and img.startswith("http")
                else f"data:image/jpeg;base64,{img}"
            )
            content.append({"type": "image_url", "image_url": {"url": url}})

    messages.append({"role": "user", "content": content})
    return messages


def _parse_file_details(result_dict: dict) -> list[FileProcessingDetail] | None:
    if "file_details" not in result_dict:
        return None
    file_details = result_dict["file_details"]
    if not isinstance(file_details, list):
        return None
    return [
        FileProcessingDetail(
            file_path=d.get("file_path", ""),
            file_name=os.path.basename(d.get("file_path", "")),
            status=IndexingStatus.SUCCESS
            if d.get("success", False)
            else IndexingStatus.FAILED,
            error=d.get("error"),
        )
        for d in file_details
    ]
