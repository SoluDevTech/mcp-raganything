import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, cast

from fastapi.logger import logger
from lightrag import QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.parser import register_parser

from application.requests.query_request import MultimodalContentItem
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


_BUILTIN_PARSERS = {"mineru", "paddleocr"}


def _ensure_parser_registered(parser_name: str) -> None:
    if parser_name == "kreuzberg":
        from infrastructure.rag.kreuzberg_raganything_parser import (
            KreuzbergRAGAnythingParser,
        )

        register_parser("kreuzberg", KreuzbergRAGAnythingParser)
        return

    if parser_name in _BUILTIN_PARSERS:
        return

    raise ValueError(
        f"Unknown document parser: {parser_name!r}. "
        f"Choose from: kreuzberg, {', '.join(sorted(_BUILTIN_PARSERS))}"
    )


class LightRAGAdapter(RAGEnginePort):
    """Adapter for RAGAnything/LightRAG implementing RAGEnginePort."""

    def __init__(self, llm_config: LLMConfig, rag_config: RAGConfig) -> None:
        self._llm_config = llm_config
        self._rag_config = rag_config
        self.rag: dict[str, RAGAnything] = {}

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

        # Capture config values as locals to avoid passing bound methods.
        # Bound methods reference `self` which holds `self.rag` (all previous
        # RAGAnything instances). RAGAnything/LightRAG deepcopies the callables
        # during init, which traverses the entire object graph including asyncpg
        # connections — and asyncpg objects are not picklable/copyable.
        llm_config = self._llm_config

        async def llm_call(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            return await openai_complete_if_cache(
                llm_config.CHAT_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                **kwargs,
            )

        async def vision_call(
            prompt, system_prompt=None, history_messages=None, image_data=None, **kwargs
        ):
            if history_messages is None:
                history_messages = []
            messages = _build_vision_messages(
                system_prompt, history_messages, prompt, image_data
            )
            return await openai_complete_if_cache(
                llm_config.VISION_MODEL,
                "Image Description Task",
                system_prompt=None,
                history_messages=messages,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                messages=messages,
                **kwargs,
            )

        safe_working_dir = os.path.join(
            tempfile.gettempdir(), "raganything", working_dir.strip("/")
        )
        parser_name = self._rag_config.DOCUMENT_PARSER
        _ensure_parser_registered(parser_name)
        self.rag[working_dir] = RAGAnything(
            config=RAGAnythingConfig(
                working_dir=safe_working_dir,
                parser=parser_name,
                parse_method="txt",
                enable_image_processing=self._rag_config.ENABLE_IMAGE_PROCESSING,
                enable_table_processing=self._rag_config.ENABLE_TABLE_PROCESSING,
                enable_equation_processing=self._rag_config.ENABLE_EQUATION_PROCESSING,
                max_concurrent_files=self._rag_config.MAX_CONCURRENT_FILES,
            ),
            llm_model_func=llm_call,
            vision_model_func=vision_call,
            embedding_func=EmbeddingFunc(
                embedding_dim=llm_config.EMBEDDING_DIM,
                max_token_size=llm_config.MAX_TOKEN_SIZE,
                func=lambda texts: openai_embed(
                    texts,
                    model=llm_config.EMBEDDING_MODEL,
                    api_key=llm_config.api_key,
                    base_url=llm_config.api_base_url,
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
    # Port implementation — indexing
    # ------------------------------------------------------------------

    def _ensure_initialized(self, working_dir: str) -> RAGAnything:
        rag = self.rag.get(working_dir)
        if rag is None:
            raise RuntimeError(
                f"RAG engine not initialized for '{working_dir}'. Call init_project() first."
            )
        return rag

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

    @staticmethod
    def _determine_folder_status(
        total: int, succeeded: int, failed: int, folder_path: str
    ) -> tuple[IndexingStatus, str]:
        if total == 0:
            return IndexingStatus.SUCCESS, f"No files found in '{folder_path}'"
        if failed == 0:
            return (
                IndexingStatus.SUCCESS,
                f"Successfully indexed {succeeded} file(s) from '{folder_path}'",
            )
        if succeeded == 0:
            return IndexingStatus.FAILED, f"Failed to index folder '{folder_path}'"
        return (
            IndexingStatus.PARTIAL,
            f"Partially indexed: {succeeded} succeeded, {failed} failed",
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

        glob_pattern = "**/*" if recursive else "*"
        folder = Path(folder_path)
        all_files = [f for f in folder.glob(glob_pattern) if f.is_file()]

        if file_extensions:
            exts = set(file_extensions)
            all_files = [f for f in all_files if f.suffix in exts]

        succeeded, failed, file_results = await self._process_files_concurrently(
            rag, all_files, output_dir
        )

        processing_time_ms = (time.time() - start_time) * 1000
        total = len(all_files)
        status, message = self._determine_folder_status(
            total, succeeded, failed, folder_path
        )

        return FolderIndexingResult(
            status=status,
            message=message,
            folder_path=folder_path,
            recursive=recursive,
            stats=FolderIndexingStats(
                total_files=total,
                files_processed=succeeded,
                files_failed=failed,
                files_skipped=0,
            ),
            file_results=file_results,
            processing_time_ms=round(processing_time_ms, 2),
        )

    async def _process_files_concurrently(
        self,
        rag: RAGAnything,
        all_files: list[Path],
        output_dir: str,
    ) -> tuple[int, int, list[FileProcessingDetail]]:
        max_workers = max(1, self._rag_config.MAX_CONCURRENT_FILES)
        semaphore = asyncio.Semaphore(max_workers)
        succeeded = 0
        failed = 0
        file_results: list[FileProcessingDetail] = []

        async def _process_file(file_path_obj: Path) -> None:
            nonlocal succeeded, failed
            async with semaphore:
                try:
                    await rag.process_document_complete(
                        file_path=str(file_path_obj),
                        output_dir=output_dir,
                        parse_method="txt",
                    )
                    succeeded += 1
                    file_results.append(
                        FileProcessingDetail(
                            file_path=str(file_path_obj),
                            file_name=file_path_obj.name,
                            status=IndexingStatus.SUCCESS,
                        )
                    )
                    logger.info(
                        f"Indexed {file_path_obj.name} ({succeeded}/{len(all_files)})"
                    )
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to index {file_path_obj.name}: {e}")
                    file_results.append(
                        FileProcessingDetail(
                            file_path=str(file_path_obj),
                            file_name=file_path_obj.name,
                            status=IndexingStatus.FAILED,
                            error=str(e),
                        )
                    )

        await asyncio.gather(*[_process_file(f) for f in all_files])
        return succeeded, failed, file_results

    # ------------------------------------------------------------------
    # Port implementation — query
    # ------------------------------------------------------------------

    async def query(
        self, query: str, mode: str = "naive", top_k: int = 10, working_dir: str = ""
    ) -> dict:
        rag = self._ensure_initialized(working_dir)
        await rag._ensure_lightrag_initialized()
        if rag.lightrag is None:
            return {
                "status": "failure",
                "message": "RAG engine not initialized",
                "data": {},
            }
        param = QueryParam(mode=cast(QueryMode, mode), top_k=top_k, chunk_top_k=top_k)
        result = await rag.lightrag.aquery_data(query=query, param=param)
        if isinstance(result.get("data"), dict):
            result["data"]["entities"] = []
            result["data"]["relationships"] = []
        return result

    async def query_multimodal(
        self,
        query: str,
        multimodal_content: list[MultimodalContentItem],
        mode: str = "hybrid",
        top_k: int = 10,
        working_dir: str = "",
    ) -> str:
        rag = self._ensure_initialized(working_dir)
        await rag._ensure_lightrag_initialized()
        raw_content = [
            item.model_dump(exclude_none=True) for item in multimodal_content
        ]
        return await rag.aquery_with_multimodal(
            query=query,
            multimodal_content=raw_content,
            mode=mode,
            top_k=top_k,
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
