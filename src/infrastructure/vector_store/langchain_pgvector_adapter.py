import hashlib
import logging
import uuid

from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import text

from domain.errors.messages import ErrorMessage
from domain.errors.vector_store import VectorStoreConfigError
from domain.ports.vector_store_port import SearchResult, VectorStorePort

logger = logging.getLogger(__name__)


class LangchainPgvectorAdapter(VectorStorePort):
    """PGVector-backed vector store adapter using langchain-postgres.

    Manages per-working-dir vector tables with SHA-256-hashed names. Lazily
    initializes the PGEngine and PGVectorStore on first access for each
    working directory.
    """

    def __init__(
        self,
        connection_string: str,
        table_prefix: str,
        embedding_dimension: int,
        embedding_service=None,
    ):
        self._connection_string = connection_string
        self._table_prefix = table_prefix
        self._embedding_dimension = embedding_dimension
        self._embedding_service = embedding_service
        self._engine = None
        self._stores = {}

    def _get_table_name(self, working_dir: str) -> str:
        hashed = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"{self._table_prefix}{hashed}"

    async def ensure_table(self, working_dir: str) -> None:
        if self._engine is None:
            self._engine = PGEngine.from_connection_string(self._connection_string)

        table_name = self._get_table_name(working_dir)

        if working_dir not in self._stores:
            try:
                await self._engine.ainit_vectorstore_table(
                    table_name=table_name,
                    vector_size=self._embedding_dimension,
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
            store = await PGVectorStore.create(
                engine=self._engine,
                table_name=table_name,
                embedding_service=self._embedding_service,
            )
            self._stores[working_dir] = store

    async def add_documents(
        self,
        working_dir: str,
        documents: list[tuple[str, str, dict[str, str | int | None]]],
    ) -> list[str]:
        store = self._stores.get(working_dir)
        if store is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        lc_docs = []
        ids = []
        for content, file_path, metadata in documents:
            doc_id = str(uuid.uuid4())
            meta = {"file_path": file_path, "chunk_id": doc_id, **metadata}
            lc_docs.append(Document(id=doc_id, page_content=content, metadata=meta))
            ids.append(doc_id)

        await store.aadd_documents(lc_docs, ids=ids)
        return ids

    async def similarity_search(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        store = self._stores.get(working_dir)
        if store is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        lc_results = await store.asimilarity_search_with_score(query, k=top_k)

        filtered = lc_results
        if score_threshold is not None:
            filtered = [
                (doc, score) for doc, score in lc_results if score <= score_threshold
            ]

        return [
            SearchResult(
                chunk_id=str(doc.metadata.get("chunk_id", doc.id)),
                content=doc.page_content,
                file_path=doc.metadata.get("file_path", ""),
                score=float(score),
                metadata={
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in ("chunk_id", "file_path")
                },
            )
            for doc, score in filtered
        ]

    async def delete_documents(self, working_dir: str, file_path: str) -> int:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        table_name = self._get_table_name(working_dir)
        stmt = text(
            f'DELETE FROM "{table_name}" '
            f"WHERE langchain_metadata->>'file_path' = :file_path"
        )
        logger.info(
            "Deleting vectors by file_path: table=%s file_path=%s",
            table_name,
            file_path,
        )

        # langchain-postgres PGEngine._pool and _run_as_async are private but
        # the only way to execute raw SQL. Pin langchain-postgres in pyproject.toml.
        async def _delete() -> int:
            async with self._engine._pool.connect() as conn:
                result = await conn.execute(stmt, {"file_path": file_path})
                await conn.commit()
                return result.rowcount

        count = await self._engine._run_as_async(_delete())
        logger.info(
            "Vectors deleted by file_path: table=%s count=%d", table_name, count
        )
        return count

    async def delete_by_prefix(self, working_dir: str, file_path_prefix: str) -> int:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        table_name = self._get_table_name(working_dir)
        stmt = text(
            f'DELETE FROM "{table_name}" '
            f"WHERE starts_with(langchain_metadata->>'file_path', :prefix)"
        )
        logger.info(
            "Deleting vectors by prefix: table=%s prefix=%s",
            table_name,
            file_path_prefix,
        )

        async def _delete() -> int:
            async with self._engine._pool.connect() as conn:
                result = await conn.execute(stmt, {"prefix": file_path_prefix})
                await conn.commit()
                return result.rowcount

        count = await self._engine._run_as_async(_delete())
        logger.info("Vectors deleted by prefix: table=%s count=%d", table_name, count)
        return count

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None
