import hashlib
import uuid

from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore

from domain.ports.vector_store_port import SearchResult, VectorStorePort


class LangchainPgvectorAdapter(VectorStorePort):
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
        self._id_maps = {}

    def _get_table_name(self, working_dir: str) -> str:
        hashed = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"{self._table_prefix}{hashed}"

    async def ensure_table(self, working_dir: str) -> None:
        if self._engine is None:
            self._engine = PGEngine.from_connection_string(self._connection_string)

        table_name = self._get_table_name(working_dir)

        if working_dir not in self._stores:
            store = PGVectorStore.create(
                engine=self._engine,
                table_name=table_name,
                embedding_service=self._embedding_service,
                embedding_dimension=self._embedding_dimension,
            )
            if hasattr(store, "__await__"):
                store = await store
            self._stores[working_dir] = store
            self._id_maps[working_dir] = {}

    async def add_documents(
        self,
        working_dir: str,
        documents: list[tuple[str, str, dict[str, str | int | None]]],
    ) -> list[str]:
        store = self._stores.get(working_dir)
        if store is None:
            raise ValueError(
                f"Vector store not initialized for working_dir: {working_dir}"
            )

        id_map = self._id_maps.get(working_dir, {})
        lc_docs = []
        ids = []
        for content, file_path, metadata in documents:
            doc_id = str(uuid.uuid4())
            meta = {"file_path": file_path, "chunk_id": doc_id, **metadata}
            lc_docs.append(Document(id=doc_id, page_content=content, metadata=meta))
            ids.append(doc_id)
            id_map[doc_id] = file_path

        await store.aadd_documents(lc_docs, ids=ids)
        return ids

    async def similarity_search(
        self, working_dir: str, query: str, top_k: int = 10
    ) -> list[SearchResult]:
        store = self._stores.get(working_dir)
        if store is None:
            raise ValueError(
                f"Vector store not initialized for working_dir: {working_dir}"
            )

        lc_results = await store.asimilarity_search_with_score(query, k=top_k)
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
            for doc, score in lc_results
        ]

    async def delete_documents(self, working_dir: str, file_path: str) -> int:
        store = self._stores.get(working_dir)
        if store is None:
            raise ValueError(
                f"Vector store not initialized for working_dir: {working_dir}"
            )

        id_map = self._id_maps.get(working_dir, {})
        ids_to_delete = [doc_id for doc_id, fp in id_map.items() if fp == file_path]
        if ids_to_delete:
            await store.adelete(ids=ids_to_delete)
            for doc_id in ids_to_delete:
                id_map.pop(doc_id, None)
        return len(ids_to_delete)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None
