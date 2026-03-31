from domain.ports.rag_engine import RAGEnginePort


class QueryUseCase:
    """Use case for querying the RAG knowledge base."""

    def __init__(self, rag_engine: RAGEnginePort) -> None:
        self.rag_engine = rag_engine

    async def execute(
        self, working_dir: str, query: str, mode: str = "naive", top_k: int = 10
    ) -> dict:
        self.rag_engine.init_project(working_dir)
        return await self.rag_engine.query(
            query=query, mode=mode, top_k=top_k, working_dir=working_dir
        )
