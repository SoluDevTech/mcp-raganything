from application.requests.query_request import MultimodalContentItem
from domain.ports.rag_engine import RAGEnginePort


class MultimodalQueryUseCase:
    """Use case for querying the RAG knowledge base with multimodal content."""

    def __init__(self, rag_engine: RAGEnginePort) -> None:
        self.rag_engine = rag_engine

    async def execute(
        self,
        working_dir: str,
        query: str,
        multimodal_content: list[MultimodalContentItem],
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> dict:
        working_dir = working_dir if working_dir.endswith("/") else f"{working_dir}/"
        self.rag_engine.init_project(working_dir)
        result = await self.rag_engine.query_multimodal(
            query=query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            working_dir=working_dir,
        )
        return {"data": result}
