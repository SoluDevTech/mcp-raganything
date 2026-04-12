from fastapi import APIRouter, Depends, status

from application.requests.query_request import MultimodalQueryRequest, QueryRequest
from application.responses.query_response import (
    ChunkResponse,
    MultimodalQueryResponse,
    QueryResponse,
)
from application.use_cases.multimodal_query_use_case import MultimodalQueryUseCase
from application.use_cases.query_use_case import QueryUseCase
from dependencies import get_multimodal_query_use_case, get_query_use_case

query_router = APIRouter(tags=["RAG Query"])


@query_router.post("/query", status_code=status.HTTP_200_OK)
async def query_knowledge_base(
    request: QueryRequest,
    use_case: QueryUseCase = Depends(get_query_use_case),
) -> list[ChunkResponse]:
    result = await use_case.execute(
        working_dir=request.working_dir,
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
    )
    response = QueryResponse(**result)
    return response.data.chunks


@query_router.post(
    "/query/multimodal",
    status_code=status.HTTP_200_OK,
)
async def query_knowledge_base_multimodal(
    request: MultimodalQueryRequest,
    use_case: MultimodalQueryUseCase = Depends(get_multimodal_query_use_case),
) -> MultimodalQueryResponse:
    result = await use_case.execute(
        working_dir=request.working_dir,
        query=request.query,
        multimodal_content=request.multimodal_content,
        mode=request.mode,
        top_k=request.top_k,
    )
    return MultimodalQueryResponse(**result)
