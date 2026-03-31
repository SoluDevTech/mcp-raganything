from fastapi import APIRouter, Depends, status

from application.requests.query_request import QueryRequest
from application.responses.query_response import QueryResponse
from application.use_cases.query_use_case import QueryUseCase
from dependencies import get_query_use_case

query_router = APIRouter(tags=["RAG Query"])


@query_router.post(
    "/query", response_model=QueryResponse, status_code=status.HTTP_200_OK
)
async def query_knowledge_base(
    request: QueryRequest,
    use_case: QueryUseCase = Depends(get_query_use_case),
) -> QueryResponse:
    result = await use_case.execute(
        working_dir=request.working_dir,
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
    )
    return QueryResponse(**result)
