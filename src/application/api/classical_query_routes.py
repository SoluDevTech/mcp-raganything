from fastapi import APIRouter, Depends, status

from application.requests.classical_query_request import ClassicalQueryRequest
from application.responses.classical_query_response import ClassicalChunkResponse
from application.use_cases.classical_query_use_case import ClassicalQueryUseCase
from dependencies import get_classical_query_use_case

classical_query_router = APIRouter(tags=["Classical Query"])


@classical_query_router.post(
    "/classical/query",
    status_code=status.HTTP_200_OK,
)
async def classical_query(
    request: ClassicalQueryRequest,
    use_case: ClassicalQueryUseCase = Depends(get_classical_query_use_case),
) -> list[ClassicalChunkResponse]:
    response = await use_case.execute(
        working_dir=request.working_dir,
        query=request.query,
        top_k=request.top_k,
        num_variations=request.num_variations,
        relevance_threshold=request.relevance_threshold,
        vector_distance_threshold=request.vector_distance_threshold,
        enable_llm_judge=request.enable_llm_judge,
    )
    return response.chunks
