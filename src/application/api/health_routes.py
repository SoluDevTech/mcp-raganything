from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from application.use_cases.liveness_check_use_case import LivenessCheckUseCase
from dependencies import get_liveness_check_use_case

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status message indicating the API is running.
    """
    return {"message": "RAG Anything API is running"}


@health_router.get("/health/live")
async def liveness_check(
    use_case: LivenessCheckUseCase = Depends(get_liveness_check_use_case),
):
    """Liveness probe that checks PostgreSQL and MinIO connectivity.

    Returns:
        200 if both connections are healthy, 503 if any is unreachable.
    """
    result = await use_case.execute()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)
