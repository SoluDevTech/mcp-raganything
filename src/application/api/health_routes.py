from fastapi import APIRouter

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status message indicating the API is running.
    """
    return {"message": "RAG Anything API is running"}
