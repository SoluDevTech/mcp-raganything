"""LLM-related domain errors."""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class LlmNotConfiguredError(DomainError):
    """Raised when an authenticated user has no configured LLM provider.

    The user is authenticated (``current_user_id`` is set) but has no row in
    the ``user_llm_settings`` table, so the per-user LLM/embeddings factories
    cannot build a client. The user must configure their provider via
    composable-agents' settings endpoint before using LLM features.

    Maps to HTTP 422 (Unprocessable Entity): the request is syntactically
    valid but the user cannot use the LLM features until they configure their
    provider.
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY
