"""Domain entity for per-user LLM provider settings (read-only projection).

mcp-raganything shares the same Postgres database as composable-agents, which
owns the ``user_llm_settings`` table (created by composable-agents' Alembic).
This service READS the table (read-only) to resolve per-user
LLM/embeddings credentials. Writes (upsert/delete) are composable-agents'
concern — they are NOT exposed here.

The ``api_key_masked`` field is a safe preview (``first3...last3``) derived
from the decrypted key — never the full plaintext. The full plaintext is only
ever returned by ``UserLlmSettingsRepositoryPort.get_decrypted`` to the
per-user LLM factory (in-process, never serialized over the wire).

Attributes:
    user_id: Owner identifier (from the auth context, PK of the table).
    provider: Free-form label, e.g. "openai", "openrouter" (display only).
    base_url: OpenAI-compatible base URL used by the LLM/embeddings factory.
    api_key_masked: Masked preview of the API key (never the full key) —
        ``None`` when no settings exist or decryption failed.
    created_at: Creation timestamp (UTC).
    updated_at: Last update timestamp (UTC).
"""

from datetime import datetime

from pydantic import BaseModel


class UserLlmSettings(BaseModel):
    """Per-user LLM provider settings (safe projection — no plaintext key)."""

    user_id: str
    provider: str
    base_url: str
    api_key_masked: str | None = None
    created_at: datetime
    updated_at: datetime


class UserLlmSettingsInput(BaseModel):
    """Input DTO for an upsert (PUT) operation (composable-agents' concern).

    Kept here for parity with composable-agents so the entity module is the
    single source of truth for the ``user_llm_settings`` schema. mcp-raganything
    does NOT use this DTO (it is read-only).
    """

    provider: str
    base_url: str
    api_key: str
