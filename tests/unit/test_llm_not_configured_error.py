"""Tests for the ``LlmNotConfiguredError`` domain error (TDD Red phase).

The error is raised by the per-user LLM factories when a user is authenticated
(``current_user_id`` is set) but has no configured LLM provider settings in
the ``user_llm_settings`` table. It maps to HTTP 422 Unprocessable Entity
(the request is syntactically valid but the user cannot use the LLM features
until they configure their provider).
"""

import pytest

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode
from domain.errors.llm import LlmNotConfiguredError
from domain.errors.messages import ErrorMessage


class TestLlmNotConfiguredError:
    """Tests for the LlmNotConfiguredError class."""

    def test_inherits_from_domain_error(self) -> None:
        assert issubclass(LlmNotConfiguredError, DomainError)

    def test_status_code_is_422(self) -> None:
        assert LlmNotConfiguredError.status_code == ErrorCode.UNPROCESSABLE_ENTITY
        assert LlmNotConfiguredError.status_code == 422

    def test_carries_detail_message(self) -> None:
        exc = LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
        assert exc.detail == ErrorMessage.LLM_NOT_CONFIGURED

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LlmNotConfiguredError) as exc_info:
            raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
        assert ErrorMessage.LLM_NOT_CONFIGURED in exc_info.value.detail
