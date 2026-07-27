"""Tests for the ``JwtAdapter`` infrastructure component.

Mirrors the composable-agents JWT adapter pattern: the adapter fetches the
JWKS document via ``httpx.AsyncClient`` and caches it in an in-memory
``cachetools.TTLCache`` (single entry, TTL 300s). On any decode error it
returns ``None`` and logs — it never raises. Algorithms accepted are
``RS256``, ``ES256`` and ``ES384``; no issuer validation.

The external boundary (JWKS HTTP fetch) is mocked by replacing the adapter's
``_jwks_http_client`` with an ``AsyncMock``; ``PyJWK.from_dict``,
``jwt.get_unverified_header`` and ``jwt.decode`` are patched on the adapter
module. The adapter itself (internal component) is instantiated for real.
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from jwt.exceptions import PyJWKClientError

from domain.entities.user.user import User
from domain.logging.messages import LogMessage
from infrastructure.security.jwt_adapter import JwtAdapter

AUDIENCE = "test-audience"
JWKS_URL = "http://test/oidc/jwks"


def _valid_claims() -> dict:
    """Return a minimal valid JWT payload (only fields User.model_validate needs)."""
    now = int(time.time())
    return {
        "sub": "user-abc-123",
        "email": "alice@example.com",
        "name": "Alice Martin",
        "username": "alice",
        "created_at": now,
        "updated_at": now,
    }


def _jwks_dict() -> dict:
    return {"keys": [{"kid": "test-kid", "kty": "RSA", "n": "x", "e": "AQAB"}]}


class TestJwtAdapter:
    """Tests for ``JwtAdapter.decode_token``."""

    @pytest.fixture
    def adapter(self) -> JwtAdapter:
        """A real ``JwtAdapter`` with a JWKS URL + audience (cache enabled)."""
        return JwtAdapter(jwks_url=JWKS_URL, audience=AUDIENCE)

    @pytest.fixture(autouse=False)
    def _stub_signing_key(
        self, adapter: JwtAdapter, request: pytest.FixtureRequest
    ) -> None:
        """Stub JWKS fetch + signing key resolution.

        Replaces the adapter's ``_jwks_http_client`` with an ``AsyncMock`` whose
        ``get`` returns a fake JWKS response. Patches ``PyJWK.from_dict`` to
        return a mock signing key and ``jwt.get_unverified_header`` to return a
        fake header with ``kid=test-kid``. ``jwt.decode`` is left for the
        individual test to patch.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _jwks_dict()

        adapter._jwks_http_client = AsyncMock()
        adapter._jwks_http_client.get = AsyncMock(return_value=mock_response)

        patcher_jwk = patch("infrastructure.security.jwt_adapter.PyJWK.from_dict")
        mock_jwk = patcher_jwk.start()
        mock_key = MagicMock()
        mock_key.key = "mock-key"
        mock_jwk.return_value = mock_key
        request.addfinalizer(patcher_jwk.stop)

        patcher_header = patch(
            "infrastructure.security.jwt_adapter.jwt.get_unverified_header",
            return_value={"kid": "test-kid"},
        )
        patcher_header.start()
        request.addfinalizer(patcher_header.stop)

    # -- No JWKS configured -----------------------------------------------------

    async def test_no_jwks_url_returns_none(self) -> None:
        # Arrange — adapter without jwks_url has no HTTP client
        adapter = JwtAdapter(jwks_url="", audience=AUDIENCE)

        # Act
        result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    # -- Valid decode -----------------------------------------------------------

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_valid_token_returns_user_with_sub(self, adapter: JwtAdapter) -> None:
        # Arrange
        claims = _valid_claims()

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode", return_value=claims
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is not None
        assert isinstance(result, User)
        assert result.sub == claims["sub"]

    # -- Audience validation ----------------------------------------------------

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_audience_passed_to_jwt_decode(self, adapter: JwtAdapter) -> None:
        # Arrange
        claims = _valid_claims()

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode", return_value=claims
        ) as mock_decode:
            await adapter.decode_token("some-token")

        # Assert
        _, kwargs = mock_decode.call_args
        assert kwargs["audience"] == AUDIENCE

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_invalid_audience_returns_none(self, adapter: JwtAdapter) -> None:
        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode",
            side_effect=jwt.InvalidAudienceError("Audience mismatch"),
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    # -- Expiry / algorithm / malformed claims ---------------------------------

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_expired_token_returns_none(self, adapter: JwtAdapter) -> None:
        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode",
            side_effect=jwt.ExpiredSignatureError("Token expired"),
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_invalid_algorithm_returns_none(self, adapter: JwtAdapter) -> None:
        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode",
            side_effect=jwt.InvalidAlgorithmError("Algorithm not supported"),
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_malformed_claims_returns_none(self, adapter: JwtAdapter) -> None:
        # Arrange — payload missing required ``sub`` so model_validate raises ValueError
        bad_payload = {"email": "no-sub@example.com"}

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode", return_value=bad_payload
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    # -- JWKS fetch HTTP errors -------------------------------------------------

    async def test_jwks_http_error_returns_none(self, adapter: JwtAdapter) -> None:
        # Arrange
        adapter._jwks_http_client = AsyncMock()
        adapter._jwks_http_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        # Act
        result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    async def test_jwks_no_matching_kid_returns_none(self, adapter: JwtAdapter) -> None:
        """Empty JWKS keys list → PyJWKClientError → returns None."""
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"keys": []}
        adapter._jwks_http_client = AsyncMock()
        adapter._jwks_http_client.get = AsyncMock(return_value=mock_response)

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.get_unverified_header",
            return_value={"kid": "missing-kid"},
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None

    # -- Cache behaviour --------------------------------------------------------

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_jwks_cached_on_second_decode_call(self, adapter: JwtAdapter) -> None:
        # Arrange
        claims = _valid_claims()

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode", return_value=claims
        ):
            await adapter.decode_token("some-token")
            await adapter.decode_token("some-token")

        # Assert — JWKS fetched only once; second call uses the in-memory TTLCache
        assert adapter._jwks_http_client.get.await_count == 1

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_cache_reset_between_decodes_refetches(
        self, adapter: JwtAdapter
    ) -> None:
        """Clearing the cache forces a refetch on the next decode."""
        # Arrange
        claims = _valid_claims()

        # Act
        with patch(
            "infrastructure.security.jwt_adapter.jwt.decode", return_value=claims
        ):
            await adapter.decode_token("some-token")
            adapter._jwks_cache.clear()  # type: ignore[attr-defined]
            await adapter.decode_token("some-token")

        # Assert — two fetches because the cache was cleared between calls
        assert adapter._jwks_http_client.get.await_count == 2

    # -- PII / logging ----------------------------------------------------------

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_decode_token_does_not_print(self, adapter: JwtAdapter) -> None:
        # Arrange
        claims = _valid_claims()

        # Act & Assert
        with (
            patch(
                "infrastructure.security.jwt_adapter.jwt.decode", return_value=claims
            ),
            patch("builtins.print") as mock_print,
        ):
            await adapter.decode_token("some-token")

        mock_print.assert_not_called()

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_decode_token_logs_failure_without_pii(
        self, adapter: JwtAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """On decode failure the log must not leak the email/sub from the payload."""
        # Arrange
        claims = _valid_claims()

        # Act
        with (
            patch(
                "infrastructure.security.jwt_adapter.jwt.decode",
                side_effect=jwt.ExpiredSignatureError("Token expired"),
            ),
            caplog.at_level(
                logging.DEBUG, logger="infrastructure.security.jwt_adapter"
            ),
        ):
            await adapter.decode_token("some-token")

        # Assert — no PII (email, sub value, full payload) in any log record
        all_messages = " ".join(r.getMessage() for r in caplog.records)
        assert claims["email"] not in all_messages, "Email PII leaked in logs"
        assert claims["sub"] not in all_messages, "Sub PII leaked in logs"
        assert "Alice Martin" not in all_messages, "Name PII leaked in logs"
        # A failure log was emitted
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.usefixtures("_stub_signing_key")
    async def test_decode_failure_logs_decode_failed_message(
        self, adapter: JwtAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed decode should log the centralized ``AUTH_JWT_DECODE_FAILED`` message."""
        # Act
        with (
            patch(
                "infrastructure.security.jwt_adapter.jwt.decode",
                side_effect=jwt.ExpiredSignatureError("Token expired"),
            ),
            caplog.at_level(
                logging.WARNING, logger="infrastructure.security.jwt_adapter"
            ),
        ):
            await adapter.decode_token("some-token")

        # Assert
        assert any(
            LogMessage.AUTH_JWT_DECODE_FAILED in r.getMessage() for r in caplog.records
        )

    async def test_jwks_fetch_failure_logs_jwks_fetch_failed_message(
        self, adapter: JwtAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A JWKS HTTP error should log ``AUTH_JWKS_FETCH_FAILED``."""
        # Arrange
        adapter._jwks_http_client = AsyncMock()
        adapter._jwks_http_client.get = AsyncMock(
            side_effect=httpx.ConnectError("boom")
        )

        # Act
        with caplog.at_level(
            logging.ERROR, logger="infrastructure.security.jwt_adapter"
        ):
            await adapter.decode_token("some-token")

        # Assert
        assert any(
            LogMessage.AUTH_JWKS_FETCH_FAILED in r.getMessage() for r in caplog.records
        )

    # -- PyJWKClientError path --------------------------------------------------

    async def test_pyjwk_client_error_returns_none(self, adapter: JwtAdapter) -> None:
        """When ``PyJWK.from_dict`` raises ``PyJWKClientError`` the adapter returns None."""
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _jwks_dict()
        adapter._jwks_http_client = AsyncMock()
        adapter._jwks_http_client.get = AsyncMock(return_value=mock_response)

        # Act
        with (
            patch(
                "infrastructure.security.jwt_adapter.jwt.get_unverified_header",
                return_value={"kid": "test-kid"},
            ),
            patch(
                "infrastructure.security.jwt_adapter.PyJWK.from_dict",
                side_effect=PyJWKClientError("bad key"),
            ),
        ):
            result = await adapter.decode_token("some-token")

        # Assert
        assert result is None
