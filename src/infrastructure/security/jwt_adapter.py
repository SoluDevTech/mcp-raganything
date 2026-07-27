"""JWT adapter — verifies bearer tokens against a remote JWKS endpoint.

Mirrors the composable-agents JWT adapter pattern: uses an in-memory
``cachetools.TTLCache`` (single entry, TTL 300s) for the JWKS document. The
JWKS document is fetched via ``httpx.AsyncClient`` and cached; individual
signing keys are reconstructed from the cached JWKS without any network call.

On ANY decode error (expired, invalid signature, bad audience, unreachable
JWKS endpoint, malformed payload, …) the adapter returns ``None`` and logs a
warning — it NEVER raises, so the caller (``AuthService``) can fall through to
other auth methods or reject the request with a clean 401.

No PII is ever logged: only the error type and message are interpolated into
the failure log line.
"""

import asyncio
import logging

import httpx
import jwt
from cachetools import TTLCache
from jwt import PyJWK, PyJWKClientConnectionError
from jwt.exceptions import PyJWKClientError

from domain.entities.user.user import User
from domain.logging.messages import LogMessage
from domain.ports.auth.jwt_service import JwtServicePort

logger = logging.getLogger(__name__)

# Algorithms accepted for verification. No issuer validation is performed.
_ACCEPTED_ALGORITHMS = ["RS256", "ES256", "ES384"]


class JwtAdapter(JwtServicePort):
    """Verify JWT bearer tokens against a JWKS endpoint with in-memory caching.

    The JWKS document is fetched lazily on the first ``decode_token`` call and
    cached in a ``cachetools.TTLCache`` (single entry, TTL 300s). An
    ``asyncio.Lock`` guards the fetch to prevent a stampede of concurrent
    decodes all hitting the JWKS endpoint on a cold cache.

    Args:
        jwks_url: URL of the OIDC JWKS endpoint. When empty, ``decode_token``
            always returns ``None`` (no JWKS configured).
        audience: Expected JWT ``aud`` claim. Passed to ``jwt.decode``.
    """

    _JWKS_CACHE_TTL = 300
    _JWKS_CACHE_KEY = "jwks"

    def __init__(self, jwks_url: str, audience: str) -> None:
        self._jwks_url = jwks_url
        self._audience = audience
        # The HTTP client is only needed when a JWKS endpoint is configured;
        # the cache + lock are cheap and always present so the rest of the
        # code does not need to branch on ``jwks_url``.
        self._jwks_http_client: httpx.AsyncClient | None = (
            httpx.AsyncClient(
                timeout=30.0, headers={"User-Agent": "mcp-raganything/1.0"}
            )
            if jwks_url
            else None
        )
        self._jwks_cache: TTLCache = TTLCache(maxsize=1, ttl=self._JWKS_CACHE_TTL)
        self._jwks_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # JWKS fetching / caching
    # ------------------------------------------------------------------

    async def _fetch_jwks(self) -> dict:
        """Fetch the JWKS document from the endpoint as a dict.

        Raises:
            httpx.HTTPError: on any transport / HTTP error (caller logs + None).
            PyJWKClientConnectionError: on a JWKS client connection failure.
        """
        response = await self._jwks_http_client.get(self._jwks_url)  # type: ignore[union-attr]
        response.raise_for_status()
        return response.json()

    async def _get_cached_jwks(self) -> dict:
        """Return the cached JWKS, fetching it on a cache miss.

        An ``asyncio.Lock`` serialises concurrent cold-cache fetches so only one
        HTTP call is made even under burst load. On fetch failure nothing is
        cached (the next call will retry).
        """
        cached = self._jwks_cache.get(self._JWKS_CACHE_KEY)
        if cached is not None:
            return cached
        async with self._jwks_lock:
            # Re-check inside the lock — another task may have populated it.
            cached = self._jwks_cache.get(self._JWKS_CACHE_KEY)
            if cached is not None:
                return cached
            jwks = await self._fetch_jwks()
            self._jwks_cache[self._JWKS_CACHE_KEY] = jwks
            return jwks

    # ------------------------------------------------------------------
    # Token / key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_kid(token: str) -> str | None:
        """Extract the ``kid`` from the JWT header without verifying signature."""
        header = jwt.get_unverified_header(token)
        return header.get("kid")

    @staticmethod
    def _find_key_by_kid(jwks: dict, kid: str | None) -> dict:
        """Find the JWK with the given ``kid`` in the JWKS document.

        Falls back to the first key when ``kid`` is ``None`` or no match is
        found (single-key JWKS). Raises ``PyJWKClientError`` when the JWKS is
        empty.
        """
        keys = jwks.get("keys", [])
        if kid is not None:
            for key in keys:
                if key.get("kid") == kid:
                    return key
        if keys:
            return keys[0]
        raise PyJWKClientError(f"Unable to find a signing key that matches: '{kid}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decode_token(self, token: str) -> User | None:
        """Verify ``token`` and return the resolved :class:`User`, or ``None``.

        Args:
            token: The raw JWT string (without the ``Bearer `` prefix).

        Returns:
            The authenticated :class:`User` on success, or ``None`` on any
            verification failure. Never raises.
        """
        if self._jwks_http_client is None:
            # No JWKS configured — cannot verify anything.
            return None

        try:
            # 1. Fetch (cached) JWKS — done first so a JWKS endpoint outage is
            #    reported as AUTH_JWKS_FETCH_FAILED rather than masked by a
            #    header-decode error on an opaque token.
            jwks = await self._get_cached_jwks()

            # 2. Resolve the signing key from the JWKS by kid.
            kid = self._extract_kid(token)
            key_dict = self._find_key_by_kid(jwks, kid)
            signing_key = PyJWK.from_dict(key_dict)

            # 3. Verify + decode.
            payload = jwt.decode(
                token,
                key=signing_key.key,
                algorithms=_ACCEPTED_ALGORITHMS,
                audience=self._audience,
            )

            return User.model_validate(payload)
        except (httpx.HTTPError, PyJWKClientConnectionError) as e:
            # Both signal a JWKS endpoint outage / transport failure.
            logger.error("%s: %s", LogMessage.AUTH_JWKS_FETCH_FAILED, e)
            return None
        except (
            jwt.ExpiredSignatureError,
            jwt.InvalidAlgorithmError,
            jwt.PyJWTError,
        ) as e:
            # PyJWKClientError is a subclass of PyJWTError and is caught here.
            # The enum value is a stable prefix kept verbatim in the formatted
            # line so tests can assert membership without PII leakage.
            logger.warning(
                "%s: %s: %s", LogMessage.AUTH_JWT_DECODE_FAILED, type(e).__name__, e
            )
            return None
        except ValueError as e:
            # Pydantic ValidationError is a ValueError subclass — covers
            # malformed claims (e.g. missing ``sub``) without leaking PII.
            logger.warning(
                "%s: %s: %s", LogMessage.AUTH_JWT_DECODE_FAILED, type(e).__name__, e
            )
            return None

    async def close(self) -> None:
        """Close the internal HTTP client if one was created."""
        if self._jwks_http_client is not None:
            await self._jwks_http_client.aclose()
