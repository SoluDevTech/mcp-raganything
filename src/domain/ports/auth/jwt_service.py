"""Port for JWT token verification (outbound boundary).

The application layer depends on this abstraction; the concrete ``JwtAdapter``
(infra) implements it against a remote JWKS endpoint.
"""

from abc import ABC, abstractmethod

from domain.entities.user.user import User


class JwtServicePort(ABC):
    """Outbound port: decode and verify a JWT bearer token.

    Implementations MUST be async (the JWKS endpoint is fetched over HTTP) and
    MUST return ``None`` (never raise) when the token is invalid, expired or the
    JWKS endpoint is unreachable — so the caller can fall through to other auth
    methods or reject the request with a clean 401.
    """

    @abstractmethod
    async def decode_token(self, token: str) -> User | None:
        """Verify ``token`` and return the resolved :class:`User`, or ``None``.

        Args:
            token: The raw JWT string (without the ``Bearer `` prefix).

        Returns:
            The authenticated :class:`User` on success, or ``None`` on any
            verification failure (expired, invalid signature, bad audience,
            unreachable JWKS endpoint, malformed payload, …).
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the adapter (HTTP client, etc.)."""
        ...
