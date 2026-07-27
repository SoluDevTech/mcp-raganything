"""API key hashing / generation utility (domain logic).

Pure helper (no port, no I/O) used by the auth domain service to hash incoming
API keys before looking them up in the repository, and to generate new
plaintext keys with the ``cpk_`` prefix.

Hashing uses SHA-256 (deterministic, fast, sufficient for a high-entropy
secret — bcrypt would be overkill and slower on every request). This lives in
the domain layer because the hashing policy is part of the authentication
business rules, not an infrastructure concern.
"""

import hashlib
import secrets


class ApiKeyHasher:
    """Hash and generate Composable Agents API keys (``cpk_`` prefixed)."""

    @staticmethod
    def hash_key(plaintext: str) -> str:
        """Return the SHA-256 hex digest of ``plaintext``.

        Args:
            plaintext: The raw API key (e.g. ``cpk_xxx``).

        Returns:
            The 64-char lowercase hex digest.
        """
        return hashlib.sha256(plaintext.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new random API key with the ``cpk_`` prefix.

        Returns:
            A string of the form ``cpk_`` + 32+ url-safe base64 chars.
        """
        return "cpk_" + secrets.token_urlsafe(32)
