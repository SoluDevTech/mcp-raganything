"""Fernet-based secret cipher for encrypting secrets at rest.

Uses :class:`cryptography.fernet.Fernet` to encrypt/decrypt sensitive strings
(headers, env, auth_token) stored in the MCP server registry. The cipher
methods are ``async`` to match the :class:`SecretCipher` port contract, even
though Fernet operations are synchronous and CPU-bound (negligible for the
short strings involved in server configuration).
"""

from cryptography.fernet import Fernet

from domain.ports.secret_cipher import SecretCipher


class FernetSecretCipher(SecretCipher):
    """Encrypt/decrypt strings at rest using Fernet symmetric encryption.

    Attributes:
        _fernet: The Fernet instance built from the provided key.
    """

    def __init__(self, key: str) -> None:
        """Initialize the cipher with a Fernet key.

        Args:
            key: A url-safe base64-encoded 32-byte Fernet key. Required — the
                same key must be used across all processes/restarts so that
                previously encrypted secrets can be decrypted. Generate one
                with ``Fernet.generate_key()`` and set it via the
                ``SECRET_ENCRYPTION_KEY`` environment variable.

        Raises:
            ValueError: If ``key`` is empty or not a valid Fernet key.
        """
        if not key:
            raise ValueError(
                "A Fernet key is required for FernetSecretCipher. "
                "Set SECRET_ENCRYPTION_KEY in the environment."
            )
        self._fernet = Fernet(key.encode())

    async def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string into a Fernet ciphertext string.

        Args:
            plaintext: The UTF-8 string to encrypt.

        Returns:
            A base64-encoded Fernet token as a string.
        """
        return self._fernet.encrypt(plaintext.encode()).decode()

    async def decrypt(self, ciphertext: str) -> str:
        """Decrypt a Fernet ciphertext string back into the original plaintext.

        Args:
            ciphertext: The base64-encoded Fernet token string.

        Returns:
            The original plaintext string.

        Raises:
            cryptography.fernet.InvalidToken: If the ciphertext is invalid or
                was encrypted with a different key.
        """
        return self._fernet.decrypt(ciphertext.encode()).decode()
