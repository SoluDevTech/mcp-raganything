"""Port for encrypting/decrypting secrets at rest.

The registry store stores ``headers``, ``env`` and ``auth_token`` as ciphertext.
The cipher abstracts the encryption mechanism so the domain stays independent
of the concrete Fernet implementation.
"""

from abc import ABC, abstractmethod


class SecretCipher(ABC):
    """Outbound port: encrypt and decrypt sensitive strings at rest.

    Implementations must satisfy ``decrypt(encrypt(x)) == x`` for any string
    ``x``; the ciphertext produced by ``encrypt`` may vary across calls
    (e.g. Fernet uses a random IV).
    """

    @abstractmethod
    async def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string into a ciphertext string.

        Args:
            plaintext: The UTF-8 string to encrypt.

        Returns:
            The ciphertext as a string (e.g. base64-encoded Fernet token).
        """
        ...

    @abstractmethod
    async def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string back into the original plaintext.

        Args:
            ciphertext: The ciphertext string produced by ``encrypt``.

        Returns:
            The original plaintext string.

        Raises:
            Exception: If the ciphertext is invalid or was encrypted with a
                different key.
        """
        ...
