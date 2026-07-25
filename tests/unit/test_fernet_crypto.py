"""Tests for the FernetSecretCipher adapter.

Covers the ``encrypt``/``decrypt`` round-trip and the constructor validation
(empty key raises). The cipher wraps ``cryptography.fernet.Fernet`` so secrets
stored in the MCP server registry are encrypted at rest.
"""

import pytest
from cryptography.fernet import Fernet, InvalidToken

from infrastructure.security.crypto import FernetSecretCipher


class TestFernetSecretCipher:
    """Constructor validation for FernetSecretCipher."""

    def test_accepts_a_valid_fernet_key(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()

        # Act
        cipher = FernetSecretCipher(key)

        # Assert
        assert cipher is not None

    def test_raises_on_empty_key(self) -> None:
        # Act & Assert
        with pytest.raises(ValueError):
            FernetSecretCipher("")


class TestFernetSecretCipherRoundTrip:
    """encrypt -> decrypt must return the original plaintext."""

    async def test_encrypt_then_decrypt_returns_original_plaintext(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)
        plaintext = "my-secret-token"

        # Act
        ciphertext = await cipher.encrypt(plaintext)
        recovered = await cipher.decrypt(ciphertext)

        # Assert
        assert recovered == plaintext

    async def test_encrypt_produces_a_different_ciphertext_than_plaintext(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)
        plaintext = "my-secret-token"

        # Act
        ciphertext = await cipher.encrypt(plaintext)

        # Assert
        assert ciphertext != plaintext

    async def test_encrypt_is_non_deterministic(self) -> None:
        """Fernet uses a random IV so two encryptions of the same value differ."""
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)
        plaintext = "my-secret-token"

        # Act
        ct1 = await cipher.encrypt(plaintext)
        ct2 = await cipher.encrypt(plaintext)

        # Assert
        assert ct1 != ct2

    async def test_decrypt_with_a_different_key_raises_invalid_token(self) -> None:
        # Arrange
        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()
        cipher1 = FernetSecretCipher(key1)
        cipher2 = FernetSecretCipher(key2)
        ciphertext = await cipher1.encrypt("secret")

        # Act & Assert
        with pytest.raises(InvalidToken):
            await cipher2.decrypt(ciphertext)

    async def test_decrypt_of_a_garbage_string_raises_invalid_token(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)

        # Act & Assert
        with pytest.raises(InvalidToken):
            await cipher.decrypt("not-a-valid-fernet-token")

    async def test_round_trip_preserves_json_string(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)
        import json

        payload = json.dumps({"Authorization": "Bearer tok", "X-Api-Key": "abc"})

        # Act
        recovered = await cipher.decrypt(await cipher.encrypt(payload))

        # Assert
        assert json.loads(recovered) == {"Authorization": "Bearer tok", "X-Api-Key": "abc"}

    async def test_encrypt_then_decrypt_empty_string(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)

        # Act
        recovered = await cipher.decrypt(await cipher.encrypt(""))

        # Assert
        assert recovered == ""

    async def test_encrypt_then_decrypt_unicode(self) -> None:
        # Arrange
        key = Fernet.generate_key().decode()
        cipher = FernetSecretCipher(key)
        plaintext = "clé-secrète-🔑"

        # Act
        recovered = await cipher.decrypt(await cipher.encrypt(plaintext))

        # Assert
        assert recovered == plaintext
