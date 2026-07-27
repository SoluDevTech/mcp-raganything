"""Tests for the ApiKeyHasher utility.

The hasher is a pure helper (no port) used by the auth domain service to hash
incoming API keys before looking them up in the repository and to generate new
plaintext keys with the ``cpk_`` prefix. Tests assert deterministic sha256 hex
output and the prefix/length/uniqueness guarantees of ``generate_key``.
"""

import hashlib

from domain.services.auth.api_key_hasher import ApiKeyHasher


class TestApiKeyHasherHashKey:
    """Tests for ``ApiKeyHasher.hash_key``."""

    def test_hash_key_returns_deterministic_sha256_hex(self) -> None:
        # Arrange
        plaintext = "cpk_super-secret-value"

        # Act
        result = ApiKeyHasher.hash_key(plaintext)

        # Assert
        expected = hashlib.sha256(plaintext.encode()).hexdigest()
        assert result == expected

    def test_hash_key_is_stable_across_calls(self) -> None:
        # Arrange
        plaintext = "cpk_stable-key"

        # Act
        first = ApiKeyHasher.hash_key(plaintext)
        second = ApiKeyHasher.hash_key(plaintext)

        # Assert
        assert first == second

    def test_hash_key_differs_for_different_inputs(self) -> None:
        # Arrange
        a = "cpk_one"
        b = "cpk_two"

        # Act
        hash_a = ApiKeyHasher.hash_key(a)
        hash_b = ApiKeyHasher.hash_key(b)

        # Assert
        assert hash_a != hash_b

    def test_hash_key_returns_64_char_hex_string(self) -> None:
        # Arrange
        plaintext = "cpk_length-check"

        # Act
        result = ApiKeyHasher.hash_key(plaintext)

        # Assert
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestApiKeyHasherGenerateKey:
    """Tests for ``ApiKeyHasher.generate_key``."""

    def test_generate_key_starts_with_cpk_prefix(self) -> None:
        # Act
        key = ApiKeyHasher.generate_key()

        # Assert
        assert key.startswith("cpk_")

    def test_generate_key_is_long_enough(self) -> None:
        # Act
        key = ApiKeyHasher.generate_key()

        # Assert — ``cpk_`` prefix (4) + at least 32 urlsafe chars
        assert len(key) > 40

    def test_generate_key_is_unique_across_many_calls(self) -> None:
        # Arrange
        keys: set[str] = set()

        # Act
        for _ in range(1000):
            keys.add(ApiKeyHasher.generate_key())

        # Assert
        assert len(keys) == 1000
