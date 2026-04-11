"""Tests for _coerce_file_extensions validator in indexing_request.py.

The validator converts various input types into a consistent list[str] | None
format for the IndexFolderRequest.file_extensions field.
"""


from application.requests.indexing_request import (
    IndexFolderRequest,
    _coerce_file_extensions,
)


class TestCoerceFileExtensions:
    """Tests for the _coerce_file_extensions validator function."""

    def test_string_input_wrapped_in_list(self) -> None:
        """A single string should be wrapped in a list."""
        result = _coerce_file_extensions(".pdf")
        assert result == [".pdf"]

    def test_list_passthrough(self) -> None:
        """A list should be returned unchanged."""
        extensions = [".pdf", ".docx", ".txt"]
        result = _coerce_file_extensions(extensions)
        assert result == extensions

    def test_none_returns_none(self) -> None:
        """None input should return None."""
        result = _coerce_file_extensions(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string should be treated as None."""
        result = _coerce_file_extensions("")
        assert result is None

    def test_single_extension_string(self) -> None:
        """A single extension string should produce a single-element list."""
        result = _coerce_file_extensions(".xlsx")
        assert result == [".xlsx"]


class TestIndexFolderRequestFileExtensions:
    """Integration tests for file_extensions coercion via Pydantic model."""

    def test_accepts_list_of_extensions(self) -> None:
        """Should accept a list of file extensions directly."""
        request = IndexFolderRequest(
            working_dir="/tmp/test",
            file_extensions=[".pdf", ".docx"],
        )
        assert request.file_extensions == [".pdf", ".docx"]

    def test_coerces_single_string_to_list(self) -> None:
        """Should coerce a single string into a one-element list."""
        request = IndexFolderRequest(
            working_dir="/tmp/test",
            file_extensions=".pdf",
        )
        assert request.file_extensions == [".pdf"]

    def test_defaults_to_none(self) -> None:
        """Should default to None when file_extensions is not provided."""
        request = IndexFolderRequest(working_dir="/tmp/test")
        assert request.file_extensions is None
