"""Tests for FastAPI lifespan management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLifespan:
    """Tests for lifespan context managers in main.py."""

    @pytest.mark.asyncio
    async def test_db_lifespan_closes_classical_bm25_pool_on_shutdown(self):
        """Should close Classical BM25 adapter connection pool on shutdown."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()

        with (
            patch("main.classical_bm25_adapter", mock_bm25),
            patch("main.classical_vector_store", None),
            patch("main.bricks_api_adapter", MagicMock(close=AsyncMock())),
        ):
            # Arrange
            async with db_lifespan(mock_app):
                pass
            # Assert
            mock_bm25.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_no_classical_bm25_adapter(self):
        """Should handle gracefully when classical_bm25_adapter is None."""
        from main import db_lifespan

        mock_app = MagicMock()

        with (
            patch("main.classical_bm25_adapter", None),
            patch("main.classical_vector_store", None),
            patch("main.bricks_api_adapter", MagicMock(close=AsyncMock())),
        ):
            # Arrange
            async with db_lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_close_failure(self):
        """Should not crash if Classical BM25 close fails."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()
        mock_bm25.close = AsyncMock(side_effect=Exception("Close failed"))

        with (
            patch("main.classical_bm25_adapter", mock_bm25),
            patch("main.classical_vector_store", None),
            patch("main.bricks_api_adapter", MagicMock(close=AsyncMock())),
        ):
            # Arrange
            async with db_lifespan(mock_app):
                pass
            # Assert
            mock_bm25.close.assert_called_once()

    def test_run_fastapi_starts_uvicorn(self):
        """Should start uvicorn when run_fastapi is called."""
        with patch("main.uvicorn.run") as mock_uvicorn:
            # Arrange
            from main import run_fastapi

            run_fastapi()
            # Assert
            mock_uvicorn.assert_called_once()
