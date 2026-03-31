# Multi-stage build for RAG-Anything API
# Stage 1: Build dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Stage 2: Runtime image
FROM python:3.13-slim-bookworm

# Install system dependencies required by docling
# tesseract-ocr: required by docling PDF pipeline init
# libgl1: required by cv2 (used by docling table structure model)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    git \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ /app/src/
COPY .env.example /app/.env

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["python","src/main.py"]
