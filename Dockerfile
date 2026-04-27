# Multi-stage build for RAG-Anything API
# Stage 1: Build dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (no pip cache, CPU-only PyTorch)
RUN UV_TORCH_BACKEND=cpu uv sync --frozen --no-dev --no-cache

# Stage 2: Runtime image
FROM python:3.13-slim-bookworm

# Install only critical runtime system deps, then clean up apt metadata to keep image slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tesseract-ocr tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ /app/src/
COPY .env.example /app/.env

# Set Python path to include src directory
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["python","src/main.py"]
