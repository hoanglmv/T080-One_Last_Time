FROM ghcr.io/astral-sh/uv:0.11.26 AS uv
FROM python:3.11-slim

WORKDIR /app

# Install system libraries needed for LightGBM/XGBoost OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Astral uv binaries
COPY --from=uv /uv /uvx /bin/

# Copy dependency specifications first for optimal layer caching
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Security: non-root user
RUN useradd -m appuser

# Copy source code, web UI and pre-trained artifacts
COPY . .

# Ensure appuser owns app directory
RUN mkdir -p /app/data /app/artifacts/models && chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
