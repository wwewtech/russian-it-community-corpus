# Production Dockerfile for Russian IT Community Data Platform
FROM python:3.13-slim

LABEL org.opencontainers.image.title="RICC Data Platform" \
      org.opencontainers.image.description="Russian IT Community Corpus — Zero-PII curation + SFT/DPO/RAG + LoRA" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# System dependencies — one layer, no recommends, cleaned apt lists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first for Docker layer caching.
# requirements.txt is the single source of truth; pyproject mirrors it.
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Non-root runtime user (hardening: no container-as-root).
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/.hf_cache /app/dataset_output /app/reports \
    && chown -R appuser:appuser /app
USER appuser

# Application code last (changes most often).
COPY --chown=appuser:appuser . .

EXPOSE 8501

# Healthcheck hits Streamlit internal health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit Data Studio.
# Pipeline runs via: docker compose --profile pipeline up pipeline-runner
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
