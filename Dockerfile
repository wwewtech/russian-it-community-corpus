# Production Dockerfile for Russian IT Community Data Platform
FROM python:3.13-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml* setup.py* requirements.txt* ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    pandas pyarrow scikit-learn rich tiktoken \
    natasha slovnet razdel yargy pymorphy3 pymorphy3-dicts-ru \
    streamlit pydantic ujson xxhash

COPY . .

EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit Data Studio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
