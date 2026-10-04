# Installation

## Requirements

- Python 3.11+
- 8GB+ RAM (16GB+ recommended for LoRA training)
- GPU optional but recommended for training (8GB+ VRAM)

## Standard Installation

```bash
pip install ricc
```

This installs the core dependencies:
- `pandas`, `pyarrow` — data processing
- `pydantic`, `pydantic-settings` — configuration
- `natasha`, `slovnet`, `razdel`, `yargy` — Russian NLP
- `torch`, `transformers`, `peft`, `trl` — LLM fine-tuning
- `prometheus-client`, `structlog` — observability

## Platform Extras (Optional)

```bash
# With Prefect orchestration + DVC data versioning
pip install ricc[platform]
```

## Development Installation

```bash
git clone https://github.com/wwewtech/russian-it-community-corpus.git
cd russian-it-community-corpus
pip install -e .[dev]
pre-commit install
```

## Docker

```bash
docker pull ghcr.io/wwewtech/russian-it-community-corpus:latest
docker run -p 8501:8501 -v $(pwd)/dataset_output:/app/dataset_output ghcr.io/wwewtech/russian-it-community-corpus
```

## Environment Variables

Create `.env` from template:

```bash
cp .env.example .env
# Edit .env with your tokens
```

Required:
- `HF_TOKEN` — Hugging Face Hub token (for gated models, dataset/model pushes)

Optional:
- `WANDB_API_KEY` — Weights & Biases experiment tracking
- `PREFECT_API_KEY` — Prefect Cloud orchestration
- Cloud provider credentials (AWS, GCP, Azure)

## Verification

```bash
# Run tests
python -m pytest -q

# Type check
python -m mypy src/ricc --strict

# SLO gate
make slo
```