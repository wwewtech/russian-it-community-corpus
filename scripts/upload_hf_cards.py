"""Upload corrected dataset/model cards to Hugging Face Hub.

Requires HF_TOKEN env var with write access:
    set HF_TOKEN=hf_... && python scripts/upload_hf_cards.py
"""

import os
import sys
from pathlib import Path


def _resolve_token() -> str | None:
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip() or None
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


token = _resolve_token()
if not token:
    sys.exit(
        "HF token not found. Either set HF_TOKEN environment variable with write access, "
        "or run 'huggingface-cli login' (see docs/adr/0001-hf-token-handling.md)."
    )

from huggingface_hub import HfApi  # noqa: E402

DATASET_REPO = "wwewtech/russian-it-community-corpus"
MODEL_REPO = "wwewtech/russian-it-community-lora"

UPLOADS = [
    ("reports/DATASET_AND_ANALYTICS.md", DATASET_REPO, "README.md", "dataset"),
    ("reports/HF_MODEL_CARD.md", MODEL_REPO, "README.md", "model"),
]

api = HfApi(token=token)
for local, repo_id, path_in_repo, repo_type in UPLOADS:
    p = Path(local)
    if not p.exists():
        print(f"SKIP (missing): {local}")
        continue
    api.upload_file(path_or_fileobj=str(p), path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)
    print(f"Uploaded {local} -> {repo_id}/{path_in_repo} ({repo_type})")
