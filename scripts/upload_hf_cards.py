"""Upload corrected dataset/model cards to Hugging Face Hub.

Requires HF_TOKEN env var with write access:
    set HF_TOKEN=hf_... && python scripts/upload_hf_cards.py
"""

import os
import sys
from pathlib import Path

if not os.getenv("HF_TOKEN"):
    sys.exit("HF_TOKEN is not set; cannot upload.")

from huggingface_hub import HfApi  # noqa: E402

DATASET_REPO = "wwewtech/russian-it-community-corpus"
MODEL_REPO = "wwewtech/russian-it-community-lora"

UPLOADS = [
    ("reports/DATASET_AND_ANALYTICS.md", DATASET_REPO, "README.md", "dataset"),
    ("reports/HF_MODEL_CARD.md", MODEL_REPO, "README.md", "model"),
]

api = HfApi(token=os.environ["HF_TOKEN"])
for local, repo_id, path_in_repo, repo_type in UPLOADS:
    p = Path(local)
    if not p.exists():
        print(f"SKIP (missing): {local}")
        continue
    api.upload_file(path_or_fileobj=str(p), path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)
    print(f"Uploaded {local} -> {repo_id}/{path_in_repo} ({repo_type})")
