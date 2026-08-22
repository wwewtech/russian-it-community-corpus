"""
Hugging Face Hub Uploader for Russian IT Community Corpus (RICC).
Uploads Parquet datasets, JSONL files, Dataset Card, and LoRA adapters directly to HF Hub.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HFUploader")


def upload_to_huggingface(
    repo_id: str,
    token: str | None = None,
    dataset_dir: Path = Path("dataset_output/parquet"),
    dataset_card_path: Path = Path("reports/DATASET_CARD.md"),
    private: bool = False,
):
    """
    Upload entire RICC dataset and documentation to Hugging Face Datasets Hub.
    """
    api = HfApi(token=token)
    logger.info(f"🚀 Initializing Hugging Face upload to repository: {repo_id}...")

    # 1. Create or ensure repository exists
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    logger.info(f"✅ Repository {repo_id} verified on Hugging Face Datasets.")

    # 2. Upload Dataset Card (README.md on HF)
    if dataset_card_path.exists():
        logger.info(f"📄 Uploading Dataset Card from {dataset_card_path}...")
        api.upload_file(
            path_or_fileobj=str(dataset_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("✅ Dataset Card published as README.md on Hugging Face.")

    # 3. Upload Parquet Datasets
    if dataset_dir.exists():
        logger.info(f"📦 Uploading Parquet datasets from {dataset_dir}...")
        for pfile in sorted(dataset_dir.glob("*.parquet")):
            size_mb = pfile.stat().st_size / (1024 * 1024)
            logger.info(f" - Uploading {pfile.name} ({size_mb:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(pfile),
                path_in_repo=f"data/{pfile.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
        logger.info("✅ All Parquet partitions uploaded successfully.")

    logger.info(f"🎉 Publication complete! View your dataset at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload RICC to Hugging Face Hub")
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Target HF repo e.g. username/russian-it-community-corpus"
    )
    parser.add_argument("--token", type=str, default=None, help="HF Write Access Token")
    parser.add_argument("--private", action="store_true", help="Make dataset repository private")
    args = parser.parse_args()

    upload_to_huggingface(repo_id=args.repo_id, token=args.token, private=args.private)
