#!/usr/bin/env python3
"""Fetch and verify the exact immutable dataset release for CI or publication.

No Hub mutation is performed. Every downloaded byte is checked against the
committed revision/size/SHA-256 snapshot before any downstream gate sees it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ricc.validation.hub_reconciliation import PATH_MAPPINGS, PINNED_HUB_REVISION  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    pinned = json.loads((args.root / "reports/pinned_hub_manifest.json").read_text(encoding="utf-8"))
    if pinned["pinned_revision"] != PINNED_HUB_REVISION:
        raise RuntimeError("Dataset manifest revision and reconciliation constant diverge")
    repo = pinned["repo_id"]
    token = os.environ.get("HF_TOKEN") or None
    info = HfApi(token=token).dataset_info(repo_id=repo, revision=PINNED_HUB_REVISION, files_metadata=True)
    if info.sha != PINNED_HUB_REVISION:
        raise RuntimeError("HF did not resolve the pinned immutable revision")
    siblings = {file.rfilename: file for file in info.siblings or []}
    for local_path, remote_path in PATH_MAPPINGS.items():
        expected = pinned["artifacts"][remote_path]
        remote = siblings[remote_path]
        if remote.size != expected["bytes"] or remote.lfs is None or remote.lfs.sha256 != expected["lfs_sha256"]:
            raise RuntimeError(f"Remote dataset metadata mismatch: {remote_path}")
        cached = hf_hub_download(
            repo_id=repo,
            filename=remote_path,
            revision=PINNED_HUB_REVISION,
            repo_type="dataset",
            token=token,
        )
        data = Path(cached).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["lfs_sha256"]:
            raise RuntimeError(f"Downloaded dataset content mismatch: {remote_path}")
        target = args.root / local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        print(f"verified {remote_path} -> {target} ({len(data)} bytes, SHA-256 match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
