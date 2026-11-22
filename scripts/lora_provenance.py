"""Remote LoRA provenance helpers; Hub is queried read-only throughout."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import huggingface_hub
import requests

from ricc.lora.policy_gate import evaluate_compliance
from scripts.release_catalog import published_revision


def classify_training_meta(meta: dict[str, Any] | None) -> str:
    """Classify a historical artifact without calling it a final model."""
    if not meta:
        return "legacy_no_provenance"
    policy = evaluate_meta_json(meta)
    if policy.compliant:
        return "full"
    if meta.get("time_budget_hours"):
        return "time_boxed"
    return "partial"


def evaluate_meta_json(meta: dict[str, Any]) -> Any:
    """Reuse the binding policy without downloading or writing Hub files."""
    with tempfile.TemporaryDirectory(prefix="ricc-provenance-") as folder:
        root = Path(folder)
        (root / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return evaluate_compliance(root)


def download_text(
    repo_id: str,
    filename: str,
    *,
    repo_type: str,
    token: str | None,
    revision: str | None = None,
) -> str | None:
    """Download only small metadata, never model weights."""
    url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def fetch_hub_zoo_records(
    repo_id: str, *, token: str | None = None, repo_type: str = "model"
) -> tuple[str, list[dict[str, Any]]]:
    """Collect authoritative remote tree metadata, provenance, and weight hashes."""
    api = huggingface_hub.HfApi(token=token)
    registry_path = Path(__file__).resolve().parents[1] / "lora_adapters" / "registry.json"
    revision = published_revision(registry_path) if repo_type == "model" else None
    info = (
        api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        if repo_type == "model"
        else api.dataset_info(repo_id=repo_id, files_metadata=True)
    )
    siblings = info.siblings or []
    folders = {item.rfilename for item in siblings if item.rfilename.endswith("/adapter_model.safetensors")}
    registry_path = Path(__file__).resolve().parents[1] / "lora_adapters" / "registry.json"
    local_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for weight_path in sorted(folders):
        slug = weight_path.split("/")[0]
        nested = len(weight_path.split("/")) == 3
        prefix = "/".join(weight_path.split("/")[:-1])
        meta_text = download_text(
            repo_id, f"{prefix}/training_meta.json", repo_type=repo_type, token=token, revision=revision
        )
        meta: dict[str, Any] | None = None
        error = None
        if meta_text:
            try:
                candidate = json.loads(meta_text)
                if not isinstance(candidate, dict):
                    raise TypeError("training_meta.json must be an object")
                meta = candidate
            except (json.JSONDecodeError, TypeError) as exc:
                error = str(exc)
        local_classification = classify_training_meta(meta)
        classification = "legacy_nested_checkpoint" if nested else local_classification
        weight = next(item for item in siblings if item.rfilename == weight_path)
        expected_sha: str | None = None
        local_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        row = next((a for a in local_registry["adapters"] if a["slug"] == slug), None)
        if row:
            expected_sha = row.get("safetensors_sha256")
        actual_lfs_sha = weight.lfs.sha256 if weight.lfs else None
        complete = (
            not nested and actual_lfs_sha is not None and (weight.size or 0) > 0 and meta is not None and error is None
        )
        if expected_sha and actual_lfs_sha != expected_sha:
            complete = False
        config = download_text(
            repo_id, f"{prefix}/adapter_config.json", repo_type=repo_type, token=token, revision=revision
        )
        base_model = None
        if config:
            try:
                base_model = json.loads(config).get("base_model_name_or_path")
            except json.JSONDecodeError:
                error = "invalid adapter_config.json"
        if classification == "full" and expected_sha != actual_lfs_sha:
            classification = "full_mismatched_digest"
            complete = False
        compliance = evaluate_meta_json(meta) if meta else None
        entries.append(
            {
                "id": slug,
                "subfolder": prefix,
                "base_model": base_model,
                "base_model_revision": row.get("base_model_revision") if row else None,
                "hf_model_revision": published_revision(registry_path),
                "classification": classification,
                "final": classification == "full" and complete,
                "complete": complete,
                "provenance_error": error,
                "actual_steps": meta.get("actual_steps") if meta else None,
                "required_steps": compliance.required_steps if compliance else None,
                "coverage_pct": meta.get("corpus_coverage_pct") if meta else None,
                "train_dialogues": meta.get("train_dialogues") if meta else None,
                "time_budget_hours": meta.get("time_budget_hours") if meta else None,
                "weight_bytes": weight.size,
                "weight_sha256": actual_lfs_sha,
                "registry_sha256": expected_sha,
                "sha256_matches": actual_lfs_sha == expected_sha if expected_sha else None,
                "reasons": compliance.reasons if compliance else [],
            }
        )
    if info.sha is None or repo_type != "model":
        raise RuntimeError(f"No immutable revision returned for {repo_id}")
    return info.sha, entries


def download_bytes(url: str, *, token: str | None = None, timeout: int = 180) -> bytes:
    """Fetch a release artifact over HTTP with authentication when supplied."""
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=timeout)
    response.raise_for_status()
    return response.content


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "classify_training_meta",
    "download_bytes",
    "download_text",
    "evaluate_meta_json",
    "fetch_hub_zoo_records",
    "sha256_bytes",
]
