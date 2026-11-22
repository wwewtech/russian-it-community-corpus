"""Read-only release preflight: no HF writes, no Docker push, no GitHub mutation.

Verify immutable dataset/model revisions, exact LFS and registry SHA-256 values,
strict Parquet schema, full-epoch remote metadata and repository credentials.
Build the actual wheel, sdist and catalog, then emit a complete upload plan.
Only local build/report files are written; no remote release mutation is made.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import huggingface_hub
import requests

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from ricc.validation.hub_reconciliation import PINNED_HUB_REVISION  # noqa: E402
from scripts.lora_provenance import classify_training_meta, sha256_bytes  # noqa: E402
from scripts.release_catalog import validate_registry, write_release_catalog  # noqa: E402

MODEL_REPO = "wwewtech/russian-it-community-lora"
DATASET_REPO = "wwewtech/russian-it-community-corpus"
DATASETS = ("full_clean_messages", "sft_dialogues", "rag_knowledge_base")


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for release preflight")
    return token


def verify_write_permission(identity: dict[str, Any], repo_ids: tuple[str, ...]) -> str:
    """Verify token scope and owned namespaces without a trial remote write.

    The Hub repository ``auth-check`` endpoint is a read-access check: an
    anonymous request succeeds for public repositories. It must not be used as
    evidence of write access. These releases target the user's own namespace,
    so ownership plus an explicit write-capable token can be checked read-only.
    Fine-grained tokens must expose the repository-write scope explicitly.
    """
    user = str(identity.get("name") or "")
    owners = {repo.split("/", 1)[0] for repo in repo_ids}
    if not user or owners != {user}:
        raise RuntimeError("Dry-run requires verified owner/admin access to every target Hub namespace")
    access = identity.get("auth", {}).get("accessToken", {})
    role = access.get("role")
    if role == "fineGrained" and "repo.write" not in json.dumps(access):
        raise RuntimeError("Fine-grained HF token does not declare repo.write")
    if role not in {"write", "fineGrained"}:
        raise RuntimeError("HF token must have write scope; read-only release preflight failed")
    return user


def _download_metadata(filename: str, revision: str, token: str) -> dict[str, Any]:
    response = requests.get(
        f"https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{filename}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    data = json.loads(response.text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON metadata for {filename}")
    return data


def _sha_lfs(repo_file: Any) -> str:
    lfs = repo_file.lfs
    if lfs is None or not lfs.sha256:
        raise RuntimeError(f"Missing LFS sha256 for {repo_file.rfilename}")
    return str(lfs.sha256)


def preflight(root: Path, tag: str, output: Path, token: str) -> dict[str, Any]:
    api = huggingface_hub.HfApi(token=token)
    user = verify_write_permission(api.whoami(), (MODEL_REPO, DATASET_REPO))
    for repo_id, repo_type in ((DATASET_REPO, "dataset"), (MODEL_REPO, "model")):
        api.auth_check(repo_id=repo_id, repo_type=repo_type)
    _, adapters = validate_registry(root / "lora_adapters" / "registry.json")
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if tag != f"v{version}" or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise RuntimeError(f"Release tag {tag} != package v{version}")

    model_revision = str(adapters[0]["hf_model_revision"])
    if any(adapter["hf_model_revision"] != model_revision for adapter in adapters):
        raise RuntimeError("Publishable adapters do not share one immutable Hub revision")
    dataset_info = api.dataset_info(repo_id=DATASET_REPO, revision=PINNED_HUB_REVISION, files_metadata=True)
    model_info = api.model_info(repo_id=MODEL_REPO, revision=model_revision, files_metadata=True)
    if dataset_info.sha != PINNED_HUB_REVISION or model_info.sha != model_revision:
        raise RuntimeError("Hub did not resolve the requested immutable revisions")
    dataset_files = {f.rfilename: f for f in dataset_info.siblings or []}
    remote_files = {f.rfilename for f in model_info.siblings or []}
    pinned = json.loads((root / "reports" / "pinned_hub_manifest.json").read_text(encoding="utf-8"))
    if pinned["repo_id"] != DATASET_REPO or pinned["pinned_revision"] != PINNED_HUB_REVISION:
        raise RuntimeError("Pinned dataset manifest does not match the validation revision")
    steps = [f"Hub write preflight for {user}: token scope and owned namespaces verified (GET only)"]
    for name in DATASETS:
        path = f"data/{name}.parquet"
        remote = dataset_files.get(path)
        if remote is None:
            raise RuntimeError(f"Missing published dataset {path}")
        expected = pinned["artifacts"][path]
        if remote.size != expected["bytes"] or _sha_lfs(remote) != expected["lfs_sha256"]:
            raise RuntimeError(f"Dataset {name} differs from pinned_hub_manifest.json")
        steps.append(f"verify {path}: {remote.size} bytes; LFS sha256 {_sha_lfs(remote)}")

    with tempfile.TemporaryDirectory(prefix="ricc-release-preflight-") as tmp:
        temp = Path(tmp)
        parquet_dir = temp / "dataset_output" / "parquet"
        parquet_dir.mkdir(parents=True)
        (temp / "reports").mkdir()
        cards: list[str] = []
        for adapter in adapters:
            slug = adapter["slug"]
            base = api.model_info(repo_id=adapter["base_model"], revision=adapter["base_model_revision"])
            if base.sha != adapter["base_model_revision"]:
                raise RuntimeError(f"Base model revision cannot be resolved for {slug}")
            remote_path = f"{slug}/adapter_model.safetensors"
            if remote_path not in remote_files:
                raise RuntimeError(f"Policy-compliant {slug} is not published")
            source = huggingface_hub.hf_hub_download(
                repo_id=MODEL_REPO,
                repo_type="model",
                filename=remote_path,
                revision=model_revision,
                token=token,
            )
            data = Path(source).read_bytes()
            if sha256_bytes(data) != adapter["safetensors_sha256"] or len(data) != adapter["safetensors_bytes"]:
                raise RuntimeError(f"Adapter digest/size mismatch for {slug}")
            network_info = next(f for f in (model_info.siblings or []) if f.rfilename == remote_path)
            if _sha_lfs(network_info) != adapter["safetensors_sha256"]:
                raise RuntimeError(f"Remote LFS digest mismatch for {slug}")
            meta = _download_metadata(f"{slug}/training_meta.json", model_revision, token)
            if classify_training_meta(meta) != "full":
                raise RuntimeError(f"{slug} remote training provenance is not a full epoch")
            if (
                meta.get("dataset_total_dialogues")
                != pinned["artifacts"]["data/sft_dialogues.parquet"]["published_rows"]
            ):
                raise RuntimeError(f"{slug} was not trained on the pinned dialogue population")
            cards.append(
                f"{slug}: full epoch, 100 % coverage, exact weights, base revision and remote metadata verified"
            )
            steps.append(f"{remote_path}: exact registry/LFS digest verified")
            steps.append(f"{adapter['base_model']}: immutable {adapter['base_model_revision']} verified")

        for name in DATASETS:
            cached = huggingface_hub.hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=PINNED_HUB_REVISION,
                filename=f"data/{name}.parquet",
                token=token,
            )
            target = parquet_dir / f"{name}.parquet"
            shutil.copyfile(cached, target)
            remote = dataset_files[f"data/{name}.parquet"]
            if sha256_bytes(target.read_bytes()) != _sha_lfs(remote):
                raise RuntimeError(f"Dataset {name} content does not match remote LFS digest")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "ricc.validation.artifact_manifest",
                "create",
                "--root",
                str(temp),
                "--manifest",
                "reports/dataset_manifest.json",
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ricc.validation.artifact_manifest",
                "verify",
                "--root",
                str(temp),
                "--manifest",
                "reports/dataset_manifest.json",
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_real_corpus.py",
                "--root",
                str(temp / "dataset_output"),
                "--reports",
                str(temp / "reports"),
            ],
            cwd=root,
            check=True,
        )
        from ricc.validation.hub_reconciliation import generate_reconciliation_report

        generate_reconciliation_report(
            temp / "reports/dataset_manifest.json",
            root / "reports/pinned_hub_manifest.json",
            temp / "reports/dataset_reconciliation_report.json",
            online=False,
            strict=True,
        )
        package_dist = temp / "dist"
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(package_dist)], cwd=root, check=True)
        wheels = sorted(package_dist.glob("*.whl"))
        sdists = sorted(package_dist.glob("*.tar.gz"))
        expected = {f"ricc-{version}-py3-none-any.whl", f"ricc-{version}.tar.gz"}
        if {p.name for p in (*wheels, *sdists)} != expected or len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError("Expected exactly the tagged wheel and sdist")
        install_root = temp / "wheel-smoke"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_root), str(wheels[0])],
            cwd=temp,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import ricc; "
                "assert ricc.__version__ == sys.argv[2]; print('isolated wheel import:', ricc.__version__)",
                str(install_root),
                version,
            ],
            cwd=temp,
            check=True,
        )
        dist = root / "dist"
        dist.mkdir(exist_ok=True)
        for package in (*wheels, *sdists):
            shutil.copyfile(package, dist / package.name)

        from scripts.audit_hf_zoo import audit

        audit_report = audit(MODEL_REPO, token=token)
        if audit_report["revision"] != model_revision:
            raise RuntimeError("Hub revision changed since the registry publication pin")
        final_entries = [a for a in audit_report["adapters"] if a["final"]]
        if [a["id"] for a in final_entries] != sorted(a["slug"] for a in adapters):
            raise RuntimeError("Remote final adapters and registry diverge")
        for entry in final_entries:
            if entry["weight_sha256"] != next(a["safetensors_sha256"] for a in adapters if a["slug"] == entry["id"]):
                raise RuntimeError("Remote final adapter digest diverges from registry")
        write_release_catalog(root, audit_report, root / "lora_adapters" / "registry.json")
        for adapter in adapters:
            catalog_text = (root / "reports" / "LORA_MODEL_ZOO.md").read_text(encoding="utf-8")
            for required in (
                adapter["slug"],
                adapter["base_model"],
                adapter["base_model_revision"],
                adapter["hf_model_revision"],
            ):
                if required not in catalog_text:
                    raise RuntimeError(f"Release catalog omitted verified identity: {required}")
        proof = {
            "version": version,
            "tag": tag,
            "hub_user": user,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": PINNED_HUB_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": model_revision,
            "dataset_artifacts": [
                {
                    "path": n,
                    "bytes": dataset_files[f"data/{n}.parquet"].size,
                    "sha256": _sha_lfs(dataset_files[f"data/{n}.parquet"]),
                }
                for n in DATASETS
            ],
            "release_artifacts": [
                {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_bytes(p.read_bytes())}
                for p in (*wheels, *sdists)
            ],
            "catalog_assets": [
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_bytes(path.read_bytes())}
                for path in (
                    root / "reports" / "LORA_MODEL_ZOO.md",
                    root / "reports" / "HF_MODEL_CARD.md",
                    root / "reports" / "hf_hub_audit.json",
                )
            ],
            "adapters": cards,
            "classification_counts": audit_report["classification_counts"],
            "checks": [
                *steps,
                "strict Parquet contract: passed",
                "manifest create/verify: passed",
                "immutable reconciliation: EXACT_MATCH",
            ],
            "write_steps_skipped": [
                "HF parquet upload",
                "HF model card upload",
                "registry upload",
                "GitHub release creation/upload",
                "Docker push",
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Release {tag} dry-run READY: {len(adapters)} final adapter; 2 package files and 3 catalog assets")
        print("No HF/GitHub/Docker mutations performed")
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = args.out or (args.root / "reports" / "release_dry_run.json")
    preflight(args.root, args.tag, output, _token())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
