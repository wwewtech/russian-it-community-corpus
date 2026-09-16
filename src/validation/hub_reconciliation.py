"""Reconciliation and verification of local dataset artifacts against pinned Hugging Face Hub revision."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger("HubReconciliation")

PINNED_HUB_REPO_ID = "wwewtech/russian-it-community-corpus"
PINNED_HUB_REVISION = "81a3495de997b0690f67d175c4ebd9cfdef35b76"

# Canonical mappings between local path and Hub remote path
PATH_MAPPINGS: dict[str, str] = {
    "dataset_output/parquet/full_clean_messages.parquet": "data/full_clean_messages.parquet",
    "dataset_output/parquet/sft_dialogues.parquet": "data/sft_dialogues.parquet",
    "dataset_output/parquet/rag_knowledge_base.parquet": "data/rag_knowledge_base.parquet",
}


class HubArtifactRecord(TypedDict):
    path: str
    bytes: int
    lfs_sha256: str
    published_rows: int


class ArtifactDelta(TypedDict):
    local_path: str
    hub_path: str
    local_sha256: str
    hub_lfs_sha256: str
    hashes_match: bool
    local_bytes: int
    hub_bytes: int
    delta_bytes: int
    local_rows: int
    hub_rows: int
    delta_rows: int
    divergence_reason: str


class ReconciliationReport(TypedDict):
    schema_version: int
    generated_at: str
    status: str
    repo_id: str
    pinned_revision: str
    online_verified: bool
    artifacts_analyzed: int
    artifacts: list[ArtifactDelta]
    summary: str


def load_pinned_hub_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the machine-readable pinned Hub manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pinned Hub manifest not found at: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest content must be a JSON object")
    return data


def fetch_remote_hub_tree(
    repo_id: str = PINNED_HUB_REPO_ID,
    revision: str = PINNED_HUB_REVISION,
) -> dict[str, dict[str, Any]]:
    """Fetch live metadata for pinned revision from Hugging Face Hub."""
    import huggingface_hub

    api = huggingface_hub.HfApi()
    files = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True)
    remote_tree: dict[str, dict[str, Any]] = {}
    for f in files:
        path = getattr(f, "path", str(f))
        size = getattr(f, "size", 0) or 0
        lfs_info = getattr(f, "lfs", None)
        lfs_sha = getattr(lfs_info, "sha256", "") if lfs_info else ""
        remote_tree[path] = {
            "path": path,
            "bytes": size,
            "lfs_sha256": lfs_sha,
        }
    return remote_tree


def reconcile_artifacts(
    local_manifest: dict[str, Any],
    hub_manifest: dict[str, Any],
    strict: bool = False,
) -> ReconciliationReport:
    """Compute exact numerical deltas between local artifacts and pinned Hub revision."""
    local_artifacts: list[dict[str, Any]] = local_manifest.get("artifacts", [])
    local_by_path: dict[str, dict[str, Any]] = {a["path"]: a for a in local_artifacts}

    hub_artifacts: dict[str, Any] = hub_manifest.get("artifacts", {})

    deltas: list[ArtifactDelta] = []
    all_matched = True

    for local_path, hub_path in PATH_MAPPINGS.items():
        loc = local_by_path.get(local_path)
        if loc is None:
            raise ValueError(f"Local artifact missing from local manifest: {local_path}")

        hub_meta = hub_artifacts.get(hub_path, {})
        hub_sha = hub_meta.get("lfs_sha256", "")
        hub_bytes = int(hub_meta.get("bytes", 0))
        hub_rows = int(hub_meta.get("published_rows", 0))

        loc_sha = str(loc.get("sha256", ""))
        loc_bytes = int(loc.get("bytes", 0))
        loc_rows = int(loc.get("rows", 0))

        hashes_match = (loc_sha.lower() == hub_sha.lower()) and bool(loc_sha)
        delta_bytes = loc_bytes - hub_bytes
        delta_rows = loc_rows - hub_rows

        if hashes_match and delta_bytes == 0:
            reason = "IDENTICAL_TO_HUB_RELEASE"
        else:
            all_matched = False
            reason = (
                "LOCAL_WORKING_COPY_DIVERGES: local pipeline output differs from pinned Hub "
                f"snapshot {PINNED_HUB_REVISION[:8]} (delta: {delta_rows:+d} rows, {delta_bytes:+d} bytes)"
            )

        deltas.append(
            ArtifactDelta(
                local_path=local_path,
                hub_path=hub_path,
                local_sha256=loc_sha,
                hub_lfs_sha256=hub_sha,
                hashes_match=hashes_match,
                local_bytes=loc_bytes,
                hub_bytes=hub_bytes,
                delta_bytes=delta_bytes,
                local_rows=loc_rows,
                hub_rows=hub_rows,
                delta_rows=delta_rows,
                divergence_reason=reason,
            )
        )

    if strict and not all_matched:
        status = "STRICT_MISMATCH"
    elif all_matched:
        status = "EXACT_MATCH"
    else:
        status = "DOCUMENTED_DIVERGENCE"

    summary = (
        f"Reconciled {len(deltas)} canonical artifacts against pinned Hub revision {PINNED_HUB_REVISION[:8]}. "
        f"Verdict: {status}."
    )

    return ReconciliationReport(
        schema_version=1,
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        repo_id=PINNED_HUB_REPO_ID,
        pinned_revision=PINNED_HUB_REVISION,
        online_verified=False,
        artifacts_analyzed=len(deltas),
        artifacts=deltas,
        summary=summary,
    )


def generate_reconciliation_report(
    local_manifest_path: Path,
    hub_manifest_path: Path,
    output_report_path: Path,
    online: bool = False,
    strict: bool = False,
) -> ReconciliationReport:
    """Generate and write the machine-readable reconciliation report."""
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    hub_manifest = load_pinned_hub_manifest(hub_manifest_path)

    online_verified = False
    if online:
        try:
            remote_tree = fetch_remote_hub_tree(PINNED_HUB_REPO_ID, PINNED_HUB_REVISION)
            # Cross-verify remote tree with pinned manifest
            for hub_path, expected in hub_manifest.get("artifacts", {}).items():
                if hub_path in remote_tree:
                    actual = remote_tree[hub_path]
                    if actual["bytes"] != expected["bytes"] or actual["lfs_sha256"] != expected["lfs_sha256"]:
                        raise ValueError(f"Remote HF Hub mismatch for {hub_path} at revision {PINNED_HUB_REVISION}")
            online_verified = True
            logger.info("Online verification against Hugging Face API succeeded.")
        except Exception as exc:
            logger.warning(f"Online Hub query failed or could not verify: {exc}")

    report = reconcile_artifacts(local_manifest, hub_manifest, strict=strict)
    report["online_verified"] = online_verified

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Wrote reconciliation report to {output_report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile local dataset artifacts with pinned HF Hub revision")
    parser.add_argument("--local-manifest", type=Path, default=Path("reports/dataset_manifest.json"))
    parser.add_argument("--hub-manifest", type=Path, default=Path("reports/pinned_hub_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_reconciliation_report.json"))
    parser.add_argument("--online", action="store_true", help="Fetch and verify remote metadata via HF API")
    parser.add_argument("--strict", action="store_true", help="Fail if local files do not match Hub byte-for-byte")
    args = parser.parse_args()

    try:
        report = generate_reconciliation_report(
            args.local_manifest,
            args.hub_manifest,
            args.report,
            online=args.online,
            strict=args.strict,
        )
        print(f"Reconciliation Status: {report['status']}")
        for art in report["artifacts"]:
            print(
                f" - {art['local_path']}: local={art['local_rows']} rows ({art['local_bytes']} B) vs "
                f"hub={art['hub_rows']} rows ({art['hub_bytes']} B) -> delta={art['delta_rows']:+d} rows"
            )
        if args.strict and report["status"] != "EXACT_MATCH":
            return 1
        return 0
    except Exception as exc:
        logger.error(f"Reconciliation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
