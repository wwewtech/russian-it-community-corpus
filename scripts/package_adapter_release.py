"""Package a policy-compliant adapter and its evaluation provenance safely.

The archive includes the actual adapter weights (not a placeholder card) and
all raw/recomputed evidence. Every source file is hashed before packaging;
the adapter must pass the binding full-epoch policy or no package is created.
Run: python scripts/package_adapter_release.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from ricc.lora.policy_gate import assert_publishable

REPO = Path(__file__).resolve().parent.parent
ADAPTER = REPO / "lora_adapters" / "smollm2_135m_instruct"
REQUIRED_REPORTS: tuple[str, ...] = (
    "academic_scientific_benchmarks_matrix.json",
    "academic_benchmarks_raw_outputs.jsonl",
    "benchmark_smollm2_135m_instruct.json",
    "benchmark_smollm2_135m_instruct.md",
    "OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md",
    "RELEASE_QC_SMOLLM2_135M.md",
    "adapter_release_digest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_release_archive(
    adapter: Path = ADAPTER,
    output: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, object]:
    """Create a byte-hashed release archive, failing closed for partial adapters."""
    assert_publishable(adapter)

    metadata_path = adapter / "training_meta.json"
    training = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_files = sorted(p for p in adapter.iterdir() if p.is_file())
    if "adapter_model.safetensors" not in {p.name for p in source_files}:
        raise RuntimeError("no actual adapter weights: refusing to package a metadata-only placeholder")

    reports_root = reports_dir if reports_dir is not None else REPO / "reports"
    source_archive_files = [(PurePosixPath("lora_adapters") / adapter.name / p.name, p) for p in source_files]
    for filename in REQUIRED_REPORTS:
        path = reports_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing release evidence file: {path}")
        source_archive_files.append((PurePosixPath("reports") / filename, path))

    entries: list[tuple[str, bytes]] = []
    digest_entries: dict[str, tuple[str, int]] = {}
    for relative, path in source_archive_files:
        payload = path.read_bytes()
        rel_key = str(relative)
        entries.append((rel_key, payload))
        digest_entries[rel_key] = (hashlib.sha256(payload).hexdigest(), len(payload))

    output = output or REPO / "dist" / f"ricc-adapter-{adapter.name}-provenance.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Archive bytes are independent of local mtimes and host/system metadata:
    # fixed ZipInfo date/fields plus sorted entries gives reproducible digests
    # on Windows and Linux, not just equal file counts.
    fixed_epoch = (2026, 1, 1, 0, 0, 0)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, payload in entries:
            info = ZipInfo(name, fixed_epoch)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload, compress_type=ZIP_DEFLATED)
        manifest = {
            "adapter": adapter.name,
            "adapter_training_steps": training["actual_steps"],
            "required_steps": training["required_steps_for_full_epoch"],
            "corpus_coverage_pct": training["corpus_coverage_pct"],
            "heldout_tail_dialogues": training["holdout_tail_dialogues"],
            "dataset_sha256": training["dataset_sha256"],
            "source_sha256": {name: sha for name, (sha, size) in sorted(digest_entries.items())},
            "source_bytes": {name: size for name, (sha, size) in sorted(digest_entries.items())},
        }
        provenance_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        info = ZipInfo("PROVENANCE.json", fixed_epoch)
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        bundle.writestr(info, provenance_payload, compress_type=ZIP_DEFLATED)

    with ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError(f"archive integrity test failed: {output}")
        packaged_count = len(bundle.namelist())
    try:
        archive_ref = output.relative_to(REPO).as_posix()
    except ValueError:
        # Any outputs outside the checkout remain in a platform-neutral
        # absolute representation rather than hiding their destination.
        archive_ref = str(output)
    return {
        "archive": archive_ref,
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "file_count": packaged_count,
        "adapter": adapter.name,
        "training_steps": training["actual_steps"],
        "coverage_pct": training["corpus_coverage_pct"],
        "source_file_count": len(entries),
    }


def main() -> None:
    manifest = create_release_archive()
    path = REPO / "dist" / "RELEASE_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
