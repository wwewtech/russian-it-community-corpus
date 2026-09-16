"""Local artifact identity snapshots; not original source or training provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TypedDict, cast

import pyarrow.parquet as pq

CANONICAL_PATHS = tuple(
    f"dataset_output/parquet/{name}.parquet" for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base")
)
SCHEMA_VERSION = 1
PROVENANCE = "Local artifact snapshot only; original source and training provenance are not established."
HASH_CHUNK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class ManifestError(ValueError):
    """A snapshot could not be safely created or verified."""


class FieldSummary(TypedDict):
    name: str
    type: str
    nullable: bool


class Artifact(TypedDict):
    path: str
    sha256: str
    bytes: int
    rows: int
    schema: list[FieldSummary]


class Manifest(TypedDict):
    schema_version: int
    created_at: str
    provenance: str
    artifacts: list[Artifact]


Fingerprint = tuple[int, int, int, int]


def _fingerprint(info: os.stat_result) -> Fingerprint:
    # st_ctime (creation time on Windows) is intentionally excluded:
    # NTFS can report a slightly newer creation time via fstat() than via
    # path.stat() right after the file is written, which breaks comparisons.
    # Content identity is guaranteed by the sha256 check, not by ctime.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _root(root: Path) -> Path:
    result = root.resolve(strict=True)
    if not result.is_dir():
        raise ManifestError("Root must be an existing directory")
    return result


def _safe_path(root: Path, path: Path) -> Path:
    """Reject traversal and all descendant symlinks/reparse points, including output."""
    if ".." in path.parts:
        raise ManifestError("Parent traversal is forbidden")
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError("Path must be inside root") from exc
    current = root
    for part in relative.parts:
        if ":" in part:
            raise ManifestError("Alternate streams and drive paths are forbidden")
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ManifestError("Symlinks and reparse points below root are forbidden")
    if not candidate.resolve().is_relative_to(root):
        raise ManifestError("Path escapes root")
    return candidate


def _regular(path: Path) -> os.stat_result:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ManifestError("Expected a regular file")
    return info


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _inspect(root: Path, relative: str) -> tuple[Artifact, Fingerprint]:
    path = _safe_path(root, Path(relative))
    before = _fingerprint(_regular(path))
    with path.open("rb") as stream:
        if _fingerprint(os.fstat(stream.fileno())) != before:
            raise ManifestError("Artifact changed before hashing")
        digest, size = _hash_stream(stream)
        stream.seek(0)
        parquet = pq.ParquetFile(stream)
        rows = parquet.metadata.num_rows
        schema: list[FieldSummary] = [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in parquet.schema_arrow
        ]
        if _fingerprint(os.fstat(stream.fileno())) != before or size != before[2]:
            raise ManifestError("Artifact changed during hashing or metadata inspection")
    if _fingerprint(_regular(_safe_path(root, Path(relative)))) != before:
        raise ManifestError("Artifact replaced during inspection")
    return {"path": relative, "sha256": digest, "bytes": size, "rows": rows, "schema": schema}, before


def _inventory(root: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    fingerprints: list[Fingerprint] = []
    for relative in CANONICAL_PATHS:
        try:
            artifact, fingerprint = _inspect(root, relative)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ManifestError(f"Cannot inventory canonical artifact: {relative}") from exc
        artifacts.append(artifact)
        fingerprints.append(fingerprint)
    # Also catch a previously inspected file changing while later files are hashed.
    for relative, fingerprint in zip(CANONICAL_PATHS, fingerprints, strict=True):
        if _fingerprint(_regular(_safe_path(root, Path(relative)))) != fingerprint:
            raise ManifestError("Artifact changed during inventory")
    return artifacts


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError("Duplicate JSON key")
        result[key] = value
    return result


def _validate(value: Any) -> Manifest:
    def require(condition: bool) -> None:
        if not condition:
            raise ManifestError("Invalid manifest structure or unsupported schema")

    require(isinstance(value, dict))
    require(set(value) == {"schema_version", "created_at", "provenance", "artifacts"})
    require(type(value["schema_version"]) is int and value["schema_version"] == SCHEMA_VERSION)
    require(value["provenance"] == PROVENANCE)
    require(isinstance(value["created_at"], str))
    try:
        timestamp = datetime.fromisoformat(value["created_at"])
        require(timestamp.tzinfo is not None and timestamp.utcoffset() == UTC.utcoffset(None))
    except ValueError as exc:
        raise ManifestError("Invalid snapshot timestamp") from exc
    artifacts = value["artifacts"]
    require(isinstance(artifacts, list) and len(artifacts) == len(CANONICAL_PATHS))
    paths: list[str] = []
    for artifact in artifacts:
        require(isinstance(artifact, dict))
        require(set(artifact) == {"path", "sha256", "bytes", "rows", "schema"})
        require(isinstance(artifact["path"], str) and artifact["path"] in CANONICAL_PATHS)
        paths.append(artifact["path"])
        require(isinstance(artifact["sha256"], str))
        require(re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None)
        require(type(artifact["bytes"]) is int and artifact["bytes"] > 0)
        require(type(artifact["rows"]) is int and artifact["rows"] >= 0)
        require(isinstance(artifact["schema"], list))
        for field in artifact["schema"]:
            require(isinstance(field, dict) and set(field) == {"name", "type", "nullable"})
            require(isinstance(field["name"], str))
            require(isinstance(field["type"], str) and bool(field["type"]))
            require(type(field["nullable"]) is bool)
    require(set(paths) == set(CANONICAL_PATHS))
    return cast(Manifest, value)


def _manifest_path(root: Path, manifest: Path) -> Path:
    path = _safe_path(root, manifest)
    if path in [root / relative for relative in CANONICAL_PATHS]:
        raise ManifestError("Manifest must not overwrite a canonical artifact")
    return path


def _atomic_write(root: Path, path: Path, manifest: Manifest) -> None:
    payload = (json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ManifestError("Manifest exceeds size limit")
    # Require an existing parent; write on the same filesystem before atomic replacement.
    fd, temporary = tempfile.mkstemp(prefix=".artifact-manifest-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _safe_path(root, path)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_manifest(root: Path, manifest: Path) -> Manifest:
    """Snapshot all three artifacts and atomically write JSON; relative manifest is root-relative."""
    try:
        root = _root(root)
        path = _manifest_path(root, manifest)
        result: Manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "provenance": PROVENANCE,
            "artifacts": _inventory(root),
        }
        _validate(result)
        _atomic_write(root, path, result)
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError("Could not create artifact manifest") from exc


def verify_manifest(root: Path, manifest: Path) -> Manifest:
    """Fail closed unless the complete inventory exactly matches a valid saved snapshot."""
    try:
        root = _root(root)
        path = _manifest_path(root, manifest)
        before = _fingerprint(_regular(path))
        if before[2] > MAX_MANIFEST_BYTES:
            raise ManifestError("Manifest exceeds size limit")
        with path.open("rb") as stream:
            if _fingerprint(os.fstat(stream.fileno())) != before:
                raise ManifestError("Manifest replaced before reading")
            payload = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(payload) > MAX_MANIFEST_BYTES or _fingerprint(os.fstat(stream.fileno())) != before:
                raise ManifestError("Manifest changed or exceeds size limit")
        expected = _validate(json.loads(payload, object_pairs_hook=_object))
        actual = _inventory(root)
        by_path = {artifact["path"]: artifact for artifact in expected["artifacts"]}
        for artifact in actual:
            if artifact != by_path[artifact["path"]]:
                raise ManifestError(f"Artifact does not match snapshot: {artifact['path']}")
        if _fingerprint(_regular(_safe_path(root, path))) != before:
            raise ManifestError("Manifest changed during verification")
        return expected
    except (OSError, ValueError, RuntimeError, RecursionError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError("Could not verify artifact manifest") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--root", required=True, type=Path, help="Repository root containing dataset_output/parquet")
    parser.add_argument("--manifest", required=True, type=Path, help="JSON path inside root (relative to root)")
    args = parser.parse_args(argv)
    try:
        operation = create_manifest if args.command == "create" else verify_manifest
        operation(args.root, args.manifest)
    except ManifestError as exc:
        print(f"Artifact manifest failed: {exc}", file=sys.stderr)
        return 1
    print(f"Artifact manifest {args.command}: OK (3 canonical artifacts; snapshot only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
