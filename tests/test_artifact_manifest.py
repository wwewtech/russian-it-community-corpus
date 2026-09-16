"""Synthetic-only tests for artifact snapshots; never read repository datasets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.validation import artifact_manifest as am


def _write(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"id": pa.array([value, value + 1], type=pa.int64())}),
        path,
        compression=None,
        use_dictionary=False,
        write_statistics=False,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for relative in am.CANONICAL_PATHS:
        _write(tmp_path / relative)
    return tmp_path


def test_roundtrip(root: Path) -> None:
    result = am.create_manifest(root, Path("manifest.json"))
    assert am.verify_manifest(root, root / "manifest.json") == result
    assert result["schema_version"] == 1
    assert result["provenance"] == am.PROVENANCE
    assert len(result["artifacts"]) == 3
    for artifact in result["artifacts"]:
        path = root / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["rows"] == 2
        assert artifact["schema"] == [{"name": "id", "type": "int64", "nullable": True}]
    assert "original source and training provenance are not established" in result["provenance"]


def test_same_size_and_rows_tampering(root: Path) -> None:
    am.create_manifest(root, Path("manifest.json"))
    path = root / am.CANONICAL_PATHS[0]
    before = path.stat()
    _write(path, 8)
    assert path.stat().st_size == before.st_size
    assert pq.read_metadata(path).num_rows == 2
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(am.ManifestError, match="does not match"):
        am.verify_manifest(root, Path("manifest.json"))


@pytest.mark.parametrize("relative", am.CANONICAL_PATHS)
@pytest.mark.parametrize("damage", ["missing", "corrupt", "directory"])
def test_missing_or_corrupt_artifact(root: Path, relative: str, damage: str) -> None:
    am.create_manifest(root, Path("manifest.json"))
    path = root / relative
    path.unlink()
    if damage == "corrupt":
        path.write_bytes(b"not parquet")
    elif damage == "directory":
        path.mkdir()
    for operation in (am.create_manifest, am.verify_manifest):
        with pytest.raises(am.ManifestError):
            operation(root, Path("manifest.json"))


def test_no_empty_inventory(tmp_path: Path) -> None:
    with pytest.raises(am.ManifestError):
        am.create_manifest(tmp_path, Path("manifest.json"))
    assert not (tmp_path / "manifest.json").exists()


def test_absent_manifest(root: Path) -> None:
    with pytest.raises(am.ManifestError):
        am.verify_manifest(root, Path("absent.json"))


@pytest.mark.parametrize("payload", [b"", b"{", b"null", b"[]", b"{}", b"\xff", b'{"x":1,"x":2}'])
def test_invalid_json(root: Path, payload: bytes) -> None:
    (root / "manifest.json").write_bytes(payload)
    with pytest.raises(am.ManifestError):
        am.verify_manifest(root, Path("manifest.json"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("created_at", "not a date"),
        ("created_at", "2026-01-01"),
        ("provenance", "original training source"),
        ("artifacts", []),
        ("artifacts", None),
        ("unexpected", 1),
    ],
)
def test_invalid_manifest_fields(root: Path, key: str, value: Any) -> None:
    data: Any = am.create_manifest(root, Path("manifest.json"))
    data[key] = value
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(am.ManifestError):
        am.verify_manifest(root, Path("manifest.json"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("path", "../outside.parquet"),
        ("path", "/outside.parquet"),
        ("path", "C:\\outside.parquet"),
        ("path", "dataset_output\\parquet\\full_clean_messages.parquet"),
        ("path", "dataset_output/parquet/../parquet/full_clean_messages.parquet"),
        ("path", "dataset_output/parquet/full_clean_messages.parquet:stream"),
        ("sha256", "x" * 64),
        ("sha256", None),
        ("bytes", True),
        ("bytes", -1),
        ("rows", True),
        ("rows", -1),
        ("schema", {}),
        ("schema", [{"name": "id"}]),
        ("schema", [{"name": "id", "type": "int64", "nullable": 1}]),
        ("rows", 99),
        ("sha256", "0" * 64),
    ],
)
def test_invalid_or_mismatching_artifact_fields(root: Path, key: str, value: Any) -> None:
    data: Any = am.create_manifest(root, Path("manifest.json"))
    data["artifacts"][0][key] = value
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(am.ManifestError):
        am.verify_manifest(root, Path("manifest.json"))


def test_duplicate_inventory_entry(root: Path) -> None:
    data = am.create_manifest(root, Path("manifest.json"))
    data["artifacts"][1] = data["artifacts"][0]
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(am.ManifestError):
        am.verify_manifest(root, Path("manifest.json"))


@pytest.mark.parametrize("manifest", ["../escape.json", "dataset_output/../../escape.json", am.CANONICAL_PATHS[0]])
def test_unsafe_output_path(root: Path, manifest: str) -> None:
    with pytest.raises(am.ManifestError):
        am.create_manifest(root, Path(manifest))


def test_absolute_output_escape(root: Path) -> None:
    with pytest.raises(am.ManifestError):
        am.create_manifest(root, root.parent / "outside.json")


def _symlink(link: Path, target: Path, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform/account")


@pytest.mark.parametrize("kind", ["artifact", "parent", "manifest"])
def test_symlink_escape(root: Path, kind: str) -> None:
    am.create_manifest(root, Path("manifest.json"))
    outside = root.parent / (root.name + "-outside")
    outside.mkdir()
    if kind == "artifact":
        target = outside / "file.parquet"
        _write(target)
        link = root / am.CANONICAL_PATHS[0]
        link.unlink()
        _symlink(link, target)
    elif kind == "parent":
        directory = root / "dataset_output" / "parquet"
        directory.rename(outside / "parquet")
        _symlink(directory, outside / "parquet", directory=True)
    else:
        link = root / "manifest.json"
        target = outside / "manifest.json"
        link.rename(target)
        _symlink(link, target)
    for operation in (am.create_manifest, am.verify_manifest):
        with pytest.raises(am.ManifestError):
            operation(root, Path("manifest.json"))


def test_streaming_hash_is_bounded() -> None:
    class Guarded(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert size == am.HASH_CHUNK_BYTES
            return super().read(size)

    content = b"x" * (am.HASH_CHUNK_BYTES * 3 + 17)
    assert am._hash_stream(Guarded(content)) == (hashlib.sha256(content).hexdigest(), len(content))


@pytest.mark.parametrize("operation", [am.create_manifest, am.verify_manifest])
def test_change_during_hash(root: Path, monkeypatch: pytest.MonkeyPatch, operation: Any) -> None:
    am.create_manifest(root, Path("manifest.json"))
    original = am._hash_stream

    def changing(stream: BinaryIO) -> tuple[str, int]:
        result = original(stream)
        path = Path(stream.name)
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(am, "_hash_stream", changing)
    with pytest.raises(am.ManifestError):
        operation(root, Path("manifest.json"))


def test_prior_file_changes_during_inventory(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = am._inspect

    def changing(base: Path, relative: str) -> tuple[am.Artifact, am.Fingerprint]:
        result = original(base, relative)
        if relative == am.CANONICAL_PATHS[-1]:
            path = root / am.CANONICAL_PATHS[0]
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(am, "_inspect", changing)
    with pytest.raises(am.ManifestError, match="changed during inventory"):
        am.create_manifest(root, Path("manifest.json"))


def test_atomic_failure_preserves_existing(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    am.create_manifest(root, Path("manifest.json"))
    before = (root / "manifest.json").read_bytes()

    def fail_replace(source: str, destination: Path) -> None:
        assert Path(source).parent == destination.parent
        assert json.loads(Path(source).read_bytes())["schema_version"] == 1
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(am.os, "replace", fail_replace)
    with pytest.raises(am.ManifestError):
        am.create_manifest(root, Path("manifest.json"))
    assert (root / "manifest.json").read_bytes() == before
    assert not list(root.glob(".artifact-manifest-*.tmp"))


def test_manifest_size_bound(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(am, "MAX_MANIFEST_BYTES", 32)
    (root / "manifest.json").write_bytes(b" " * 33)
    for operation in (am.create_manifest, am.verify_manifest):
        with pytest.raises(am.ManifestError, match="size limit"):
            operation(root, Path("manifest.json"))


def test_cli(root: Path) -> None:
    command = [sys.executable, "-B", "-m", "src.validation.artifact_manifest"]
    options = ["--root", str(root), "--manifest", "manifest.json"]
    for action in ("create", "verify"):
        result = subprocess.run(command + [action] + options, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "3 canonical artifacts; snapshot only" in result.stdout
    (root / "manifest.json").unlink()
    result = subprocess.run(command + ["verify"] + options, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "failed" in result.stderr
    assert "OK" not in result.stdout
