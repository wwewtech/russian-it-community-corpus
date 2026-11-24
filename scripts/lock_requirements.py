#!/usr/bin/env python3
"""Regenerate and validate the portable runtime dependency lock.

The lock is the dependency closure installed by CI and Docker. Unlike the
former ``pip freeze`` snapshot, it has environment markers and is resolved
for every supported Python/OS combination, not for the maintainer's PC.

Regenerate (requires uv)::

    python scripts/lock_requirements.py

For a reviewed dependency update, advance the UTC resolution cutoff and lock
together (Dependabot's lock-refresh workflow runs the same command)::

    python scripts/lock_requirements.py --refresh

The resolver uses that new cutoff directly. All supported Python/OS artifacts
are resolved universally with SHA-256 hashes for wheels and sdists.

Check without changing the file::

    python scripts/lock_requirements.py --check

The check runs the resolver and compares the output byte-for-byte. It also
checks that requirements.txt and pyproject declare the same runtime roots
and that every direct locked pin satisfies its declared version specifier.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
LOCK_PATH = REPO_ROOT / "requirements.lock.txt"
PIN_PATTERN = re.compile(r"^[A-Za-z0-9._-]+==[^\s;]+(?:\s*;\s*.+)?$")
HASH_PATTERN = re.compile(r"^\s+--hash=sha256:[0-9a-f]{64}(?:\s+\\)?$")

HEADER = """# Portable lock for Python 3.11+ on Linux, Windows and macOS.
# This file, not the unconstrained ranges, is installed by CI and Docker.
# Regenerate after reviewing dependency changes:
# python scripts/lock_requirements.py
# Verify without modifying the lock: python scripts/lock_requirements.py --check

"""

# New transitive releases must not make an unchanged, previously green commit
# fail the reproducibility gate. All indexes are therefore cut off here; bump
# this timestamp deliberately together with a reviewed lock refresh.
EXCLUDE_NEWER = "2026-11-23T22:33:05Z"
COMPILE_ARGS = [
    "pip",
    "compile",
    "requirements.txt",
    "--universal",
    "--python-version",
    "3.11",
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
    "--extra-index-url",
    "https://pypi.org/simple",
    "--index-strategy",
    "unsafe-best-match",
    "--exclude-newer",
    EXCLUDE_NEWER,
    "--no-annotate",
    "--no-header",
    "--quiet",
    "--output-file",
    "requirements.lock.resolved.txt",
]


def _declared_requirements() -> list[Requirement]:
    lines = [line.split("#", 1)[0].strip() for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()]
    return [Requirement(line) for line in lines if line]


def _logical_requirement_lines(lock_text: str) -> list[str]:
    """Validate hash continuations and return direct requirement lines."""
    blocks: list[list[str]] = []
    for raw in lock_text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        if not raw[:1].isspace():
            blocks.append([raw])
        elif not blocks:
            raise ValueError("Hash without a preceding requirement")
        else:
            blocks[-1].append(raw)
    pins: list[str] = []
    for block in blocks:
        pin = block[0].rstrip("\\ ").strip()
        if not PIN_PATTERN.fullmatch(pin):
            raise ValueError("Invalid lock pin or continuation")
        if block[-1].rstrip().endswith("\\"):
            raise ValueError(f"Unterminated lock continuation: {pin}")
        hashes = [line for line in block[1:] if HASH_PATTERN.fullmatch(line)]
        if len(hashes) != len(block) - 1 or not hashes:
            raise ValueError(f"Missing artifact SHA256 hashes for {pin}")
        pins.append(pin)
    return pins


def _locked_requirements(text: str) -> list[Requirement]:
    return [Requirement(line) for line in _logical_requirement_lines(text)]


def validate_metadata(lock_text: str) -> list[str]:
    """Return consistency errors detectable without network or installation."""
    errors: list[str] = []
    if lock_text.startswith("\ufeff"):
        errors.append("lock must be UTF-8 without a BOM")
    declared = _declared_requirements()
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_requirements = [Requirement(raw) for raw in project["project"]["dependencies"]]
    if {str(req) for req in declared} != {str(req) for req in project_requirements}:
        errors.append("requirements.txt and pyproject.toml runtime dependencies differ")
    locked = _locked_requirements(lock_text)
    for req in declared:
        candidates = [candidate for candidate in locked if candidate.name.lower() == req.name.lower()]
        if not candidates:
            errors.append(f"missing direct dependency in lock: {req.name}")
        for candidate in candidates:
            pins = list(candidate.specifier)
            if len(pins) != 1 or pins[0].operator != "==":
                errors.append(f"lock entry is not an exact pin: {candidate}")
                continue
            locked_version = pins[0].version
            if not req.specifier.contains(locked_version, prereleases=True):
                errors.append(f"locked {candidate} violates declared {req}")
    return errors


def _resolve(cutoff: str = EXCLUDE_NEWER) -> subprocess.CompletedProcess[str]:
    """Resolve the closure with uv.

    uv is required to regenerate the lock; CI installs it alongside the
    other tools. The check still validates the committed file's metadata
    even if uv is missing, but a missing uv is reported loudly instead of
    being silently swallowed.
    """
    compile_args = [*COMPILE_ARGS]
    compile_args[compile_args.index("--exclude-newer") + 1] = cutoff
    result = subprocess.run(
        ["uv", "--no-config", *compile_args, "--generate-hashes"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "uv: command not found" in stderr or "No such file or directory: 'uv'" in stderr:
            raise RuntimeError(
                "uv is required to verify requirements.lock.txt; install it "
                "(e.g. `pip install uv`) before running the lock check."
            )
        sys.stderr.write(stderr)
        result.check = False  # type: ignore[attr-defined]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="resolve and compare without changing the lock")
    parser.add_argument("--refresh", action="store_true", help="advance the reviewed cutoff before resolving hashes")
    parser.add_argument("--cutoff", help="explicit UTC timestamp for a reviewed dependency refresh")
    args = parser.parse_args()
    cutoff = args.cutoff or EXCLUDE_NEWER
    if args.refresh or args.cutoff:
        if not args.cutoff:
            cutoff = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        if "." in cutoff or not cutoff.endswith("Z"):
            raise RuntimeError("Cutoff must be UTC RFC3339 with whole seconds, e.g. 2026-11-20T00:00:00Z")
        source = Path(__file__).read_text(encoding="utf-8")
        updated = re.sub(r'EXCLUDE_NEWER = "[^"]+"', f'EXCLUDE_NEWER = "{cutoff}"', source, count=1)
        if updated == source:
            raise RuntimeError("Cannot update lock cutoff")
        Path(__file__).write_text(updated, encoding="utf-8", newline="\n")
    resolved_path = REPO_ROOT / "requirements.lock.resolved.txt"
    result = _resolve(cutoff)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    try:
        resolved = resolved_path.read_text(encoding="utf-8")
        errors = validate_metadata(resolved)
        if errors:
            print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
            return 1
        current = LOCK_PATH.read_text(encoding="utf-8") if LOCK_PATH.exists() else ""
        if current != HEADER + resolved:
            if args.check:
                print("FAIL: requirements.lock.txt is stale; regenerate it", file=sys.stderr)
                return 1
            LOCK_PATH.write_text(HEADER + resolved, encoding="utf-8", newline="\n")
        if args.check:
            print(f"OK: reproducible lock verified ({len(_locked_requirements(resolved))} pins)")
        else:
            print(f"wrote {LOCK_PATH.name} ({len(_locked_requirements(resolved))} pins)")
        return 0
    finally:
        resolved_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
