#!/usr/bin/env python3
"""Regenerate and validate the portable runtime dependency lock.

The lock is the dependency closure installed by CI and Docker. Unlike the
former ``pip freeze`` snapshot, it has environment markers and is resolved
for every supported Python/OS combination, not for the maintainer's PC.

Regenerate (requires uv)::

    python scripts/lock_requirements.py

Check without changing the file::

    python scripts/lock_requirements.py --check

The check runs the resolver and compares the output byte-for-byte. It also
checks that requirements.txt and pyproject declare the same runtime roots
and that every direct locked pin satisfies its declared version specifier.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
LOCK_PATH = REPO_ROOT / "requirements.lock.txt"

HEADER = """# Portable lock for Python 3.11+ on Linux, Windows and macOS.
# This file, not the unconstrained ranges, is installed by CI and Docker.
# Regenerate after reviewing dependency changes:
# uv pip compile requirements.txt --universal --python-version 3.11 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match --no-annotate --no-header --output-file requirements.lock.txt
# Verify without modifying the lock: python scripts/lock_requirements.py --check

"""

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
    "--no-annotate",
    "--no-header",
    "--quiet",
    "--output-file",
    "requirements.lock.resolved.txt",
]


def _declared_requirements() -> list[Requirement]:
    lines = [line.split("#", 1)[0].strip() for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()]
    return [Requirement(line) for line in lines if line]


def _locked_requirements(text: str) -> list[Requirement]:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return [Requirement(line) for line in lines]


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


def _resolve() -> subprocess.CompletedProcess[str]:
    """Resolve the closure with uv.

    uv is required to regenerate the lock; CI installs it alongside the
    other tools. The check still validates the committed file's metadata
    even if uv is missing, but a missing uv is reported loudly instead of
    being silently swallowed.
    """
    result = subprocess.run(
        ["uv", *COMPILE_ARGS],
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
    args = parser.parse_args()
    resolved_path = REPO_ROOT / "requirements.lock.resolved.txt"
    result = _resolve()
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
