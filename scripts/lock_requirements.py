#!/usr/bin/env python3
"""
lock_requirements.py — capture exact versions of the project's dependency tree.

The local machine's ``pip freeze`` is NOT usable as a lock: it contains
unrelated packages with conflicting pins (verified on 2026-10-07: conflicts
in fairseq, rvc-python, google-* - none of which are RICC dependencies).

This script walks only the transitive closure of ``requirements.txt`` via
importlib.metadata and writes ``requirements.lock.txt``.

The lock records the versions this project was actually developed and
tested against. It is a REFERENCE snapshot: it was captured on the platform
recorded in its header and must be verified on Linux before being enforced
in CI (local CUDA torch builds carry ``+cuXXX`` local version segments that
do not exist on PyPI).

Run::

    python scripts/lock_requirements.py           # (re)write requirements.lock.txt
    python scripts/lock_requirements.py --check   # CI: fail if lock missing a root dep
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

try:
    from packaging.markers import Marker
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - packaging ships with pip/setuptools
    Marker = None  # type: ignore[assignment]
    Requirement = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
LOCK_PATH = REPO_ROOT / "requirements.lock.txt"

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _root_names() -> list[str]:
    names: list[str] = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQ_LINE.match(line)
        if match:
            names.append(match.group(1))
    return names


def _installed_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        version = dist.version or "0"
        index.setdefault(_canonical(name), (name, version))
    return index


def _requirements_of(name: str) -> list[str]:
    try:
        reqs = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return []
    out: list[str] = []
    for raw in reqs:
        # Skip extras-only and environment-provided requirements.
        if ";" in raw and "extra" in raw.split(";", 1)[1]:
            continue
        if Requirement is not None:
            try:
                requirement = Requirement(raw)
            except Exception:
                continue
            if requirement.marker is not None and Marker is not None:
                try:
                    if not requirement.marker.evaluate():
                        continue
                except Exception:
                    pass  # unknown marker: keep conservatively
            out.append(requirement.name)
        else:  # pragma: no cover
            out.append(_REQ_LINE.match(raw).group(1))  # type: ignore[union-attr]
    return out


def closure() -> dict[str, str]:
    index = _installed_index()
    seen: set[str] = set()
    queue = list(_root_names())
    resolved: dict[str, str] = {}
    missing: list[str] = []
    while queue:
        name = queue.pop()
        key = _canonical(name)
        if key in seen:
            continue
        seen.add(key)
        if key not in index:
            missing.append(name)
            continue
        display, version = index[key]
        resolved[display] = version
        queue.extend(_requirements_of(display))
    if missing:
        print(f"WARN: not installed, excluded from lock: {sorted(set(missing))}", file=sys.stderr)
    return dict(sorted(resolved.items(), key=lambda kv: kv[0].lower()))


def render() -> str:
    resolved = closure()
    header = [
        "# requirements.lock.txt - exact versions of the requirements.txt transitive closure.",
        f"# Captured: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# Captured on: Python {platform.python_version()} / {sys.platform} / {platform.machine()}",
        "#",
        "# REFERENCE SNAPSHOT: verify on Linux CI before enforcing with -c.",
        "# Local CUDA builds (torch==X.Y.Z+cuNNN) do not exist on PyPI.",
        "# Regenerate: python scripts/lock_requirements.py",
        "",
    ]
    body = [f"{name}=={version}" for name, version in resolved.items()]
    return "\n".join(header + body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CI: verify every root dep appears in the lock")
    args = parser.parse_args()

    if args.check:
        if not LOCK_PATH.exists():
            print("FAIL: requirements.lock.txt missing", file=sys.stderr)
            return 1
        locked = {
            _canonical(line.split("==")[0])
            for line in LOCK_PATH.read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.startswith("#")
        }
        missing = [name for name in _root_names() if _canonical(name) not in locked]
        if missing:
            print(f"FAIL: root deps missing from lock: {missing}", file=sys.stderr)
            return 1
        print(f"OK: {len(_root_names())} root deps present in lock ({len(locked)} entries total)")
        return 0

    LOCK_PATH.write_text(render(), encoding="utf-8")
    lines = LOCK_PATH.read_text(encoding="utf-8").splitlines()
    pin_count = sum(1 for line in lines if "==" in line)
    print(f"wrote {LOCK_PATH.name}: {pin_count} pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
