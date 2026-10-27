#!/usr/bin/env python3
"""Refresh the raw-export dependency list of the DVC ``ingest`` stage.

DVC does not expand globs in a ``deps:`` entry — ``ChatExport_*/result.json``
is recorded literally and then reported as a *deleted* dependency, so the
stage could never detect a change in the corpus. The ingest stage therefore
lists every export explicitly, and this script regenerates that list.

Run it after adding a new ``ChatExport_<date>/`` directory::

    python scripts/refresh_dvc_deps.py           # rewrite dvc.yaml
    python scripts/refresh_dvc_deps.py --check   # CI: fail if out of sync
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DVC_YAML = REPO_ROOT / "dvc.yaml"

# The comment block that precedes the generated list.
MARKER_START = "      # DVC does not expand globs in `deps:`, so every export is listed\n"
MARKER_END = "      # list (scripts/refresh_dvc_deps.py).\n"

# A dependency line inside the generated block.
EXPORT_LINE = re.compile(r"^      - ChatExport_[^/\s]+/result\.json$", re.MULTILINE)


def _export_dirs() -> list[Path]:
    return sorted(p.parent for p in REPO_ROOT.glob("ChatExport_*/result.json"))


def _current_block() -> str | None:
    """The generated block as it currently stands in dvc.yaml.

    The block is the marker comment plus the consecutive dependency lines
    that follow it.
    """
    text = DVC_YAML.read_text(encoding="utf-8")
    start = text.find(MARKER_START)
    if start == -1:
        return None
    end = text.find(MARKER_END, start)
    if end == -1:
        return None
    end += len(MARKER_END)

    # Consume the consecutive export dependency lines that follow.
    tail = text[end:]
    consumed = 0
    for match in EXPORT_LINE.finditer(tail):
        if match.start() != consumed:
            break
        consumed = match.end() + 1  # +1 for the newline
    return text[start : end + consumed]


def _desired_block() -> str:
    exports = _export_dirs()
    if not exports:
        # On CI (or any environment without local ChatExport_* directories),
        # the desired block is whatever is already in dvc.yaml — we cannot
        # regenerate it without the raw exports, so we consider it in sync.
        return _current_block() or ""
    lines = "\n".join(f"      - {p.relative_to(REPO_ROOT).as_posix()}/result.json" for p in exports)
    return (
        MARKER_START
        + "      # explicitly. When a new export directory is added, refresh this\n"
        + "      # list (scripts/refresh_dvc_deps.py).\n"
        + lines
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the DVC raw-export dependency list")
    parser.add_argument("--check", action="store_true", help="CI mode: fail if the list is out of sync")
    args = parser.parse_args()

    current = _current_block()
    desired = _desired_block()

    if current == desired:
        print(f"dvc.yaml is in sync ({len(_export_dirs())} raw exports)")
        return 0

    if args.check:
        listed = EXPORT_LINE.findall(DVC_YAML.read_text(encoding="utf-8"))
        print("dvc.yaml is OUT OF SYNC with the raw exports.", file=sys.stderr)
        print(f"  listed in dvc.yaml : {len(listed)}", file=sys.stderr)
        print(f"  present on disk    : {len(_export_dirs())}", file=sys.stderr)
        print("  fix: python scripts/refresh_dvc_deps.py", file=sys.stderr)
        return 1

    text = DVC_YAML.read_text(encoding="utf-8")
    start = text.find(MARKER_START)
    end = text.find(MARKER_END, start) + len(MARKER_END)
    DVC_YAML.write_text(text[:start] + desired + text[end:], encoding="utf-8")
    print(f"dvc.yaml updated ({len(_export_dirs())} raw exports listed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
