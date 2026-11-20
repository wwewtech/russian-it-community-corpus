"""Library modules under ricc/ must not trigger runtime setup at import time.

Importing any module in ``ricc/`` must not mutate ``os.environ``, reconfigure
stdio, or attach root logging handlers. Process-level setup belongs inside
``main()`` entry points only.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RICC_DIR = REPO_ROOT / "ricc"

FORBIDDEN_TOP_LEVEL_CALLS = frozenset({"setup_runtime_env", "basicConfig", "configure_logging"})


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class TestNoImportSideEffects(unittest.TestCase):
    def test_no_ricc_module_runs_env_or_logging_setup_at_import_time(self) -> None:
        offenders: list[str] = []
        for path in sorted(RICC_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for stmt in tree.body:
                # Top-level statements other than def/class must not invoke
                # runtime env or logging setup.
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Call):
                        name = _call_name(sub)
                        if name in FORBIDDEN_TOP_LEVEL_CALLS:
                            rel = path.relative_to(REPO_ROOT).as_posix()
                            offenders.append(f"{rel}:{sub.lineno}: {name}()")
        self.assertEqual(
            offenders,
            [],
            f"Import-time runtime/logging setup found in library modules: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
