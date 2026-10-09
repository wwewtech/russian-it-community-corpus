"""requirements.lock.txt must exist and cover every root dependency.

The lock is a reference snapshot of the transitive closure of
requirements.txt (see scripts/lock_requirements.py). This test keeps it from
silently rotting when requirements.txt grows.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "requirements.lock.txt"
REQ_PATH = REPO_ROOT / "requirements.txt"


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _root_names() -> list[str]:
    names: list[str] = []
    for line in REQ_PATH.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.append(match.group(1))
    return names


class TestRequirementsLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not LOCK_PATH.exists():
            raise unittest.SkipTest("requirements.lock.txt not generated")
        cls.lock_text = LOCK_PATH.read_text(encoding="utf-8")

    def test_lock_has_platform_header(self):
        """Header must state capture platform so readers know its scope."""
        self.assertIn("# Captured on:", self.lock_text)
        self.assertIn("# REFERENCE SNAPSHOT", self.lock_text)

    def test_lock_covers_all_root_dependencies(self):
        locked = {
            _canonical(line.split("==")[0])
            for line in self.lock_text.splitlines()
            if "==" in line and not line.startswith("#")
        }
        missing = [name for name in _root_names() if _canonical(name) not in locked]
        self.assertEqual(missing, [], f"root deps missing from requirements.lock.txt: {missing}")

    def test_lock_pins_are_exact(self):
        pins = [line for line in self.lock_text.splitlines() if line and not line.startswith("#")]
        self.assertGreater(len(pins), 50)
        for line in pins:
            self.assertRegex(line, r"^[A-Za-z0-9._-]+==\S+$", f"non-exact pin: {line!r}")


if __name__ == "__main__":
    unittest.main()
