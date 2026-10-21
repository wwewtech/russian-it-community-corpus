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


class TestToolVersionPinsAgree(unittest.TestCase):
    """ruff and mypy must be pinned identically everywhere.

    The three places that pin the linter/type-checker drifted apart
    (pyproject said ruff 0.17.1 / mypy 2.4.0, CI and pre-commit used
    0.16.2 / 2.3.1), so ``make lint`` locally did not reproduce CI and
    a lint failure could appear in one place only. pyproject.toml is the
    source of truth; the other two must agree with it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        cls.ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        cls.precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    @staticmethod
    def _pyproject_pin(text: str, tool: str) -> str:
        match = re.search(rf'"{tool}==([0-9][^"]*)"', text)
        if not match:
            raise AssertionError(f"{tool} is not pinned in pyproject.toml")
        return match.group(1)

    @staticmethod
    def _ci_env(text: str, tool: str) -> str:
        match = re.search(rf"{tool.upper()}_VERSION:\s*\"([0-9][^\"]*)\"", text)
        if not match:
            raise AssertionError(f"{tool.upper()}_VERSION is not set in ci.yml")
        return match.group(1)

    def test_ruff_version_agrees_everywhere(self):
        pinned = self._pyproject_pin(self.pyproject, "ruff")
        self.assertEqual(self._ci_env(self.ci, "ruff"), pinned, "ruff version differs between pyproject and CI")
        self.assertIn(f"rev: v{pinned}", self.precommit, "ruff rev differs between pyproject and pre-commit")

    def test_mypy_version_agrees_everywhere(self):
        pinned = self._pyproject_pin(self.pyproject, "mypy")
        self.assertEqual(self._ci_env(self.ci, "mypy"), pinned, "mypy version differs between pyproject and CI")


if __name__ == "__main__":
    unittest.main()
