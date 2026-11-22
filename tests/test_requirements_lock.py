"""requirements.lock.txt must exist and cover every root dependency.

The lock is a reference snapshot of the transitive closure of
requirements.txt (see scripts/lock_requirements.py). This test keeps it from
silently rotting when requirements.txt grows.
"""

from __future__ import annotations

import re
import unittest

from scripts.lock_requirements import LOCK_PATH, REPO_ROOT
from scripts.lock_requirements import REQUIREMENTS_PATH as REQ_PATH


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

    def test_lock_is_portable_and_machine_readable(self):
        self.assertIn("# Portable lock for Python 3.11+", self.lock_text)
        self.assertFalse(self.lock_text.startswith("\ufeff"), "UTF-8 BOM breaks pip requirements parsing")

    def test_lock_covers_all_root_dependencies(self):
        locked = {
            _canonical(line.split("==")[0])
            for line in self.lock_text.splitlines()
            if "==" in line and not line.startswith("#")
        }
        missing = [name for name in _root_names() if _canonical(name) not in locked]
        self.assertEqual(missing, [], f"root deps missing from requirements.lock.txt: {missing}")

    def test_lock_pins_are_exact(self):
        from scripts.lock_requirements import _logical_requirement_lines

        pins = _logical_requirement_lines(self.lock_text)
        self.assertGreater(len(pins), 50)

        self.assertGreater(len(pins), 50)
        for line in pins:
            self.assertRegex(line, r"^[A-Za-z0-9._-]+==[^\s;]+(?:\s*;\s*.+)?$", f"non-exact pin: {line!r}")

    def test_metadata_and_direct_versions_are_consistent(self):
        from scripts.lock_requirements import validate_metadata

        self.assertEqual(validate_metadata(self.lock_text), [])

    def test_stale_direct_pin_is_rejected(self):
        from scripts.lock_requirements import validate_metadata

        stale = self.lock_text.replace("torch==2.15.0", "torch==2.5.0")
        self.assertTrue(any("violates declared" in error for error in validate_metadata(stale)))

    def test_pins_have_hashes_and_hash_required_installs(self):
        self.assertIn("--hash=sha256:", self.lock_text)
        for path in (REPO_ROOT / "Dockerfile", REPO_ROOT / ".github/workflows/ci.yml"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("pip install -r requirements.lock.txt"):
                    self.assertIn("--require-hashes", line)

    def test_resolution_is_cut_off_from_future_releases(self):
        from scripts.lock_requirements import COMPILE_ARGS, EXCLUDE_NEWER

        self.assertEqual(COMPILE_ARGS[COMPILE_ARGS.index("--exclude-newer") + 1], EXCLUDE_NEWER)
        self.assertRegex(EXCLUDE_NEWER, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_dockerfile_install_uses_pip_compatible_index_flags(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("--index-strategy", dockerfile, "--index-strategy is a uv flag, not a pip flag")
        self.assertIn("--index-url https://pypi.org/simple", dockerfile)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu", dockerfile)

    def test_release_gate_docker_commands_use_pypi_as_primary_index(self):
        import yaml

        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            if job_name not in {"package", "release", "eval-suites"}:
                continue
            for step in job.get("steps", []):
                for line in step.get("run", "").splitlines():
                    if "pip install" in line and "requirements.lock" in line:
                        self.assertNotIn(
                            "--index-url https://download.pytorch.org/whl/cpu",
                            line,
                            f"{job_name} must select pinned public packages from PyPI first",
                        )

    def test_all_runtime_jobs_install_the_lock(self):
        import yaml

        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                command = step.get("run", "")
                if "pip install -r requirements.txt" in command:
                    self.fail(f"{name}/{step.get('name')} bypasses the lock")


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
