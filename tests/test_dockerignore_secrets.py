"""Regression: local secret files must never enter the Docker build context.

Added during the 2026-09-16 audit (see reports/audit/01_baseline.md, finding P1):
.dockerignore did not list .env while the Dockerfile copies the whole build
context (Dockerfile: `COPY --chown=appuser:appuser . .`). The HF token lives in
.env per docs/adr/0001-hf-token-handling.md and must not be baked into images.

Verification strategy: when the `docker` package (docker-py) is installed, use
its `exclude_paths`, which implements the same patternmatcher semantics as
Docker Engine. Otherwise fall back to the official .dockerignore spec subset:
a pattern without a slash matches the file only at the context root, so ".env"
must keep the root .env out of the context while keeping the Dockerfile in it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from docker.utils.build import exclude_paths  # type: ignore[import-untyped]

    _DOCKER_PY = True
except Exception:
    _DOCKER_PY = False


def _load_dockerignore_patterns() -> list[str]:
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _kept_paths_spec_subset(root: Path, patterns: list[str]) -> set[str]:
    """Spec-subset matcher: slash-free patterns anchor at the context root.

    This intentionally models only the subset used by this repo's .dockerignore
    (exact names and root-anchored paths). Docker/moby patternmatcher is the
    source of truth; docker-py is preferred when installed.
    """
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    ignored: set[str] = set()
    for pattern in patterns:
        if pattern.startswith("!"):
            continue  # re-includes are prohibited for secrets and rejected below
        base = pattern.rstrip("/")
        if "/" in base:
            ignored.update(f for f in files if f == base or f.startswith(base + "/"))
        else:
            ignored.update(f for f in files if f.split("/")[-1] == base)
    return files - ignored


class TestDockerignoreSecrets(unittest.TestCase):
    def test_root_env_is_excluded_from_build_context(self) -> None:
        patterns = _load_dockerignore_patterns()
        if _DOCKER_PY:
            kept = set(exclude_paths(str(REPO_ROOT), patterns, dockerfile="Dockerfile"))
        else:
            kept = _kept_paths_spec_subset(REPO_ROOT, patterns)
        self.assertNotIn(".env", kept, ".env must be excluded from the Docker build context")
        self.assertIn("Dockerfile", kept, "sanity: the Dockerfile itself stays in the context")

    def test_no_reinclude_of_secret_patterns(self) -> None:
        for line in _load_dockerignore_patterns():
            self.assertFalse(
                line.startswith("!") and ".env" in line,
                f"re-include pattern for secrets is prohibited: {line!r}",
            )

    def test_env_pattern_present(self) -> None:
        patterns = _load_dockerignore_patterns()
        self.assertIn(".env", patterns)
        self.assertIn(".env.*", patterns)

    def test_env_is_git_ignored(self) -> None:
        """Belt and braces: .env must stay out of git as well (Docker context and git are separate)."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", "-v", ".env"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, ".env must be matched by .gitignore")
        self.assertIn(".gitignore", result.stdout)


if __name__ == "__main__":
    unittest.main()
