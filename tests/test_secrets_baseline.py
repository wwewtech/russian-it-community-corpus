"""Security baseline guard: .secrets.baseline must stay machine-readable and honest.

- Baseline must be UTF-8 without BOM (PowerShell Out-File -Encoding utf8
  writes a BOM that detect-secrets cannot parse — verified 2026-10-11).
- Baseline results must stay empty: sha256 provenance hashes, safetensors
  digests, model names and test fixtures are allowlisted/excluded, so any
  recorded finding means a real new secret or a broken pragma.
- Documented allowlist pragmas must stay in place (pinned revision, DSN
  example comment, test fixtures).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / ".secrets.baseline"

# Lines that must carry `# pragma: allowlist secret` — each was triaged as a
# benign identifier (pinned public revision, DSN example, test fixtures).
REQUIRED_PRAGMAS: dict[str, list[str]] = {
    "src/validation/hub_reconciliation.py": ['PINNED_HUB_REVISION = "81a3495de997b0690f67d175c4ebd9cfdef35b76"'],
    "src/pii/deep_anonymizer.py": ["postgresql://user:pass@host:5432/db"],
    "tests/test_probabilistic_pii_audit.py": [
        "eyJhbGciOiJIUzI1NiJ9",
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
}


class TestSecretsBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BASELINE_PATH.exists():
            raise unittest.SkipTest(".secrets.baseline missing")
        cls.raw = BASELINE_PATH.read_bytes()

    def test_baseline_is_utf8_without_bom(self):
        """BOM breaks detect-secrets baseline parsing (verified 2026-10-11)."""
        self.assertFalse(
            self.raw.startswith(b"\xff\xfe") or self.raw.startswith(b"\xef\xbb\xbf"),
            ".secrets.baseline has a BOM — detect-secrets cannot read it",
        )
        try:
            self.baseline = json.loads(self.raw.decode("utf-8"))
        except Exception as e:
            self.fail(f".secrets.baseline is not UTF-8 JSON: {e}")

    def test_baseline_results_stay_empty(self):
        """Any recorded finding = real new secret or broken allowlist."""
        baseline = json.loads(self.raw.decode("utf-8"))
        results = baseline.get("results", {})
        total = sum(len(v) for v in results.values())
        self.assertEqual(total, 0, f".secrets.baseline records findings: {sorted(results.keys())}")

    def test_allowlist_pragmas_in_place(self):
        """Triaged benign identifiers must keep their pragma comments."""
        for rel, markers in REQUIRED_PRAGMAS.items():
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            for marker in markers:
                with self.subTest(file=rel, marker=marker[:30]):
                    self.assertIn(marker, text, f"triaged line changed in {rel}")
                    line = next((ln for ln in text.splitlines() if marker in ln), "")
                    self.assertIn(
                        "pragma: allowlist secret",
                        line,
                        f"allowlist pragma missing in {rel}: {line.strip()[:100]}",
                    )


if __name__ == "__main__":
    unittest.main()
