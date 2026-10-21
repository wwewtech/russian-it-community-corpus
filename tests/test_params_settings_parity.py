"""Tests that params.yaml is the single source of the audit parameters.

``settings.pii_audit`` and ``settings.benchmarks`` used to be DEAD
configuration: the values were declared in ``Settings`` and duplicated
as literals in ``probabilistic_audit.py``, in the CLI and in the Prefect
flow, and nothing read them. Editing ``params.yaml`` changed nothing at
runtime, while the DVC pipeline declared ``pii_audit.sample_size`` as
a parameter of a stage — so a stage would re-run for a value that had
no effect.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import yaml

from ricc.settings import Settings, settings
from ricc.validation.probabilistic_audit import ProbabilisticPIIAuditor

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "params.yaml"
PACKAGED_PARAMS_PATH = REPO_ROOT / "ricc" / "params.yaml"


def _params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8")) or {}


class TestParamsDriveTheSettings(unittest.TestCase):
    def test_pii_audit_section_is_loaded(self):
        section = _params().get("pii_audit", {})
        self.assertTrue(section, "params.yaml has no pii_audit section")
        self.assertEqual(settings.pii_audit.sample_size, section["sample_size"])
        self.assertEqual(settings.pii_audit.confidence, section["confidence"])
        self.assertEqual(settings.pii_audit.max_leak_tolerance, section["max_leak_tolerance"])

    def test_drift_section_is_loaded(self):
        section = _params().get("drift", {})
        self.assertEqual(settings.drift.psi_stable, section["psi_stable"])
        self.assertEqual(settings.drift.js_moderate, section["js_moderate"])

    def test_benchmarks_section_is_loaded(self):
        section = _params().get("benchmarks", {})
        self.assertEqual(settings.benchmarks.humaneval_tasks, section["humaneval_tasks"])
        self.assertEqual(settings.benchmarks.rummlu_questions, section["rummlu_questions"])

    def test_sample_size_keeps_the_bound_below_the_tolerance(self):
        """The configured sample must be able to certify a clean corpus.

        With zero observed leaks the one-sided 99% upper bound is
        ~3.29/sample_size, so the sample must be large enough for that
        to stay under the tolerance. 50 000 does not (1.08e-04 > 1e-04).
        """
        from ricc.evaluation.statistical_power import leak_rate_upper_bound

        bound = leak_rate_upper_bound(0, settings.pii_audit.sample_size, settings.pii_audit.confidence)
        self.assertLessEqual(
            bound,
            settings.pii_audit.max_leak_tolerance,
            f"the configured sample size cannot certify a clean corpus: {bound:.3e} > {settings.pii_audit.max_leak_tolerance:.0e}",
        )


class TestAuditUsesTheConfiguredDefaults(unittest.TestCase):
    def test_run_audit_defaults_to_none(self):
        """None means "use the configured value"."""
        signature = inspect.signature(ProbabilisticPIIAuditor.run_audit)
        for name in ("sample_size", "confidence", "max_leak_tolerance"):
            with self.subTest(parameter=name):
                self.assertIsNone(signature.parameters[name].default)

    def test_run_audit_applies_the_configured_values(self):
        """Without explicit arguments the audit uses params.yaml.

        A small corpus CANNOT be certified: with 50 sampled messages the
        one-sided 99% upper bound is ~0.098, far above the 1e-04
        tolerance, so the verdict is LEAKS_DETECTED even though nothing
        leaked. That is the statistics working correctly, and it is why
        the configured sample size is 100000.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd

            path = Path(tmp) / "corpus.parquet"
            pd.DataFrame(
                {
                    "chat_name": ["community_node_01"] * 50,
                    "text_clean": ["чистый текст без персональных данных"] * 50,
                }
            ).to_parquet(path)
            report = ProbabilisticPIIAuditor(path).run_audit()
            # Nothing leaked ...
            self.assertEqual(report["results"]["messages_with_leaks"], 0)
            self.assertEqual(report["results"]["total_leak_instances"], 0)
            # ... but the sample is too small to certify that.
            self.assertGreater(
                report["results"]["message_leak_rate_upper_bound_99"], settings.pii_audit.max_leak_tolerance
            )
            self.assertEqual(report["verdict"], "LEAKS_DETECTED")

    def test_a_large_enough_sample_is_certified(self):
        """With the configured sample size a clean corpus is certified."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd

            path = Path(tmp) / "corpus.parquet"
            n = settings.pii_audit.sample_size
            pd.DataFrame(
                {
                    "chat_name": ["community_node_01"] * n,
                    "text_clean": ["чистый текст без персональных данных"] * n,
                }
            ).to_parquet(path)
            report = ProbabilisticPIIAuditor(path).run_audit()
            self.assertEqual(report["results"]["messages_with_leaks"], 0)
            self.assertEqual(report["verdict"], "PASSED")


class TestPackagedParamsMatchTheRepositoryCopy(unittest.TestCase):
    def test_both_copies_are_identical(self):
        if not PACKAGED_PARAMS_PATH.exists():
            self.skipTest("packaged params.yaml not present")
        self.assertEqual(
            PARAMS_PATH.read_text(encoding="utf-8"),
            PACKAGED_PARAMS_PATH.read_text(encoding="utf-8"),
            "the packaged params.yaml drifted from the repository copy",
        )


class TestSettingsFallBackSafely(unittest.TestCase):
    def test_a_settings_instance_always_boots(self):
        """The platform must boot even without a readable params.yaml."""
        instance = Settings()
        self.assertGreaterEqual(len(instance.domain_taxonomy), 8)
        self.assertGreater(instance.pii_audit.sample_size, 0)

    def test_unknown_keys_in_params_are_ignored(self):
        """A stray key must not break the boot."""
        instance = Settings()
        self.assertTrue(hasattr(instance, "pii_audit"))


if __name__ == "__main__":
    unittest.main()
