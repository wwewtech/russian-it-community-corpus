"""
Unit tests for :mod:`src.errors`.
"""

from __future__ import annotations

import unittest

from src.errors import (
    RICCError,
    PipelineError,
    IngestionError,
    PIIAnonymizationError,
    ValidationError,
    SchemaValidationError,
    PIIValidationError,
    DataQualityError,
    DriftDetectedError,
    ConfigurationError,
    ExternalServiceError,
    HuggingFaceHubError,
    ResourceError,
    VRAMExhaustedError,
    Result,
    Ok,
    Err,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_base_error_creation(self):
        err = RICCError("test message", component="test", context={"key": "value"})
        self.assertEqual(err.message, "test message")
        self.assertEqual(err.component, "test")
        self.assertEqual(err.context, {"key": "value"})
        self.assertFalse(err.recoverable)

    def test_base_error_str(self):
        err = RICCError("msg", component="comp", context={"a": 1})
        s = str(err)
        self.assertIn("msg", s)
        self.assertIn("component=comp", s)
        self.assertIn("context={'a': 1}", s)

    def test_pipeline_error_stage(self):
        err = PipelineError("failed", stage="ingestion")
        self.assertEqual(err.stage, "ingestion")
        self.assertEqual(err.context.get("stage"), "ingestion")

    def test_ingestion_error(self):
        err = IngestionError("cannot load")
        self.assertEqual(err.stage, "ingestion")

    def test_pii_anonymization_error(self):
        err = PIIAnonymizationError("ner failed")
        self.assertEqual(err.stage, "pii_anonymization")

    def test_validation_error_check(self):
        err = ValidationError("invalid", check="schema")
        self.assertEqual(err.check, "schema")
        self.assertEqual(err.context.get("check"), "schema")

    def test_schema_validation_error(self):
        err = SchemaValidationError("bad schema")
        self.assertEqual(err.check, "schema")

    def test_pii_validation_error_with_leaks(self):
        err = PIIValidationError("leaks found", leaks={"emails": 2})
        self.assertEqual(err.check, "pii_audit")
        self.assertEqual(err.context.get("leaks"), {"emails": 2})

    def test_drift_detected_error(self):
        err = DriftDetectedError(
            "drift",
            psi_value=0.3,
            js_value=0.1,
            jaccard_value=0.7,
            verdict="significant_drift",
        )
        self.assertEqual(err.context.get("psi_value"), 0.3)
        self.assertEqual(err.context.get("verdict"), "significant_drift")

    def test_configuration_error(self):
        err = ConfigurationError("missing key", config_key="HF_TOKEN")
        self.assertEqual(err.context.get("config_key"), "HF_TOKEN")

    def test_external_service_error(self):
        err = ExternalServiceError("timeout", service="hf", status_code=504)
        self.assertTrue(err.recoverable)
        self.assertEqual(err.context.get("service"), "hf")
        self.assertEqual(err.context.get("status_code"), 504)

    def test_hf_hub_error(self):
        err = HuggingFaceHubError("repo not found")
        self.assertEqual(err.context.get("service"), "huggingface_hub")

    def test_vram_exhausted_error(self):
        err = VRAMExhaustedError("oom", requested_mb=16000, available_mb=12000)
        self.assertEqual(err.context.get("resource_type"), "vram")
        self.assertEqual(err.context.get("current_usage"), 16000)
        self.assertEqual(err.context.get("limit"), 12000)


class TestResultType(unittest.TestCase):
    def test_ok_creation(self):
        r = Ok("success")
        self.assertTrue(r.is_ok)
        self.assertFalse(r.is_err)
        self.assertEqual(r.value, "success")
        self.assertIsNone(r.error)

    def test_err_creation(self):
        r = Err(ValueError("bad"))
        self.assertFalse(r.is_ok)
        self.assertTrue(r.is_err)
        self.assertIsInstance(r.error, ValueError)

    def test_unwrap_ok(self):
        r = Ok(42)
        self.assertEqual(r.unwrap(), 42)

    def test_unwrap_err_raises(self):
        r = Err(RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            r.unwrap()

    def test_unwrap_or(self):
        self.assertEqual(Ok(1).unwrap_or(0), 1)
        self.assertEqual(Err(ValueError("x")).unwrap_or(0), 0)

    def test_map_ok(self):
        r = Ok(2).map(lambda x: x * 3)
        self.assertTrue(r.is_ok)
        self.assertEqual(r.value, 6)

    def test_map_err(self):
        r = Err(ValueError("x")).map_err(lambda e: RuntimeError("wrapped"))
        self.assertTrue(r.is_err)
        self.assertIsInstance(r.error, RuntimeError)

    def test_map_err_on_ok_passes_through(self):
        r = Ok(5).map_err(lambda e: RuntimeError("wrapped"))
        self.assertTrue(r.is_ok)
        self.assertEqual(r.value, 5)


if __name__ == "__main__":
    unittest.main()