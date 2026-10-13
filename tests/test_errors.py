"""
Unit tests for :mod:`src.errors`.
"""

from __future__ import annotations

import unittest

from src.errors import (
    AdapterLoadError,
    CheckpointError,
    ConfigurationError,
    DriftDetectedError,
    Err,
    EvaluationError,
    ExternalServiceError,
    GPUDriverError,
    GPUNotAvailableError,
    GPUOutOfMemoryError,
    HuggingFaceHubError,
    InferenceError,
    IngestionError,
    MLError,
    Ok,
    PIIAnonymizationError,
    PIIValidationError,
    PipelineError,
    QuantizationError,
    RICCError,
    SchemaValidationError,
    TokenizerError,
    TrainingError,
    ValidationError,
    VRAMExhaustedError,
    classify_ml_exception,
    is_oom_error,
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


class TestGPUMLErrorTaxonomy(unittest.TestCase):
    def test_legacy_vram_alias_is_gpu_oom(self):
        err = VRAMExhaustedError("oom", requested_mb=16000, available_mb=12000)
        self.assertIsInstance(err, GPUOutOfMemoryError)
        self.assertEqual(err.component, "gpu")
        self.assertEqual(err.context.get("resource_type"), "vram")

    def test_ml_base_carries_model_context(self):
        err = MLError("boom", model_name="Qwen/Qwen2.5-0.5B", adapter_id="qwen05")
        self.assertEqual(err.component, "ml")
        self.assertEqual(err.context.get("model_name"), "Qwen/Qwen2.5-0.5B")
        self.assertEqual(err.context.get("adapter_id"), "qwen05")

    def test_training_error_step_and_phase(self):
        err = TrainingError("step failed", step=99, phase="trainer.train()")
        self.assertEqual(err.component, "training")
        self.assertEqual(err.context.get("step"), 99)
        self.assertEqual(err.context.get("phase"), "trainer.train()")

    def test_inference_and_evaluation_components(self):
        self.assertEqual(InferenceError("gen failed").component, "inference")
        err = EvaluationError("bench failed", benchmark="HumanEval", variant="lora")
        self.assertEqual(err.component, "evaluation")
        self.assertEqual(err.context.get("benchmark"), "HumanEval")
        self.assertEqual(err.context.get("variant"), "lora")

    def test_tokenizer_checkpoint_adapter_quantization_components(self):
        self.assertEqual(TokenizerError("tok failed").component, "tokenizer")
        err = CheckpointError("save failed", checkpoint_path="lora_adapters/x")
        self.assertEqual(err.component, "checkpoint")
        self.assertEqual(err.context.get("checkpoint_path"), "lora_adapters/x")
        err = AdapterLoadError("corrupt", adapter_path="lora_adapters/x", base_model="Qwen")
        self.assertEqual(err.component, "adapter")
        self.assertEqual(err.context.get("adapter_path"), "lora_adapters/x")
        err = QuantizationError("bnb failed", quant_type="nf4")
        self.assertEqual(err.component, "quantization")
        self.assertEqual(err.context.get("quant_type"), "nf4")

    def test_gpu_unavailable_and_driver(self):
        err = GPUNotAvailableError("no cuda", device="cuda:0")
        self.assertEqual(err.component, "gpu")
        self.assertEqual(err.context.get("reason"), "cuda_not_available")
        err = GPUDriverError("cublas fail", driver_version="550", cuda_version="12.4")
        self.assertEqual(err.context.get("driver_version"), "550")

    def test_is_oom_error_markers(self):
        self.assertTrue(is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2GB")))
        self.assertFalse(is_oom_error(ValueError("bad shape")))

        class FakeTorchOOM(RuntimeError):
            pass

        FakeTorchOOM.__name__ = "OutOfMemoryError"
        self.assertTrue(is_oom_error(FakeTorchOOM("boom")))

    def test_classify_ml_exception_oom(self):
        err = classify_ml_exception(
            RuntimeError("CUDA out of memory. Tried to allocate 1.5GB"),
            model_name="Qwen",
            device="cuda:0",
        )
        self.assertIsInstance(err, GPUOutOfMemoryError)
        self.assertEqual(err.context.get("model_name"), "Qwen")

    def test_classify_ml_exception_unavailable_and_driver(self):
        err = classify_ml_exception(RuntimeError("CUDA not available, cannot use device cuda:0"))
        self.assertIsInstance(err, GPUNotAvailableError)
        err = classify_ml_exception(RuntimeError("NCCL communicator failed to init"))
        self.assertIsInstance(err, GPUDriverError)

    def test_classify_ml_exception_fallback_training(self):
        err = classify_ml_exception(ValueError("dataloader shape mismatch"), adapter_id="qwen05")
        self.assertIsInstance(err, TrainingError)
        self.assertIn("original_error", err.context)


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
