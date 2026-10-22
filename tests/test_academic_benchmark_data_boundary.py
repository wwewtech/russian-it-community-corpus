"""Tests for the academic benchmark data/execution module boundary."""

import unittest

from ricc.evaluation import academic_benchmark_data
from ricc.evaluation import official_academic_benchmarks as benchmark


class TestAcademicBenchmarkDataBoundary(unittest.TestCase):
    def test_execution_module_reexports_static_benchmark_data(self) -> None:
        self.assertIs(benchmark.HUMANEVAL_TASKS, academic_benchmark_data.HUMANEVAL_TASKS)
        self.assertIs(benchmark.RUMMLU_CS_QUESTIONS, academic_benchmark_data.RUMMLU_CS_QUESTIONS)

    def test_static_benchmark_data_shape_is_unchanged(self) -> None:
        self.assertEqual(len(academic_benchmark_data.HUMANEVAL_TASKS), 40)
        self.assertEqual(len(academic_benchmark_data.RUMMLU_CS_QUESTIONS), 50)
        self.assertEqual(academic_benchmark_data.HUMANEVAL_TASKS[0]["task_id"], "HumanEval/0")
        self.assertEqual(academic_benchmark_data.RUMMLU_CS_QUESTIONS[0]["id"], "rummlu_cs_01")


if __name__ == "__main__":
    unittest.main()
