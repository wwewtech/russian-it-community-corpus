import unittest

from src.evaluation.official_academic_benchmarks import execute_humaneval_code


class TestIsolatedHumanEvalExecution(unittest.TestCase):
    def test_successful_execution(self) -> None:
        task = {
            "entry_point": "add",
            "prompt": "def add(a, b):\n    return a + b\n",
            "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        }
        code = "def add(a, b):\n    return a + b"
        self.assertTrue(execute_humaneval_code(code, task, timeout_sec=2.0))

    def test_assertion_failure(self) -> None:
        task = {
            "entry_point": "add",
            "prompt": "def add(a, b):\n    return a + b\n",
            "test": "assert add(2, 3) == 6",
        }
        code = "def add(a, b):\n    return a + b"
        self.assertFalse(execute_humaneval_code(code, task, timeout_sec=2.0))

    def test_infinite_loop_killed_by_timeout(self) -> None:
        task = {
            "entry_point": "infinite_loop",
            "prompt": "def infinite_loop():\n    pass\n",
            "test": "assert infinite_loop() == 1",
        }
        code = "def infinite_loop():\n    while True:\n        pass\n    return 1"
        self.assertFalse(execute_humaneval_code(code, task, timeout_sec=0.5))

    def test_unsafe_builtins_rejected(self) -> None:
        task = {
            "entry_point": "danger",
            "prompt": "def danger():\n    pass\n",
            "test": "assert danger() == 1",
        }
        # Attempt to open file or use os
        code = "def danger():\n    open('secret.txt', 'w')\n    return 1"
        self.assertFalse(execute_humaneval_code(code, task, timeout_sec=1.0))
