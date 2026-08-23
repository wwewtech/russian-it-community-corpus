"""
Unit tests for CLI subcommands and argument dispatching.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cli import main


class TestCLICommands(unittest.TestCase):
    def test_cli_help(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "--help"]):
            with self.assertRaises(SystemExit) as exc:
                with redirect_stdout(f):
                    main()
            self.assertEqual(exc.exception.code, 0)
        output = f.getvalue()
        self.assertIn("Russian IT Community", output)
        self.assertIn("rag", output)
        self.assertIn("chat", output)

    def test_cli_validate(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "validate"]):
            with redirect_stdout(f):
                main()
        output = f.getvalue()
        self.assertIn("Validating datasets", output)

    def test_cli_benchmark(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "benchmark"]):
            with redirect_stdout(f):
                main()
        output = f.getvalue()
        self.assertIn("Benchmark saved", output)

    def test_cli_rag_search(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "rag", "docker nginx reverse proxy", "--top-k", "1"]):
            with redirect_stdout(f):
                main()
        output = f.getvalue()
        self.assertIn("Top 1 RAG Results", output)


if __name__ == "__main__":
    unittest.main()
