"""
Unit tests for CLI subcommands and argument dispatching.
"""

from unittest.mock import patch
import pytest
from cli import main


class TestCLICommands:
    def test_cli_help(self, capsys):
        with patch("sys.argv", ["cli.py", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Russian IT Community" in captured.out
        assert "rag" in captured.out
        assert "chat" in captured.out

    def test_cli_validate(self, capsys):
        with patch("sys.argv", ["cli.py", "validate"]):
            main()
        captured = capsys.readouterr()
        assert "Validating datasets" in captured.out

    def test_cli_benchmark(self, capsys):
        with patch("sys.argv", ["cli.py", "benchmark"]):
            main()
        captured = capsys.readouterr()
        assert "Benchmark saved" in captured.out

    def test_cli_rag_search(self, capsys):
        with patch("sys.argv", ["cli.py", "rag", "docker nginx reverse proxy", "--top-k", "1"]):
            main()
        captured = capsys.readouterr()
        assert "Top 1 RAG Results" in captured.out
