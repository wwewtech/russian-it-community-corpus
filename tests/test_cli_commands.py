"""
Unit tests for CLI subcommands and argument dispatching.

All tests run against a small synthetic dataset fixture so that the suite
stays fast, deterministic, and independent of locally generated Parquet/JSONL
artifacts (which are gitignored by design).
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cli import main


def build_synthetic_dataset(root: Path) -> Path:
    """Create a minimal but structurally valid dataset_output tree.

    Mirrors the schemas produced by src/exporter so that DatasetValidator
    exercises every branch (parquet integrity, JSONL validity, PII audit,
    SFT turn conformance) without touching real production artifacts.
    """
    data_dir = root / "dataset_output"
    parquet_dir = data_dir / "parquet"
    jsonl_dir = data_dir / "jsonl"
    parquet_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)

    messages = pd.DataFrame(
        {
            "msg_id": [1, 2],
            "chat_id": [1010, 1010],
            "chat_name": ["community_node_01", "community_node_01"],
            "timestamp": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
            "unixtime": [1767225600, 1767225900],
            "author_anon": ["Developer_00001", "Developer_00002"],
            "author_id_anon": ["user_00001", "user_00002"],
            "text_clean": [
                "How do I configure nginx as a reverse proxy?",
                "Use proxy_pass inside the location block.",
            ],
            "reply_to_id": [None, 1],
            "domain": ["devops_infra", "devops_infra"],
            "tags": [["nginx"], ["nginx"]],
            "sentiment_score": [0, 1],
            "token_count_approx": [8, 7],
            "is_question": [True, False],
            "thread_id": [1, 1],
        }
    )
    messages.to_parquet(parquet_dir / "full_clean_messages.parquet", index=False)
    pd.DataFrame({"dialogue_id": [1], "turns": [3]}).to_parquet(parquet_dir / "sft_dialogues.parquet", index=False)
    pd.DataFrame({"chunk_id": [1], "content": ["nginx reverse proxy guide"]}).to_parquet(
        parquet_dir / "rag_knowledge_base.parquet", index=False
    )

    dialogue = {
        "messages": [
            {"role": "system", "content": "You are a helpful IT assistant."},
            {"role": "user", "content": "How do I configure nginx as a reverse proxy?"},
            {"role": "assistant", "content": "Add a location block with proxy_pass pointing to your upstream."},
        ]
    }
    for name in [
        "sft_sharegpt_format.jsonl",
        "sft_alpaca_format.jsonl",
        "sft_openai_messages.jsonl",
    ]:
        (jsonl_dir / name).write_text(json.dumps(dialogue) + "\n", encoding="utf-8")

    (jsonl_dir / "rag_chunks_kb.jsonl").write_text(
        json.dumps({"chunk_id": 1, "content": "nginx reverse proxy guide"}) + "\n",
        encoding="utf-8",
    )
    (jsonl_dir / "dpo_preference_pairs.jsonl").write_text(
        json.dumps(
            {
                "prompt": "How do I configure nginx?",
                "chosen": "Use proxy_pass in the location block.",
                "rejected": "Just restart the server.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return data_dir


class TestCLICommands(unittest.TestCase):
    def test_cli_help(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "--help"]):
            with self.assertRaises(SystemExit) as exc, redirect_stdout(f):
                main()
            self.assertEqual(exc.exception.code, 0)
        output = f.getvalue()
        self.assertIn("Russian IT Community", output)
        self.assertIn("rag", output)
        self.assertIn("chat", output)

    def test_cli_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = build_synthetic_dataset(Path(tmp))
            f = io.StringIO()
            # Patch the name imported into cli's module namespace.
            with (
                patch("sys.argv", ["cli.py", "validate"]),
                patch("cli.OUTPUT_DIR", data_dir),
                redirect_stdout(f),
            ):
                main()
            output = f.getvalue()
            self.assertIn("Validating datasets", output)

            result = json.loads(output[output.index("{") :])
            self.assertTrue(result["overall_passed"])
            self.assertTrue(result["parquet_files"]["passed"])
            self.assertTrue(result["jsonl_files"]["passed"])
            self.assertEqual(result["pii_leakage_audit"]["total_leak_count"], 0)
            self.assertTrue(result["sft_turn_conformance"]["passed"])

    def test_cli_validate_detects_corrupt_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = build_synthetic_dataset(Path(tmp))
            corrupt = data_dir / "jsonl" / "rag_chunks_kb.jsonl"
            corrupt.write_text('{"chunk_id": 1, "content": "ok"}\nNOT VALID JSON\n', encoding="utf-8")

            f = io.StringIO()
            with (
                patch("sys.argv", ["cli.py", "validate"]),
                patch("cli.OUTPUT_DIR", data_dir),
                redirect_stdout(f),
            ):
                main()

            result = json.loads(f.getvalue()[f.getvalue().index("{") :])
            self.assertFalse(result["overall_passed"])
            self.assertFalse(result["jsonl_files"]["passed"])
            self.assertEqual(result["jsonl_files"]["details"]["rag_chunks_kb.jsonl"]["corrupt_lines"], 1)

    def test_cli_validate_flags_pii_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = build_synthetic_dataset(Path(tmp))
            leaky = data_dir / "jsonl" / "sft_openai_messages.jsonl"
            leaky.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "ping me at john.doe@example.com please"},
                            {"role": "assistant", "content": "Sure."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            f = io.StringIO()
            with (
                patch("sys.argv", ["cli.py", "validate"]),
                patch("cli.OUTPUT_DIR", data_dir),
                redirect_stdout(f),
            ):
                main()

            raw = f.getvalue()
            result = json.loads(raw[raw.index("{") :])
            self.assertFalse(result["pii_leakage_audit"]["passed"])
            self.assertGreaterEqual(result["pii_leakage_audit"]["leaks_found"]["unmasked_emails"], 1)

    def test_cli_benchmark(self):
        f = io.StringIO()
        with patch("sys.argv", ["cli.py", "benchmark"]), redirect_stdout(f):
            main()
        output = f.getvalue()
        self.assertIn("Benchmark saved", output)

    def test_cli_rag_search(self):
        f = io.StringIO()
        with (
            patch("sys.argv", ["cli.py", "rag", "docker nginx reverse proxy", "--top-k", "1"]),
            patch("pathlib.Path.exists", return_value=True),
            patch("src.rag.rag_pipeline.LocalRAGPipeline") as mock_rag_cls,
        ):
            mock_rag = mock_rag_cls.return_value
            mock_rag.search.return_value = [
                {"score": 0.95, "domain": "devops_infra", "content": "Sample docker nginx result"}
            ]
            with redirect_stdout(f):
                main()
        output = f.getvalue()
        self.assertIn("Top 1 RAG Results", output)
        self.assertIn("Sample docker nginx result", output)


if __name__ == "__main__":
    unittest.main()
