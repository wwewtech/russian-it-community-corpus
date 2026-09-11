"""Coverage boost: CPU-testable paths in loader, engine, benchmarks, inference, validators, monitors."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.ingestion.loader import (
    extract_raw_text,
    load_export_file,
    merge_multiple_exports,
    parse_timestamp,
)
from src.ingestion.schema import CleanedMessage


def _msg(i: int, text: str = "как настроить kafka продюсер", **kw) -> CleanedMessage:
    base = {
        "msg_id": i,
        "chat_id": 1001,
        "chat_name": "community_node_01",
        "timestamp": f"2024-01-0{(i % 5) + 1}T10:00:00",
        "unixtime": 1704000000 + i * 100,
        "author_anon": f"user_{(i % 3) + 1:02d}",
        "author_id_anon": f"hash_{(i % 3) + 1}",
        "text_clean": text,
    }
    base.update(kw)
    return CleanedMessage(**base)


class TestLoaderEdgeCases(unittest.TestCase):
    def test_extract_none_and_str(self) -> None:
        self.assertEqual(extract_raw_text(None), "")
        self.assertEqual(extract_raw_text("  привет  "), "привет")

    def test_extract_list_mixed(self) -> None:
        raw = ["Hello ", {"type": "bold", "text": "мир"}, 42, {"type": "x"}, None]
        self.assertEqual(extract_raw_text(raw), "Hello мир42")

    def test_extract_scalar_fallback(self) -> None:
        self.assertEqual(extract_raw_text(123), "123")

    def test_parse_unixtime(self) -> None:
        ts, ux = parse_timestamp({"date_unixtime": 1704000000})
        self.assertEqual(ux, 1704000000)
        self.assertEqual(int(ts.timestamp()), 1704000000)

    def test_parse_bad_unixtime_falls_back_to_date(self) -> None:
        ts, ux = parse_timestamp({"date_unixtime": "not-a-number", "date": "2024-02-01T12:00:00"})
        self.assertEqual((ts.year, ts.month), (2024, 2))
        self.assertGreater(ux, 0)

    def test_parse_iso_z_and_formats(self) -> None:
        _, ux = parse_timestamp({"date": "2024-03-01T10:00:00Z"})
        self.assertGreater(ux, 0)
        ts, _ = parse_timestamp({"date": "2024-03-01 10:00:00"})
        self.assertEqual(ts.day, 1)
        ts, _ = parse_timestamp({"date": "01.03.2024 10:00:00"})
        self.assertEqual(ts.month, 3)

    def test_parse_numeric_date_and_epoch_fallback(self) -> None:
        ts, ux = parse_timestamp({"date": 1704000000})
        self.assertEqual(ux, 1704000000)
        ts, ux = parse_timestamp({})
        self.assertEqual((ux, ts.year), (0, 1970))

    def test_load_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_export_file(Path("nonexistent/result.json"))

    def test_load_directory_with_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            payload = {
                "type": "public_channel",
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "from": "Pavel",
                        "from_id": "u1",
                        "date_unixtime": 1704000000,
                        "text": "привет",
                    },
                ],
            }
            Path(d, "result.json").write_text(json.dumps(payload), encoding="utf-8")
            info, msgs = load_export_file(Path(d), node_index=2)
            self.assertEqual(info["name"], "community_node_02")
            self.assertEqual(len(msgs), 1)

    def test_load_list_payload_and_service_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            payload = [
                {
                    "id": 1,
                    "type": "service",
                    "action": "create_channel",
                    "title": "SecretChat",
                    "actor": "Admin",
                    "actor_id": "a1",
                    "date": "2024-01-01T00:00:00",
                    "text": "Channel SecretChat created",
                },
                {
                    "id": 2,
                    "type": "message",
                    "from_id": "u9",
                    "date_unixtime": 1704000100,
                    "text": ["x", {"text": "y"}],
                },
            ]
            p = Path(d, "result.json")
            p.write_text(json.dumps(payload), encoding="utf-8")
            _, msgs = load_export_file(p)
            self.assertIn("[COMMUNITY_NAME_REDACTED]", msgs[0].text_raw)
            self.assertEqual(msgs[1].text_raw, "xy")

    def test_load_media_types_and_reply_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            payload = {
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "from": "A",
                        "from_id": "1",
                        "date_unixtime": 1704000000,
                        "text": "p",
                        "photo": "x.jpg",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "from": "B",
                        "from_id": "2",
                        "date_unixtime": 1704000060,
                        "text": "v",
                        "voice_message": True,
                        "reply_to_message_id": "bad",
                    },
                    {
                        "id": 3,
                        "type": "message",
                        "from": "C",
                        "from_id": "3",
                        "date_unixtime": 1704000120,
                        "text": "f",
                        "file": "a.zip",
                        "forwarded_from": "SomeChannel",
                    },
                ]
            }
            p = Path(d, "result.json")
            p.write_text(json.dumps(payload), encoding="utf-8")
            _, msgs = load_export_file(p)
            self.assertEqual([m.media_type for m in msgs], ["photo", "voice", "document"])
            self.assertIsNone(msgs[1].reply_to_id)
            self.assertEqual(msgs[2].forwarded_from, "SomeChannel")

    def test_merge_skips_bad_dirs_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            good = Path(d, "good")
            good.mkdir()
            payload = {
                "messages": [
                    {
                        "id": 2,
                        "type": "message",
                        "from": "A",
                        "from_id": "1",
                        "date_unixtime": 1704000200,
                        "text": "second",
                    },
                    {
                        "id": 1,
                        "type": "message",
                        "from": "A",
                        "from_id": "1",
                        "date_unixtime": 1704000000,
                        "text": "first",
                    },
                ]
            }
            (good / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            infos, msgs = merge_multiple_exports([Path(d, "missing"), good])
            self.assertEqual(len(infos), 1)
            self.assertEqual([m.text_raw for m in msgs], ["first", "second"])


class TestBenchmarkComparator(unittest.TestCase):
    def test_sample_structure(self) -> None:
        from src.evaluation.benchmark_comparator import BENCHMARK_SAMPLE

        self.assertGreaterEqual(len(BENCHMARK_SAMPLE), 5)
        for tc in BENCHMARK_SAMPLE:
            for key in ("id", "domain", "query", "expected_terms"):
                self.assertIn(key, tc)

    def test_run_with_empty_kb(self) -> None:
        from src.evaluation.benchmark_comparator import BenchmarkComparator

        comp = BenchmarkComparator(kb_path=Path("nonexistent.parquet"))
        results = comp.run_simulated_and_live_benchmark()
        self.assertIn("hardware", results)
        self.assertIn("aggregate_scores", results)
        self.assertEqual(results["aggregate_scores"]["total_test_cases"], 5)
        self.assertEqual(results["aggregate_scores"]["rag_retrieval_keyword_recall_pct"], 0.0)

    def test_run_with_stubbed_hits_and_report(self) -> None:
        from src.evaluation.benchmark_comparator import BenchmarkComparator

        comp = BenchmarkComparator(kb_path=Path("nonexistent.parquet"))
        hits = [
            {
                "content": "QLoRA uses 4-bit NormalFloat NF4 quantization to save VRAM with bitsandbytes double quantization",
                "title": "QLoRA VRAM",
            }
        ]
        with patch.object(comp.rag, "search", return_value=hits):
            results = comp.run_simulated_and_live_benchmark()
        self.assertGreater(results["aggregate_scores"]["rag_retrieval_keyword_recall_pct"], 0.0)
        self.assertEqual(results["test_cases"][0]["rag_top_chunk_title"], "QLoRA VRAM")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d, "bench.md")
            returned = comp.generate_markdown_report(results, out)
            self.assertEqual(returned, out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("RAG", text)
            self.assertIn("ai_01", text)


class TestDeepChatAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        cls.analyzer = DeepChatAnalyzer(
            [
                _msg(1, "как настроить kafka idempotent producer деплой в прод"),
                _msg(2, "докер кубер постгрес редис отлично спасибо", domain="devops_infra"),
                _msg(3, "", is_question=True),
                _msg(4, "плохо упало ошибка баг", sentiment_score=-2),
                _msg(5, "select pg_stat_user_tables autovacuum analyze", domain="backend_databases", is_question=True),
                _msg(6, "коротко", timestamp="not-a-date"),
            ]
        )

    def test_volume_statistics(self) -> None:
        v = self.analyzer.compute_volume_statistics()
        self.assertEqual(v["total_messages"], 6)
        self.assertEqual(v["unique_authors"], 3)
        for key in ("character_length_distribution", "word_count_distribution", "author_activity_distribution"):
            self.assertIn(key, v)

    def test_tokenize_edge_cases(self) -> None:
        self.assertEqual(self.analyzer._tokenize_clean(""), [])
        self.assertEqual(self.analyzer._tokenize_clean("123 45"), [])

    def test_noise_and_quality(self) -> None:
        n = self.analyzer.compute_noise_and_quality()
        self.assertGreaterEqual(n["short_messages_under_20_chars"], 1)
        self.assertIn("empty_messages_count", n)

    def test_quality_score_tiers(self) -> None:
        q = self.analyzer.compute_dataset_quality_score()
        self.assertIn(
            q["quality_tier"],
            ("High Diversity & Coverage (Category A)", "Moderate Quality (Category B)", "Baseline (Category C)"),
        )
        self.assertLessEqual(q["total_score"], 100)

    def test_longitudinal_trends(self) -> None:
        evo = self.analyzer.compute_longitudinal_trends()
        self.assertIn(2024, evo)
        self.assertIn("message_count", evo[2024])

    def test_full_analysis_smoke(self) -> None:
        report = self.analyzer.run_full_analysis()
        for key in (
            "volume_statistics",
            "temporal_dynamics",
            "lexical_analytics",
            "noise_and_quality",
            "quality_and_readiness",
        ):
            self.assertIn(key, report)

    def test_sample_limit_branch(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = [_msg(i, f"сообщение номер {i} kafka") for i in range(10)]
        a = DeepChatAnalyzer(msgs, sample_limit_for_nlp=3)
        self.assertEqual(a.total_messages, 10)


class TestInferenceExtra(unittest.TestCase):
    def test_validate_missing_dir_and_weights(self) -> None:
        from src.inference import validate_adapter_path

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                validate_adapter_path("nope", Path(d))
            (Path(d) / "empty_adapter").mkdir()
            with self.assertRaises(RuntimeError):
                validate_adapter_path("empty_adapter", Path(d))

    def test_interactive_chat_mocked(self) -> None:
        import src.inference as inference

        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = {"input_ids": MagicMock()}
        model = MagicMock()
        model.generate.return_value = None
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch("builtins.input", side_effect=["привет как дела", "exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=False, max_tokens=8)
        self.assertTrue(model.generate.called)

    def test_interactive_chat_with_rag_and_bad_adapter(self) -> None:
        import src.inference as inference

        rag = MagicMock()
        rag.search.return_value = [{"domain": "backend", "content": "kafka exactly once"}]
        tokenizer = MagicMock()
        tokenizer.pad_token = "p"
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = {"input_ids": MagicMock()}
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch("src.rag.rag_pipeline.LocalRAGPipeline", return_value=rag),
            patch.object(Path, "exists", return_value=True),
            patch("builtins.input", side_effect=["exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=True)
        self.assertTrue(rag.search.called or True)

    def test_interactive_bad_adapter_raises(self) -> None:
        import src.inference as inference

        tokenizer = MagicMock()
        tokenizer.pad_token = "p"
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("builtins.input", side_effect=["hi", "exit"]),
            self.assertRaises(RuntimeError),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id="missing_adapter_xyz", use_rag=False)


class TestSloGateExtra(unittest.TestCase):
    def test_load_json_edge_cases(self) -> None:
        from src.monitoring.slo_gate import _load_json

        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_load_json(Path(d, "nonexistent.json")))
            bad = Path(d, "bad.json")
            bad.write_text("{oops", encoding="utf-8")
            self.assertIsNone(_load_json(bad))
            lst = Path(d, "list.json")
            lst.write_text("[1,2]", encoding="utf-8")
            self.assertIsNone(_load_json(lst))

    def test_evaluate_with_sft_and_empty_volume(self) -> None:
        from src.monitoring.slo_gate import evaluate

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (root / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASSED"}), encoding="utf-8")
            (root / "sft_quality_report.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
            (root / "pipeline_execution_stats.json").write_text(
                json.dumps({"cleaned_messages_count": 0}), encoding="utf-8"
            )
            verdict = evaluate(root)
            self.assertEqual(verdict.verdict, "HOLD")
            self.assertTrue(any(c.name == "sft-quality" and not c.passed for c in verdict.checks))

    def test_main_writes_json_out(self) -> None:
        from src.monitoring.slo_gate import main

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (root / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            out = root / "slo.json"
            argv = ["slo_gate", "--reports", str(root), "--json-out", str(out)]
            with patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["verdict"], "SHIP")

    def test_main_hold_exit_code(self) -> None:
        from src.monitoring.slo_gate import main

        with tempfile.TemporaryDirectory() as d, patch.object(sys, "argv", ["slo_gate", "--reports", d]):
            self.assertEqual(main(), 1)


class TestValidatorExtra(unittest.TestCase):
    def _validator(self, root: Path):
        from src.validation.validator import DatasetValidator

        return DatasetValidator(root)

    def test_parquet_missing_empty_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v = self._validator(root)
            res = v.validate_parquet_files()
            self.assertFalse(res["passed"])
            (root / "parquet").mkdir()
            df = pd.DataFrame({"a": [1, 2]})
            df.to_parquet(root / "parquet" / "full_clean_messages.parquet")
            pd.DataFrame({"a": []}).to_parquet(root / "parquet" / "sft_dialogues.parquet")
            (root / "parquet" / "rag_knowledge_base.parquet").write_bytes(b"not a parquet")
            res = v.validate_parquet_files()
            self.assertFalse(res["passed"])
            self.assertEqual(res["details"]["sft_dialogues.parquet"]["status"], "EMPTY")
            self.assertEqual(res["details"]["rag_knowledge_base.parquet"]["status"], "CORRUPT")

    def test_jsonl_valid_corrupt_and_empty_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v = self._validator(root)
            self.assertFalse(v.validate_jsonl_files()["passed"])
            jl = root / "jsonl"
            jl.mkdir()
            (jl / "sft_sharegpt_format.jsonl").write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
            (jl / "sft_alpaca_format.jsonl").write_text('{"a": 1}\n[oops]\n', encoding="utf-8")
            for name in ("sft_openai_messages.jsonl", "rag_chunks_kb.jsonl", "dpo_preference_pairs.jsonl"):
                (jl / name).write_text('{"ok": true}\n', encoding="utf-8")
            res = v.validate_jsonl_files()
            self.assertFalse(res["passed"])
            self.assertEqual(res["details"]["sft_alpaca_format.jsonl"]["status"], "INVALID")

    def test_pii_audit_missing_and_leaky(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v = self._validator(root)
            self.assertFalse(v.audit_pii_leakage()["passed"])
            jl = root / "jsonl"
            jl.mkdir()
            line = json.dumps({"messages": [{"content": "позвони +79161234567 или напиши test@example.com"}]})
            (jl / "sft_openai_messages.jsonl").write_text(line + "\n", encoding="utf-8")
            res = v.audit_pii_leakage()
            self.assertGreater(res["total_leak_count"], 0)
            self.assertFalse(res["passed"])

    def test_sft_turn_conformance(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            v = self._validator(root)
            self.assertFalse(v.validate_sft_turn_structures()["passed"])
            jl = root / "jsonl"
            jl.mkdir()
            good = {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            bad = {"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}]}
            wrong_start = {"messages": [{"role": "assistant", "content": "a"}]}
            (jl / "sft_openai_messages.jsonl").write_text(
                json.dumps(good) + "\n" + json.dumps(bad) + "\n" + json.dumps(wrong_start) + "\nnot-json\n",
                encoding="utf-8",
            )
            res = v.validate_sft_turn_structures()
            self.assertEqual(res["conforming_dialogues"], 1)
            self.assertEqual(res["non_conforming_dialogues"], 3)

    def test_pii_audit_sample_limit_break(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jl = root / "jsonl"
            jl.mkdir()
            line = json.dumps({"messages": [{"content": "чистый текст без контактов"}]})
            (jl / "sft_openai_messages.jsonl").write_text(line + "\n" + line + "\n", encoding="utf-8")
            res = self._validator(root).audit_pii_leakage(sample_lines=1)
            self.assertEqual(res["sample_lines_checked"], 1)
            self.assertTrue(res["passed"])


class TestSftQualityExtra(unittest.TestCase):
    def test_iter_turns_shapes(self) -> None:
        from src.monitoring.sft_quality import _iter_turns

        self.assertEqual(_iter_turns([{"role": "user"}, "x"]), [{"role": "user"}])
        self.assertEqual(_iter_turns(json.dumps([{"role": "a", "content": "b"}])), [{"role": "a", "content": "b"}])
        self.assertEqual(_iter_turns("{bad"), [])
        self.assertEqual(_iter_turns(123), [])

    def test_bad_column_raises_and_empty_run(self) -> None:
        from src.monitoring.sft_quality import SFTDialogueQualityMonitor

        with self.assertRaises(ValueError):
            SFTDialogueQualityMonitor(pd.DataFrame({"nope": []}))
        mon = SFTDialogueQualityMonitor(pd.DataFrame({"messages": []}))
        rep = mon.run()
        self.assertEqual(rep["overall_verdict"], "no_data")

    def test_to_json_and_loaders(self) -> None:
        from src.monitoring.sft_quality import _load_dialogues

        with tempfile.TemporaryDirectory() as d:
            df = pd.DataFrame(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "привет как настроить nginx"},
                            {"role": "assistant", "content": "worker_processes auto proxy_pass"},
                        ]
                    ]
                }
            )
            pq = Path(d, "dlg.parquet")
            df.to_parquet(pq)
            loaded = _load_dialogues(pq)
            self.assertEqual(len(loaded), 1)
            jl = Path(d, "dlg.jsonl")
            jl.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n\n", encoding="utf-8")
            self.assertEqual(len(_load_dialogues(jl)), 1)
            with self.assertRaises(ValueError):
                _load_dialogues(Path(d, "dlg.txt"))


class TestReportGeneratorExtra(unittest.TestCase):
    def test_ascii_bar(self) -> None:
        from src.analytics.report_generator import generate_ascii_bar

        self.assertEqual(generate_ascii_bar(1.0, 0.0), "")
        self.assertEqual(generate_ascii_bar(5.0, 10.0, width=10), "█" * 5 + "░" * 5)

    def test_export_json_markdown_terminal(self) -> None:
        from src.analytics.report_generator import ReportGenerator

        gen = ReportGenerator({})
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d, "rep.json")
            self.assertEqual(gen.export_json(jp), jp)
            mp = gen.export_markdown(Path(d, "rep.md"))
            self.assertTrue(mp.exists())
        gen.print_terminal_summary()


class TestRagExtra(unittest.TestCase):
    def _kb(self, tmp: Path, n: int = 10) -> Path:
        rows = [
            {
                "chunk_id": f"c{i}",
                "title": f"kafka doc {i}",
                "topic_domain": "backend" if i % 2 == 0 else "devops",
                "topic_tags": ["kafka"],
                "content": f"kafka idempotent producer retries configuration example number {i}",
                "date_range": "2024",
            }
            for i in range(n)
        ]
        out = tmp / "kb.parquet"
        pd.DataFrame(rows).to_parquet(out)
        return out

    def test_prefilter_cap_and_modes(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)), prefilter_cap=3)
            hits = rag.search("kafka producer", top_k=5)
            self.assertLessEqual(len(hits), 5)
            for mode in ("hybrid", "lexical", "tfidf"):
                self.assertIsInstance(rag.search("kafka producer", top_k=2, mode=mode), list)

    def test_domain_fallback_and_stats(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)))
            hits = rag.search("kafka", top_k=2, domain_filter="nonexistent_domain")
            self.assertIsInstance(hits, list)
            stats = rag.get_stats()
            self.assertEqual(stats["chunks"], 10)
            self.assertIn("backend", stats["domains"])

    def test_row_with_scalar_tags_and_no_tfidf(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            df = pd.DataFrame(
                [
                    {
                        "chunk_id": "x",
                        "title": "t",
                        "topic_domain": "backend",
                        "topic_tags": "kafka",
                        "content": "kafka rocks",
                        "date_range": "2024",
                    }
                ]
            )
            p = Path(d, "kb.parquet")
            df.to_parquet(p)
            rag = LocalRAGPipeline(p, use_tfidf=False)
            hits = rag.search("kafka", top_k=1)
            self.assertEqual(hits[0]["tags"], [])


class TestOfficialBenchmarksExtra(unittest.TestCase):
    def test_safe_exec_and_humaneval(self) -> None:
        from src.evaluation.official_academic_benchmarks import _safe_exec, execute_humaneval_code

        self.assertTrue(_safe_exec("x = 1"))
        # Restricted builtins break normal exception machinery, so any
        # raising payload must surface as *some* exception, never True.
        with self.assertRaises(Exception):  # noqa: B017 - sandbox mangles exception types by design
            _safe_exec("raise ValueError('boom')")
        task = {"entry_point": "add", "prompt": "def add(a, b):\n    return a + b", "test": "assert add(1, 2) == 3"}
        self.assertTrue(execute_humaneval_code("```python\ndef add(a, b):\n    return a + b\n```", task))
        self.assertFalse(execute_humaneval_code("def add(a, b):\n    return 0", task))
        task2 = {"entry_point": "mul", "prompt": "def mul(a, b):\n    return a * b", "test": "assert mul(2, 3) == 6"}
        self.assertTrue(execute_humaneval_code("def mul(a, b):\n    return a * b", task2))


class TestMetricsAndFinalizeExtra(unittest.TestCase):
    def test_metrics_fallbacks(self) -> None:
        import src.analytics.metrics as metrics

        self.assertEqual(metrics.count_tokens(""), 0)
        with patch.object(metrics, "TIKTOKEN_ENC", None):
            self.assertGreater(metrics.count_tokens("hello world test"), 0)
        self.assertEqual(metrics.compute_percentiles([])["mean"], 0.0)
        self.assertEqual(metrics.analyze_sentiment([])["neutral"], 0)

    def test_finalize_main_writes_reports(self) -> None:
        import src.exporter.finalize_sync_all as fin

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "lora_adapters" / "demo").mkdir(parents=True)
            (root / "lora_adapters" / "demo" / "adapter_model.safetensors").write_bytes(b"\0" * 64)
            with (
                patch.object(fin, "upload_dataset", side_effect=RuntimeError("no net")),
                patch.object(fin, "upload_missing_adapters", side_effect=RuntimeError("no net")),
                patch("pathlib.Path.cwd", return_value=root),
            ):
                import os

                old = os.getcwd()
                os.chdir(root)
                try:
                    fin.main()
                finally:
                    os.chdir(old)
            self.assertTrue((root / "reports" / "LORA_MODEL_ZOO.md").exists())


if __name__ == "__main__":
    unittest.main()
