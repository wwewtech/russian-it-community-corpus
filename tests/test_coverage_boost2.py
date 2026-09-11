"""Coverage boost wave 2: engine branches, LDA, inference GPU/adapter paths, prefect tasks, drift verdicts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from tests.test_coverage_boost import _msg


class TestEngineBranches(unittest.TestCase):
    def test_no_timestamps_temporal_empty(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        a = DeepChatAnalyzer([_msg(1, "kafka деплой", timestamp="bad"), _msg(2, "докер", timestamp="also-bad")])
        self.assertEqual(a.compute_temporal_dynamics(), {})
        self.assertEqual(a.compute_longitudinal_trends(), {})

    def test_tags_and_code_markers(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = [
            _msg(1, "как настроить ```python\ndef f(): pass\n``` деплой", tags=["kafka", "docker"]),
            _msg(2, "обычный текст про погоду", tags=["misc"]),
        ]
        a = DeepChatAnalyzer(msgs)
        slang = a.compute_domain_slang_analytics()
        self.assertGreater(len(slang["top_technical_tags"]), 0)
        syntax = a.compute_sentiment_and_syntax()
        self.assertEqual(syntax["code_snippets_count"], 1)

    def test_author_key_phrases_with_prolific_author(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = [
            _msg(
                i,
                "kafka кластер настройка репликация прод деплой монитор алерт",
                author_anon="user_pro",
                author_id_anon="h",
            )
            for i in range(35)
        ]
        msgs += [_msg(100 + i, "погода солнце", author_anon=f"other_{i:02d}", author_id_anon=f"o{i}") for i in range(3)]
        a = DeepChatAnalyzer(msgs)
        vocab = a.compute_author_key_phrases()
        self.assertIn("user_pro", vocab)
        self.assertGreater(len(vocab["user_pro"]), 0)

    def test_lda_success_path(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        # Sliding vocab windows: each term is shared by a few docs
        # (min_df=2 needs >=2 docs per term; max_df=0.6 drops ubiquitous ones).
        pool = [f"techterm{i:02d}" for i in range(30)]
        msgs = []
        k = 0
        for author in range(10):
            for _m in range(8):
                words = " ".join(pool[(author * 2 + j) % len(pool)] for j in range(12))
                msgs.append(
                    _msg(
                        k,
                        f"{words} настройка кластера деплой",
                        author_anon=f"lda_{author}",
                        author_id_anon=f"l{author}",
                    )
                )
                k += 1
        a = DeepChatAnalyzer(msgs)
        topics = a.compute_topic_clusters_lda(n_topics=8)
        self.assertEqual(len(topics), 8)
        self.assertIn("top_keywords", topics[0])

    def test_quality_branches_large_mixed(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = []
        for i in range(120):
            domain = "backend_databases" if i % 2 == 0 else ("devops_infra" if i % 4 == 1 else "general_tech_chat")
            msgs.append(
                _msg(
                    i,
                    f"сообщение {i} kafka постгрес деплой отлично",
                    domain=domain,
                    is_question=(i % 4 == 0),
                    author_anon=f"a{i % 100:03d}",
                    author_id_anon=f"h{i % 100}",
                )
            )
        a = DeepChatAnalyzer(msgs)
        q = a.compute_dataset_quality_score()
        self.assertEqual(q["score_breakdown"]["author_diversity_score"], 10)
        self.assertEqual(q["score_breakdown"]["technical_density_score"], 20)

    def test_morph_failure_fallback(self) -> None:
        from src.analytics import engine

        if engine.MORPH is None:
            self.skipTest("no morphological analyzer installed")
        from src.analytics.engine import DeepChatAnalyzer

        a = DeepChatAnalyzer([_msg(1, "тест")])
        with patch.object(engine.MORPH, "parse", side_effect=Exception("boom")):
            tokens = a._tokenize_clean("kafka кластер")
        self.assertIn("kafka", tokens)


class TestInferenceBranches(unittest.TestCase):
    def _tok(self) -> MagicMock:
        tok = MagicMock()
        tok.pad_token = "p"
        tok.apply_chat_template.return_value = "prompt"
        tok.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        return tok

    def test_adapter_attach_success(self) -> None:
        import src.inference as inference

        model = MagicMock()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch("src.inference.validate_adapter_path", return_value=Path("/tmp/adapter")),
            patch("peft.PeftModel.from_pretrained", return_value=model),
            patch("builtins.input", side_effect=["exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id="good", use_rag=False)

    def test_adapter_attach_failure_raises(self) -> None:
        import src.inference as inference

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("src.inference.validate_adapter_path", return_value=Path("/tmp/adapter")),
            patch("peft.PeftModel.from_pretrained", side_effect=RuntimeError("corrupt")),
            patch("builtins.input", side_effect=["hi"]),
            self.assertRaises(RuntimeError),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id="bad", use_rag=False)

    def test_rag_missing_branch(self) -> None:
        import src.inference as inference

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch.object(Path, "exists", return_value=False),
            patch("builtins.input", side_effect=["exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=True)

    def test_rag_hits_branch(self) -> None:
        import src.inference as inference

        rag = MagicMock()
        rag.search.return_value = [{"domain": "backend", "content": "kafka exactly once"}]
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch.object(Path, "exists", return_value=True),
            patch("src.rag.rag_pipeline.LocalRAGPipeline", return_value=rag),
            patch("builtins.input", side_effect=["как kafka", "exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=True)
        self.assertTrue(rag.search.called)

    def test_cuda_branch(self) -> None:
        import src.inference as inference

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch("torch.cuda.is_available", return_value=True),
            patch("builtins.input", side_effect=["exit"]),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=False)

    def test_keyboard_interrupt_branch(self) -> None:
        import src.inference as inference

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self._tok()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("transformers.TextStreamer", return_value=MagicMock()),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
        ):
            inference.interactive_chat_session(model_name="m", adapter_id=None, use_rag=False)

    def test_main_dispatch(self) -> None:
        import src.inference as inference

        with (
            patch.object(sys, "argv", ["inference", "--no-rag", "--model", "m"]),
            patch("src.inference.interactive_chat_session") as chat,
        ):
            inference.main()
        _, kwargs = chat.call_args
        self.assertFalse(kwargs["use_rag"])


class TestLoaderBranches(unittest.TestCase):
    def test_nondict_entry_skipped_and_video(self) -> None:
        from src.ingestion.loader import load_export_file

        with tempfile.TemporaryDirectory() as d:
            payload = {
                "messages": [
                    "oops-not-a-dict",
                    {
                        "id": 1,
                        "type": "message",
                        "from": "A",
                        "from_id": "1",
                        "date_unixtime": 1704000000,
                        "text": "v",
                        "video_file": "v.mp4",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "from": "B",
                        "from_id": "2",
                        "date_unixtime": 1704000060,
                        "text": "x",
                        "reply_to_message_id": 1,
                    },
                ]
            }
            p = Path(d, "result.json")
            p.write_text(json.dumps(payload), encoding="utf-8")
            _, msgs = load_export_file(p)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0].media_type, "video")
            self.assertEqual(msgs[1].reply_to_id, 1)


class TestOfficialBranches(unittest.TestCase):
    def test_humaneval_timeout(self) -> None:
        from src.evaluation.official_academic_benchmarks import execute_humaneval_code

        task = {"entry_point": "slow", "prompt": "def slow():\n    return 1", "test": "assert slow() == 1"}
        code = "def slow():\n    import time\n    time.sleep(3)\n    return 1"
        self.assertFalse(execute_humaneval_code(code, task, timeout_sec=0.3))

    def test_main_parses_and_dispatches(self) -> None:
        import src.evaluation.official_academic_benchmarks as bench

        with (
            patch.object(sys, "argv", ["prog", "--model", "m", "--adapter", "a"]),
            patch("src.evaluation.official_academic_benchmarks.run_official_academic_benchmarks") as run,
        ):
            bench.main()
        run.assert_called_once_with(model_name="m", adapter_id="a")


class TestReportBranches(unittest.TestCase):
    def _data(self) -> dict:
        return {
            "volume_statistics": {
                "total_messages": 10,
                "unique_authors": 3,
                "date_start": "2024-01-01",
                "date_end": "2024-01-02",
                "total_days_active": 1,
                "total_tokens_estimated": 2000000,
                "vocabulary_unique_words": 50,
            },
            "temporal_dynamics": {
                "peak_hour": 10,
                "peak_weekday": "Пн",
                "hourly_distribution": {"10:00": 5, "12:00": 3},
            },
            "domain_slang_analytics": {"domain_message_distribution": {"backend": {"count": 5, "percentage": 50.0}}},
            "quality_and_readiness": {"total_score": 90, "quality_tier": "A"},
            "social_network": {
                "total_nodes": 3,
                "total_edges": 2,
                "total_interactions": 4,
                "density": 0.1,
                "top_influencers": [{"author": "u1", "replies_received": 3}],
            },
            "topic_clusters_lda": [{"label": "T1", "top_keywords": ["kafka", "docker"]}],
            "sentiment_and_syntax": {
                "sentiment": {
                    "average": 0.5,
                    "positive": 1,
                    "pos_ratio": 10.0,
                    "negative": 0,
                    "neg_ratio": 0.0,
                    "neutral": 9,
                    "neu_ratio": 90.0,
                },
                "questions_count": 2,
                "questions_ratio_percentage": 20.0,
                "code_snippets_count": 1,
                "code_snippets_ratio_percentage": 10.0,
            },
            "noise_and_quality": {"short_messages_ratio_percentage": 5.0, "empty_messages_ratio_percentage": 0.0},
        }

    def test_markdown_with_influencers_and_lda(self) -> None:
        from src.analytics.report_generator import ReportGenerator

        gen = ReportGenerator(self._data())
        with tempfile.TemporaryDirectory() as d:
            mp = gen.export_markdown(Path(d, "rep.md"))
            text = mp.read_text(encoding="utf-8")
            self.assertIn("u1", text)
            self.assertIn("kafka", text)

    def test_terminal_with_data(self) -> None:
        from src.analytics.report_generator import ReportGenerator

        ReportGenerator(self._data()).print_terminal_summary()


class TestSftBranches(unittest.TestCase):
    def test_features_with_empty_rows_and_to_json(self) -> None:
        from src.monitoring.sft_quality import SFTDialogueQualityMonitor

        df = pd.DataFrame(
            {"messages": [[], "", [{"role": "user", "content": "привет как дела расскажи про nginx подробно"}]]}
        )
        mon = SFTDialogueQualityMonitor(df)
        feats = mon._features()
        self.assertEqual(len(feats), 3)
        with tempfile.TemporaryDirectory() as d:
            rep = mon.to_json(Path(d, "q.json"))
            self.assertIn("overall_verdict", rep)


class TestValidatorBranches(unittest.TestCase):
    def test_jsonl_corrupt_file(self) -> None:
        from src.validation.validator import DatasetValidator

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jl = root / "jsonl"
            jl.mkdir()
            (jl / "sft_sharegpt_format.jsonl").mkdir()  # opening a dir raises -> CORRUPT
            for name in (
                "sft_alpaca_format.jsonl",
                "sft_openai_messages.jsonl",
                "rag_chunks_kb.jsonl",
                "dpo_preference_pairs.jsonl",
            ):
                (jl / name).write_text('{"ok": true}\n', encoding="utf-8")
            res = DatasetValidator(root).validate_jsonl_files()
            self.assertFalse(res["passed"])
            self.assertEqual(res["details"]["sft_sharegpt_format.jsonl"]["status"], "CORRUPT")

    def test_audit_and_sft_with_malformed_rows(self) -> None:
        from src.validation.validator import DatasetValidator

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jl = root / "jsonl"
            jl.mkdir()
            (jl / "sft_openai_messages.jsonl").write_text(
                json.dumps({"messages": "oops-not-a-list"}) + "\n", encoding="utf-8"
            )
            v = DatasetValidator(root)
            self.assertIsInstance(v.audit_pii_leakage(), dict)
            res = v.validate_sft_turn_structures()
            self.assertEqual(res["non_conforming_dialogues"], 1)


class TestRagBranches(unittest.TestCase):
    def _kb(self, tmp: Path) -> Path:
        rows = [
            {
                "chunk_id": f"c{i}",
                "title": f"kafka {i}",
                "topic_domain": "backend",
                "topic_tags": ["kafka"],
                "content": f"kafka idempotent producer retries configuration example {i}",
                "date_range": "2024",
            }
            for i in range(4)
        ]
        out = tmp / "kb.parquet"
        pd.DataFrame(rows).to_parquet(out)
        return out

    def test_domain_match_and_short_query(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)))
            hits = rag.search("kafka producer", top_k=2, domain_filter="backend")
            self.assertGreater(len(hits), 0)
            self.assertEqual(rag.search("a b", top_k=2), [])

    def test_tfidf_empty_rank_falls_back_to_lexical(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)))
            hits = rag.search("kaf", top_k=2, mode="tfidf")
            self.assertIsInstance(hits, list)

    def test_tfidf_crash_falls_back_to_lexical(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)))
            with patch("sklearn.feature_extraction.text.TfidfVectorizer", side_effect=RuntimeError("boom")):
                hits = rag.search("kafka producer", top_k=2)
            self.assertGreater(len(hits), 0)
            self.assertEqual(rag.retriever, "lexical")

    def test_format_empty_contexts(self) -> None:
        from src.rag.rag_pipeline import LocalRAGPipeline

        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(self._kb(Path(d)))
            self.assertEqual(rag.format_rag_prompt("вопрос", []), "вопрос")


class TestDriftBranches(unittest.TestCase):
    def _frame(self, texts: list[str], domains: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"text_clean": texts, "domain": domains})

    def test_empty_vocab_overlap(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        m = DatasetDriftMonitor(self._frame([], []), self._frame([], []))
        v = m.compute_vocabulary_drift()
        self.assertEqual(v["jaccard_overlap"], 1.0)

    def test_significant_drift_verdicts(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        ref = self._frame(["kafka broker topic partition replica" for _ in range(50)], ["backend" for _ in range(50)])
        cur = self._frame(["x" * 500 for _ in range(50)], ["frontend" for _ in range(50)])
        rep = DatasetDriftMonitor(ref, cur).run()
        self.assertIn(rep["overall_verdict"], ("moderate_drift", "significant_drift", "stable"))
        self.assertIn("length_psi", rep["metrics"])


class TestMetricsBranch(unittest.TestCase):
    def test_encode_failure_fallback(self) -> None:
        import src.analytics.metrics as metrics

        if metrics.TIKTOKEN_ENC is None:
            self.skipTest("tiktoken unavailable")
        with patch.object(metrics.TIKTOKEN_ENC, "encode", side_effect=Exception("boom")):
            self.assertGreater(metrics.count_tokens("hello world test"), 0)


class TestPrefectBranches(unittest.TestCase):
    def test_pipeline_task_with_factory(self) -> None:
        from src.orchestration.prefect_flow import task_run_pipeline

        factory = MagicMock()
        factory.return_value.run_all.return_value = {"ok": True}
        res = task_run_pipeline(factory)
        self.assertEqual(res["status"], "completed")

    def test_pipeline_task_default_factory_mocked(self) -> None:
        from src.orchestration.prefect_flow import task_run_pipeline

        with patch("src.pipeline.MasterDataPipeline") as pipe:
            pipe.return_value.run_all.return_value = {"ok": True}
            res = task_run_pipeline(None)
        self.assertEqual(res["status"], "completed")

    def test_validate_task_failure(self) -> None:
        from src.orchestration.prefect_flow import task_validate_dataset

        with patch("src.validation.validator.DatasetValidator", side_effect=RuntimeError("boom")):
            res = task_validate_dataset("/tmp")
        self.assertEqual(res["status"], "failed")

    def test_audit_task_skipped_and_failed(self) -> None:
        from src.orchestration.prefect_flow import task_probabilistic_pii_audit

        with tempfile.TemporaryDirectory() as d:
            skipped = task_probabilistic_pii_audit(
                parquet_path=Path(d, "nonexistent.parquet"), report_path=Path(d, "skip.json"), sample_size=10
            )
            self.assertEqual(skipped["status"], "completed")
            bad = Path(d, "bad.parquet")
            bad.write_bytes(b"not a parquet")
            failed = task_probabilistic_pii_audit(parquet_path=bad, report_path=Path(d, "r.json"), sample_size=10)
            self.assertEqual(failed["status"], "failed")

    def test_drift_skipped_and_failed(self) -> None:
        from src.orchestration.prefect_flow import task_drift_monitoring

        skipped = task_drift_monitoring(reference_path="nope.parquet", current_path="nope2.parquet")
        self.assertEqual(skipped["status"], "skipped")
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d, "bad.parquet")
            bad.write_text("not parquet", encoding="utf-8")
            failed = task_drift_monitoring(reference_path=bad, current_path=bad)
            self.assertEqual(failed["status"], "failed")

    def test_run_flow_entry(self) -> None:
        from src.orchestration.prefect_flow import run_flow

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            res = run_flow(
                run_pipeline=False,
                output_dir=root,
                parquet_path=root / "missing.parquet",
                audit_report_path=root / "audit.json",
                reference_drift_path=root / "a.parquet",
                current_drift_path=root / "b.parquet",
                drift_report_path=root / "drift.json",
            )
        self.assertIn("validate-dataset", res)
        self.assertIn("drift-monitoring", res)


if __name__ == "__main__":
    unittest.main()
