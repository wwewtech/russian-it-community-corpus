"""
Command-Line Interface (CLI) for Russian IT Community Data Engineering Pipeline.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.analytics.engine import DeepChatAnalyzer
from src.analytics.report_generator import ReportGenerator
from src.config import OUTPUT_DIR, PARQUET_OUTPUT_DIR, RAW_EXPORT_DIRS, REPORTS_DIR
from src.ingestion.loader import merge_multiple_exports
from src.pipeline import MasterDataPipeline
from src.validation.benchmark import BenchmarkRunner
from src.validation.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Russian IT Community Data Engineering & Curation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run (all)
    parser_run = subparsers.add_parser("run", help="Run the full end-to-end data pipeline")
    parser_run.add_argument("--skip-ner", action="store_true", help="Skip Natasha NER layer (fast regex-only)")

    # Command: analyze
    parser_analyze = subparsers.add_parser("analyze", help="Run analytical report on exports")
    parser_analyze.add_argument("--limit", type=int, default=100000, help="Sample limit for NLP lemmatization")

    # Command: validate
    subparsers.add_parser("validate", help="Run dataset validation and zero-PII audit")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Export and display benchmark questions")

    # Command: audit-prob
    parser_audit = subparsers.add_parser(
        "audit-prob", help="Run stratified probabilistic PII audit with confidence bounds"
    )
    parser_audit.add_argument(
        "--parquet",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Path to production messages parquet",
    )
    parser_audit.add_argument("--sample-size", type=int, default=50000, help="Sample size to audit")
    parser_audit.add_argument("--confidence", type=float, default=0.99, help="Confidence level for bounds")
    parser_audit.add_argument("--tolerance", type=float, default=1e-4, help="Max acceptable leak rate tolerance")
    parser_audit.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "probabilistic_pii_audit.json",
        help="Output report JSON path",
    )

    # Command: drift
    parser_drift = subparsers.add_parser("drift", help="Run dataset drift monitoring (PSI / JS / vocabulary)")
    parser_drift.add_argument(
        "--reference",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Reference snapshot parquet",
    )
    parser_drift.add_argument(
        "--current",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Current snapshot parquet",
    )
    parser_drift.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "drift_report.json",
        help="Output report JSON path",
    )

    # Command: orchestrate
    parser_orch = subparsers.add_parser("orchestrate", help="Run Prefect curation orchestration flow")
    parser_orch.add_argument("--skip-pipeline", action="store_true", help="Skip curation stage and run downstream")
    parser_orch.add_argument("--sample-size", type=int, default=50000, help="Audit sample size")

    # Command: chat
    parser_chat = subparsers.add_parser("chat", help="Start interactive LLM + RAG terminal chat session")
    parser_chat.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser_chat.add_argument("--adapter", type=str, default="heavyweight_qwen2.5_coder_7b", help="LoRA Adapter ID")
    parser_chat.add_argument("--no-rag", action="store_true", help="Disable RAG knowledge retrieval")

    # Command: rag
    parser_rag = subparsers.add_parser("rag", help="Search the 325k RAG knowledge base from the CLI")
    parser_rag.add_argument("query", type=str, help="Technical search query")
    parser_rag.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")

    args = parser.parse_args()

    if args.command in ("run", None):
        print("🚀 Starting full data curation pipeline...")
        pipeline = MasterDataPipeline()
        if hasattr(args, "skip_ner") and args.skip_ner:
            pipeline.anonymizer.ner_scrubber = None
        pipeline.run_all()

    elif args.command == "analyze":
        print("🔬 Loading messages for analytical profiling...")
        clean_parquet_path = PARQUET_OUTPUT_DIR / "full_clean_messages.parquet"
        from src.ingestion.schema import CleanedMessage

        if clean_parquet_path.exists():
            import pandas as pd

            print(f"📂 Loading pre-cleaned dataset from {clean_parquet_path}...")
            df = pd.read_parquet(clean_parquet_path)
            cleaned = []
            for row in df.itertuples(index=False):
                cleaned.append(
                    CleanedMessage(
                        msg_id=row.msg_id,
                        chat_id=row.chat_id,
                        chat_name=row.chat_name,
                        timestamp=str(row.timestamp),
                        unixtime=int(row.unixtime),
                        author_anon=str(row.author_anon),
                        author_id_anon=str(row.author_id_anon),
                        text_clean=str(row.text_clean) if row.text_clean is not None else "",
                        reply_to_id=int(row.reply_to_id) if pd.notna(row.reply_to_id) else None,
                        domain=str(row.domain) if pd.notna(row.domain) else "general_tech_chat",
                        tags=list(row.tags) if isinstance(row.tags, (list, tuple)) else [],
                        sentiment_score=int(row.sentiment_score) if pd.notna(row.sentiment_score) else 0,
                        token_count_approx=int(row.token_count_approx) if pd.notna(row.token_count_approx) else 0,
                        is_question=bool(row.is_question) if pd.notna(row.is_question) else False,
                        thread_id=int(row.thread_id) if pd.notna(row.thread_id) else None,
                    )
                )
            print(f"✅ Loaded {len(cleaned):,} cleaned and tagged messages from Parquet.")
        else:
            chats_info, msgs = merge_multiple_exports(RAW_EXPORT_DIRS)
            cleaned = [
                CleanedMessage(
                    msg_id=m.msg_id,
                    chat_id=m.chat_id,
                    chat_name=m.chat_name,
                    timestamp=m.timestamp.isoformat(),
                    unixtime=m.unixtime,
                    author_anon=m.author_raw,
                    author_id_anon=m.author_id_raw,
                    text_clean=m.text_raw,
                    reply_to_id=m.reply_to_id,
                )
                for m in msgs
            ]

        analyzer = DeepChatAnalyzer(cleaned, sample_limit_for_nlp=args.limit)
        report = analyzer.run_full_analysis()
        gen = ReportGenerator(report)
        gen.print_terminal_summary()
        gen.export_markdown(REPORTS_DIR / "DEEP_ANALYTICAL_REPORT.md")
        gen.export_json(REPORTS_DIR / "analytics_summary.json")

    elif args.command == "validate":
        print("🔍 Validating datasets in output directory...")
        validator = DatasetValidator(OUTPUT_DIR)
        res = validator.validate_all()
        import json

        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.command == "benchmark":
        print("🎯 Generating domain evaluation benchmark...")
        bench = BenchmarkRunner()
        out = bench.export_benchmark_file(REPORTS_DIR / "domain_benchmark_100.json")
        print(f"✅ Benchmark saved to {out} (Total questions: {len(bench.questions)})")

    elif args.command == "audit-prob":
        from src.validation.probabilistic_audit import ProbabilisticPIIAuditor

        print("🛡️ Running stratified probabilistic PII audit...")
        auditor = ProbabilisticPIIAuditor(args.parquet)
        report = auditor.run_audit(
            sample_size=args.sample_size,
            confidence=args.confidence,
            max_leak_tolerance=args.tolerance,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Verdict: {report.get('verdict', report.get('status'))}")
        print(f"Report saved to {out}")

    elif args.command == "drift":
        import pandas as pd

        from src.monitoring.drift import DatasetDriftMonitor

        print("📊 Running dataset drift monitoring...")
        if not args.reference.exists() or not args.current.exists():
            print(f"Snapshots missing: reference={args.reference.exists()} current={args.current.exists()}")
            sys.exit(0)
        monitor = DatasetDriftMonitor(pd.read_parquet(args.reference), pd.read_parquet(args.current))
        report = monitor.run()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Overall drift verdict: {report['overall_verdict']}")
        print(f"Report saved to {out}")

    elif args.command == "orchestrate":
        import json

        from src.orchestration.prefect_flow import run_flow

        print("🔄 Executing Prefect orchestration flow...")
        results = run_flow(run_pipeline=not args.skip_pipeline, audit_sample_size=args.sample_size)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    elif args.command == "chat":
        from src.inference import interactive_chat_session

        interactive_chat_session(
            model_name=args.model,
            adapter_id=args.adapter,
            use_rag=not args.no_rag,
        )

    elif args.command == "rag":
        from src.rag.rag_pipeline import LocalRAGPipeline

        kb_path = Path("dataset_output/parquet/rag_knowledge_base.parquet")
        if not kb_path.exists():
            print("❌ RAG knowledge base not found at dataset_output/parquet/rag_knowledge_base.parquet")
            sys.exit(1)
        rag = LocalRAGPipeline(kb_path)
        hits = rag.search(args.query, top_k=args.top_k)
        print(f"\n🔍 Top {len(hits)} RAG Results for: '{args.query}'\n" + "=" * 60)
        for i, hit in enumerate(hits, 1):
            print(f"[{i}] Score: {hit['score']:.3f} | Domain: {hit.get('domain', 'general')}")
            print(f"    {hit['content']}\n" + "-" * 60)


if __name__ == "__main__":
    main()
