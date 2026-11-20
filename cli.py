"""
Command-Line Interface (CLI) for Russian IT Community Data Engineering Pipeline.

Heavy, command-specific dependencies (torch, transformers, pandas, natasha,
prefect, pymorphy3, sklearn, pyarrow) are imported lazily inside each
subcommand handler so that ``it-pipeline --help`` and lightweight subcommands
start instantly without dragging the full ML stack through import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ricc.bootstrap import setup_runtime_env
from ricc.settings import settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Russian IT Community Data Engineering & Curation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    parser_run = subparsers.add_parser("run", help="Run the full end-to-end data pipeline")
    parser_run.add_argument("--skip-ner", action="store_true", help="Skip Natasha NER layer (fast regex-only)")

    parser_analyze = subparsers.add_parser("analyze", help="Run analytical report on exports")
    parser_analyze.add_argument("--limit", type=int, default=100000, help="Sample limit for NLP lemmatization")

    subparsers.add_parser("validate", help="Run dataset validation and zero-PII audit")
    subparsers.add_parser("benchmark", help="Export and display benchmark questions")

    parser_audit = subparsers.add_parser(
        "audit-prob", help="Run stratified probabilistic PII audit with confidence bounds"
    )
    parser_audit.add_argument(
        "--parquet",
        type=Path,
        default=settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Path to production messages parquet",
    )
    parser_audit.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Sample size to audit. The default comes from params.yaml "
            "(pii_audit.sample_size = 100000), which keeps the 99%% upper bound "
            "(5.41e-05) below the 1e-04 tolerance even with zero observed leaks."
        ),
    )
    parser_audit.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence level for bounds (default: params.yaml pii_audit.confidence)",
    )
    parser_audit.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Max acceptable leak rate tolerance (default: params.yaml pii_audit.max_leak_tolerance)",
    )
    parser_audit.add_argument(
        "--output",
        type=Path,
        default=settings.REPORTS_DIR / "probabilistic_pii_audit.json",
        help="Output report JSON path",
    )

    parser_drift = subparsers.add_parser("drift", help="Run dataset drift monitoring (PSI / JS / vocabulary)")
    parser_drift.add_argument(
        "--reference",
        type=Path,
        default=settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Reference snapshot parquet",
    )
    parser_drift.add_argument(
        "--current",
        type=Path,
        default=settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Current snapshot parquet",
    )
    parser_drift.add_argument(
        "--output",
        type=Path,
        default=settings.REPORTS_DIR / "drift_report.json",
        help="Output report JSON path",
    )

    parser_orch = subparsers.add_parser("orchestrate", help="Run Prefect curation orchestration flow")
    parser_orch.add_argument("--skip-pipeline", action="store_true", help="Skip curation stage and run downstream")
    parser_orch.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Audit sample size (default: params.yaml pii_audit.sample_size)",
    )

    parser_chat = subparsers.add_parser("chat", help="Start interactive LLM + RAG terminal chat session")
    parser_chat.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser_chat.add_argument("--adapter", type=str, default="heavyweight_qwen2.5_coder_7b", help="LoRA Adapter ID")
    parser_chat.add_argument("--no-rag", action="store_true", help="Disable RAG knowledge retrieval")

    parser_rag = subparsers.add_parser("rag", help="Search the 325k RAG knowledge base from the CLI")
    parser_rag.add_argument("query", type=str, help="Technical search query")
    parser_rag.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")

    return parser


def _cmd_run(args: argparse.Namespace) -> None:
    from ricc.errors import PipelineError
    from ricc.pipeline import MasterDataPipeline

    print("Starting full data curation pipeline...")
    pipeline = MasterDataPipeline()
    if getattr(args, "skip_ner", False):
        pipeline.anonymizer.ner_scrubber = None
    try:
        summary = pipeline.run_all()
    except PipelineError as exc:
        print(f"Pipeline failed at stage '{exc.stage}': {exc}", file=sys.stderr)
        sys.exit(1)
    if not summary.get("validation_passed", False):
        print(
            "Dataset validation FAILED after the pipeline run — see reports/validation_results.json",
            file=sys.stderr,
        )
        sys.exit(1)


def _row_to_cleaned_message(row: Mapping[Any, Any], pd_module: Any) -> Any:
    from ricc.ingestion.schema import CleanedMessage

    reply_raw = row.get("reply_to_id")
    thread_raw = row.get("thread_id")
    tags_raw = row.get("tags")
    domain_raw = row.get("domain")
    sent_raw = row.get("sentiment_score")
    tok_raw = row.get("token_count_approx")
    q_raw = row.get("is_question")

    return CleanedMessage(
        msg_id=int(row["msg_id"]),
        chat_id=int(row["chat_id"]),
        chat_name=str(row["chat_name"]),
        timestamp=str(row["timestamp"]),
        unixtime=int(row["unixtime"]),
        author_anon=str(row["author_anon"]),
        author_id_anon=str(row["author_id_anon"]),
        text_clean=str(row["text_clean"]) if row.get("text_clean") is not None else "",
        reply_to_id=int(reply_raw) if reply_raw is not None and pd_module.notna(reply_raw) else None,
        domain=str(domain_raw) if domain_raw is not None and pd_module.notna(domain_raw) else "general_tech_chat",
        tags=[str(t) for t in tags_raw] if isinstance(tags_raw, (list, tuple)) else [],
        sentiment_score=int(sent_raw) if sent_raw is not None and pd_module.notna(sent_raw) else 0,
        token_count_approx=int(tok_raw) if tok_raw is not None and pd_module.notna(tok_raw) else 0,
        is_question=bool(q_raw) if q_raw is not None and pd_module.notna(q_raw) else False,
        thread_id=int(thread_raw) if thread_raw is not None and pd_module.notna(thread_raw) else None,
    )


def _cmd_analyze(args: argparse.Namespace) -> None:
    from ricc.analytics.engine import DeepChatAnalyzer
    from ricc.analytics.report_generator import ReportGenerator
    from ricc.ingestion.schema import CleanedMessage

    print("Loading messages for analytical profiling...")
    clean_parquet_path = settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet"

    if clean_parquet_path.exists():
        import pandas as pd

        print(f"Loading pre-cleaned dataset from {clean_parquet_path}...")
        df = pd.read_parquet(clean_parquet_path)
        cleaned = [_row_to_cleaned_message(rec, pd) for rec in df.to_dict("records")]
        print(f"Loaded {len(cleaned):,} cleaned and tagged messages from Parquet.")
    else:
        from ricc.ingestion.loader import merge_multiple_exports

        _chats_info, msgs = merge_multiple_exports(settings.RAW_EXPORT_DIRS)
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
    analysis_report = analyzer.run_full_analysis()
    gen = ReportGenerator(analysis_report)
    gen.print_terminal_summary()
    gen.export_markdown(settings.REPORTS_DIR / "DEEP_ANALYTICAL_REPORT.md")
    gen.export_json(settings.REPORTS_DIR / "analytics_summary.json")


def _cmd_validate(_args: argparse.Namespace) -> None:
    from ricc.validation.validator import DatasetValidator

    print("Validating datasets in output directory...")
    validator = DatasetValidator(settings.OUTPUT_DIR)
    res = validator.validate_all()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not res.get("overall_passed", False):
        print("Validation FAILED (overall_passed=false)", file=sys.stderr)
        sys.exit(1)


def _cmd_benchmark(_args: argparse.Namespace) -> None:
    from ricc.validation.benchmark import BenchmarkRunner

    print("Generating domain evaluation benchmark...")
    bench = BenchmarkRunner()
    out = bench.export_benchmark_file(settings.REPORTS_DIR / "domain_benchmark_100.json")
    print(f"Benchmark saved to {out} (Total questions: {len(bench.questions)})")


def _cmd_audit_prob(args: argparse.Namespace) -> None:
    from ricc.validation.probabilistic_audit import ProbabilisticPIIAuditor

    print("Running stratified probabilistic PII audit...")
    auditor = ProbabilisticPIIAuditor(args.parquet)
    audit_report = auditor.run_audit(
        sample_size=args.sample_size,
        confidence=args.confidence,
        max_leak_tolerance=args.tolerance,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verdict: {audit_report.get('verdict', audit_report.get('status'))}")
    print(f"Report saved to {out}")


def _cmd_drift(args: argparse.Namespace) -> None:
    import pandas as pd

    from ricc.monitoring.drift import DatasetDriftMonitor

    print("Running dataset drift monitoring...")
    if not args.reference.exists() or not args.current.exists():
        print(
            f"Snapshots missing: reference={args.reference.exists()} current={args.current.exists()}",
            file=sys.stderr,
        )
        print("Refusing to emit a vacuous drift report (fail-closed).", file=sys.stderr)
        sys.exit(2)
    try:
        same_snapshot = args.reference.resolve() == args.current.resolve()
    except OSError:
        same_snapshot = args.reference == args.current
    if same_snapshot:
        print("reference == current: self-comparison verdict is informational, not a real drift check.")
    monitor = DatasetDriftMonitor(pd.read_parquet(args.reference), pd.read_parquet(args.current))
    drift_report = monitor.run()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drift_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Overall drift verdict: {drift_report['overall_verdict']}")
    print(f"Report saved to {out}")


def _cmd_orchestrate(args: argparse.Namespace) -> None:
    from ricc.orchestration.prefect_flow import run_flow

    print("Executing Prefect orchestration flow...")
    results = run_flow(run_pipeline=not args.skip_pipeline, audit_sample_size=args.sample_size)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


def _cmd_chat(args: argparse.Namespace) -> None:
    from ricc.inference import interactive_chat_session

    interactive_chat_session(
        model_name=args.model,
        adapter_id=args.adapter,
        use_rag=not args.no_rag,
    )


def _cmd_rag(args: argparse.Namespace) -> None:
    from ricc.rag.rag_pipeline import LocalRAGPipeline

    kb_path = settings.PARQUET_OUTPUT_DIR / "rag_knowledge_base.parquet"
    if not kb_path.exists():
        print(f"RAG knowledge base not found at {kb_path}")
        sys.exit(1)
    rag = LocalRAGPipeline(kb_path)
    hits = rag.search(args.query, top_k=args.top_k)
    print(f"\nTop {len(hits)} RAG Results for: '{args.query}'\n" + "=" * 60)
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit['score']:.3f} | Domain: {hit.get('domain', 'general')}")
        print(f"    {hit['content']}\n" + "-" * 60)


_HANDLERS = {
    "run": _cmd_run,
    "analyze": _cmd_analyze,
    "validate": _cmd_validate,
    "benchmark": _cmd_benchmark,
    "audit-prob": _cmd_audit_prob,
    "drift": _cmd_drift,
    "orchestrate": _cmd_orchestrate,
    "chat": _cmd_chat,
    "rag": _cmd_rag,
}


def main() -> None:
    setup_runtime_env()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None or args.command not in _HANDLERS:
        parser.print_help()
        sys.exit(2)

    _HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
