#!/usr/bin/env python3
"""DVC pipeline stages, one module per stage.

The stages used to be inline ``python -c "..."`` blocks in ``dvc.yaml``.
That form does not survive a Windows shell: ``cmd.exe`` splits the
multi-line quoted string and the pipeline dies with
``'from' is not recognized as an internal or external command`` — so
``dvc repro`` had never run, on any platform, in practice.

Each stage is now a real script with a ``main()``, invocable both by DVC
(``python scripts/dvc_stages.py <stage>``) and by hand, which also makes
the stages unit-testable.

Usage::

    python scripts/dvc_stages.py ingest
    python scripts/dvc_stages.py validate
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

INTERMEDIATE = REPO_ROOT / "dataset_output" / "intermediate"
REPORTS = REPO_ROOT / "reports"
PARQUET = REPO_ROOT / "dataset_output" / "parquet"


def _load_params() -> dict[str, Any]:
    import yaml

    params_path = REPO_ROOT / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------- #
# Stage 1 — ingestion
# --------------------------------------------------------------------------- #


def stage_ingest() -> None:
    from ricc.config import RAW_EXPORT_DIRS
    from ricc.ingestion.loader import merge_multiple_exports

    chats, messages = merge_multiple_exports(list(RAW_EXPORT_DIRS))
    _save((chats, messages), INTERMEDIATE / "raw_messages.pkl.gz")
    print(f"Ingested {len(messages)} messages from {len(chats)} chats")


# --------------------------------------------------------------------------- #
# Stage 2 — PII anonymization
# --------------------------------------------------------------------------- #


def stage_pii_anonymize() -> None:
    from ricc.pii.deep_anonymizer import DeepPIIAnonymizer

    _chats, raw_messages = _load(INTERMEDIATE / "raw_messages.pkl.gz")
    anonymizer = DeepPIIAnonymizer(enable_ner=True)
    anonymizer.register_authors([(m.author_id_raw, m.author_raw) for m in raw_messages])
    cleaned = [anonymizer.anonymize_message(m) for m in raw_messages]
    _save(cleaned, INTERMEDIATE / "cleaned_messages.pkl.gz")
    print(f"Anonymized {len(cleaned)} messages | PII stats: {anonymizer.stats}")


# --------------------------------------------------------------------------- #
# Stage 3 — exact deduplication
# --------------------------------------------------------------------------- #


def stage_exact_dedup() -> None:
    from ricc.deduplication.exact_dedup import ExactDeduplicator

    cleaned = _load(INTERMEDIATE / "cleaned_messages.pkl.gz")
    unique, removed = ExactDeduplicator().deduplicate(cleaned)
    _save(unique, INTERMEDIATE / "dedup_exact.pkl.gz")
    print(f"Exact dedup: removed {removed} duplicates, {len(unique)} remaining")


# --------------------------------------------------------------------------- #
# Stage 4 — MinHash LSH deduplication
# --------------------------------------------------------------------------- #


def stage_minhash_dedup() -> None:
    from ricc.config import MINHASH_NUM_PERM, MINHASH_SHINGLE_SIZE, MINHASH_THRESHOLD
    from ricc.deduplication.minhash_lsh import MinHashLSH

    messages = _load(INTERMEDIATE / "dedup_exact.pkl.gz")
    lsh = MinHashLSH(
        num_perm=MINHASH_NUM_PERM,
        threshold=MINHASH_THRESHOLD,
        shingle_size=MINHASH_SHINGLE_SIZE,
    )
    unique, removed = lsh.deduplicate_messages(messages)
    _save(unique, INTERMEDIATE / "dedup_minhash.pkl.gz")
    print(f"MinHash LSH dedup: removed {removed} near-duplicates, {len(unique)} remaining")


# --------------------------------------------------------------------------- #
# Stage 5 — taxonomy tagging
# --------------------------------------------------------------------------- #


def stage_taxonomy() -> None:
    from ricc.taxonomy.tagger import TechnicalTagger

    messages = _load(INTERMEDIATE / "dedup_minhash.pkl.gz")
    tagged = TechnicalTagger().tag_batch(messages)
    _save(tagged, INTERMEDIATE / "tagged_messages.pkl.gz")
    print(f"Tagged {len(tagged)} messages")


# --------------------------------------------------------------------------- #
# Stage 6 — thread DAG
# --------------------------------------------------------------------------- #


def stage_graph() -> None:
    from ricc.graph.thread_builder import ThreadDAGBuilder

    messages = _load(INTERMEDIATE / "tagged_messages.pkl.gz")
    builder = ThreadDAGBuilder()
    with_threads, threads = builder.build_threads(messages)
    _save((with_threads, threads), INTERMEDIATE / "threaded_messages.pkl.gz")
    print(f"Built {len(threads)} threads from {len(with_threads)} messages")


# --------------------------------------------------------------------------- #
# Stage 7 — SFT / RAG / DPO extraction
# --------------------------------------------------------------------------- #


def stage_extract() -> None:
    from ricc.graph.conversation_extractor import ConversationExtractor

    messages, threads = _load(INTERMEDIATE / "threaded_messages.pkl.gz")
    extractor = ConversationExtractor()
    sft = extractor.extract_sft_dialogues(threads)
    rag = extractor.extract_rag_chunks(threads)
    dpo = extractor.extract_dpo_pairs(threads)
    _save((sft, rag, dpo), INTERMEDIATE / "extracted.pkl.gz")
    print(f"Extracted {len(sft)} SFT dialogues, {len(rag)} RAG chunks, {len(dpo)} DPO pairs")


# --------------------------------------------------------------------------- #
# Stage 8 — export
# --------------------------------------------------------------------------- #


def stage_export() -> None:
    from ricc.exporter.dpo_exporter import DPOExporter
    from ricc.exporter.jsonl_exporter import JSONLExporter
    from ricc.exporter.parquet_exporter import ParquetExporter
    from ricc.exporter.rag_exporter import RAGExporter

    sft, rag, dpo = _load(INTERMEDIATE / "extracted.pkl.gz")
    messages = _messages()
    ParquetExporter(PARQUET).export_messages(messages, "full_clean_messages.parquet")
    ParquetExporter(PARQUET).export_sft_dialogues(sft)
    ParquetExporter(PARQUET).export_rag_chunks(rag)
    JSONLExporter(REPO_ROOT / "dataset_output" / "jsonl").export_sharegpt(sft)
    JSONLExporter(REPO_ROOT / "dataset_output" / "jsonl").export_alpaca(sft)
    JSONLExporter(REPO_ROOT / "dataset_output" / "jsonl").export_openai_chatml(sft)
    RAGExporter(REPO_ROOT / "dataset_output" / "jsonl").export_rag_jsonl(rag)
    DPOExporter(REPO_ROOT / "dataset_output" / "jsonl").export_dpo_pairs(dpo)
    print("Exported parquet + jsonl artefacts")


def _messages() -> list[Any]:
    messages, _threads = _load(INTERMEDIATE / "threaded_messages.pkl.gz")
    return messages


# --------------------------------------------------------------------------- #
# Stage 9 — analytics
# --------------------------------------------------------------------------- #


def stage_analytics() -> None:
    from ricc.analytics.engine import DeepChatAnalyzer
    from ricc.analytics.report_generator import ReportGenerator
    from ricc.config import REPORTS_DIR
    from ricc.validation.benchmark import BenchmarkRunner

    messages = _messages()
    report = DeepChatAnalyzer(messages, sample_limit_for_nlp=100000).run_full_analysis()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generator = ReportGenerator(report)
    generator.export_json(REPORTS_DIR / "analytics_summary.json")
    generator.export_markdown(REPORTS_DIR / "DEEP_ANALYTICAL_REPORT.md")
    BenchmarkRunner().export_benchmark_file(REPORTS_DIR / "domain_benchmark_100.json")
    print("Analytics + reports written")


# --------------------------------------------------------------------------- #
# Stage 10 — validation + probabilistic PII audit
# --------------------------------------------------------------------------- #


def stage_validate() -> None:
    from ricc.config import OUTPUT_DIR, REPORTS_DIR
    from ricc.validation.probabilistic_audit import ProbabilisticPIIAuditor
    from ricc.validation.validator import DatasetValidator

    params = _load_params().get("pii_audit", {})
    sample_size = int(params.get("sample_size", 100_000))
    confidence = float(params.get("confidence", 0.99))
    tolerance = float(params.get("max_leak_tolerance", 1e-4))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result = DatasetValidator(OUTPUT_DIR).validate_all()
    (REPORTS_DIR / "validation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit = ProbabilisticPIIAuditor(PARQUET / "full_clean_messages.parquet").run_audit(
        sample_size=sample_size,
        confidence=confidence,
        max_leak_tolerance=tolerance,
    )
    (REPORTS_DIR / "probabilistic_pii_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validation: {'PASSED' if result['overall_passed'] else 'FAILED'}")
    print(f"PII Audit: {audit.get('verdict', audit.get('status'))}")


# --------------------------------------------------------------------------- #
# Stage 11 — drift monitoring
# --------------------------------------------------------------------------- #


def stage_drift() -> None:
    import pandas as pd

    from ricc.monitoring.drift import DatasetDriftMonitor

    # The reference is a FROZEN snapshot, never the current file: comparing
    # the corpus against itself can only ever report "stable", which made
    # the drift gate informational in practice.
    reference = Path(
        os.environ.get(
            "RICC_DRIFT_REFERENCE",
            str(PARQUET / "full_clean_messages_reference.parquet"),
        )
    )
    current = PARQUET / "full_clean_messages.parquet"

    if not reference.exists() or not current.exists():
        print(
            f"Skipping drift: no reference snapshot at {reference}. "
            "Set RICC_DRIFT_REFERENCE to a frozen earlier release to enable "
            "the temporal drift signal."
        )
        return

    monitor = DatasetDriftMonitor(pd.read_parquet(reference), pd.read_parquet(current))
    report = monitor.run()
    monitor.generate_report(REPORTS / "drift_report.json")
    print(f"Drift verdict: {report['overall_verdict']}")


# --------------------------------------------------------------------------- #
# Stage 12 — SLO gate
# --------------------------------------------------------------------------- #


def stage_slo() -> None:
    from ricc.monitoring.slo_gate import evaluate

    verdict = evaluate()
    payload = json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2)
    (REPORTS / "slo_verdict.json").write_text(payload, encoding="utf-8")
    print(payload)
    if verdict.verdict != "SHIP":
        raise SystemExit(f"SLO gate verdict: {verdict.verdict}")


STAGES = {
    "ingest": stage_ingest,
    "pii_anonymize": stage_pii_anonymize,
    "exact_dedup": stage_exact_dedup,
    "minhash_dedup": stage_minhash_dedup,
    "taxonomy": stage_taxonomy,
    "graph": stage_graph,
    "extract": stage_extract,
    "export": stage_export,
    "analytics": stage_analytics,
    "validate": stage_validate,
    "drift": stage_drift,
    "slo": stage_slo,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="RICC DVC pipeline stages")
    parser.add_argument("stage", choices=sorted(STAGES), help="Stage to run")
    args = parser.parse_args()
    STAGES[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
