"""
Master Pipeline Orchestrator for IT Community Data Engineering & Curation.
"""

import json
import time
from pathlib import Path
from typing import Any, cast

from src.analytics.engine import DeepChatAnalyzer, FullAnalysisReport
from src.analytics.report_generator import ReportGenerator
from src.settings import settings
from src.deduplication.exact_dedup import ExactDeduplicator
from src.deduplication.minhash_lsh import MinHashLSH
from src.exporter.dpo_exporter import DPOExporter
from src.exporter.jsonl_exporter import JSONLExporter
from src.exporter.parquet_exporter import ParquetExporter
from src.exporter.rag_exporter import RAGExporter
from src.graph.conversation_extractor import ConversationExtractor
from src.graph.thread_builder import ThreadDAGBuilder
from src.ingestion.loader import merge_multiple_exports
from src.ingestion.schema import CleanedMessage, NormalizedMessage, RAGChunk, SFTDialogue
from src.observability import (
    PIPELINE_MESSAGES_PROCESSED,
    PIPELINE_STAGE_DURATION,
    PIPELINE_STAGE_TOTAL,
    get_logger,
    pipeline_logger,
    record_stage_messages,
    time_stage,
)
from src.pii.anonymizer import UnifiedPIIAnonymizer
from src.taxonomy.tagger import TechnicalTagger
from src.validation.benchmark import BenchmarkRunner
from src.validation.validator import DatasetValidator


class MasterDataPipeline:
    """
    Complete end-to-end data curation pipeline.
    """

    def __init__(self, raw_export_dirs: list[Path] | None = None):
        self.raw_export_dirs = raw_export_dirs or settings.RAW_EXPORT_DIRS
        self.anonymizer = UnifiedPIIAnonymizer(enable_ner=True)
        self.exact_dedup = ExactDeduplicator()
        self.minhash_lsh = MinHashLSH()
        self.tagger = TechnicalTagger()
        self.dag_builder = ThreadDAGBuilder()
        self.conv_extractor = ConversationExtractor()

        # Exporters
        self.parquet_exporter = ParquetExporter(settings.PARQUET_OUTPUT_DIR)
        self.jsonl_exporter = JSONLExporter(settings.JSONL_OUTPUT_DIR)
        self.rag_exporter = RAGExporter(settings.JSONL_OUTPUT_DIR)
        self.dpo_exporter = DPOExporter(settings.JSONL_OUTPUT_DIR)

        # Output states
        self.raw_messages: list[NormalizedMessage] = []
        self.cleaned_messages: list[CleanedMessage] = []
        self.threads: dict[int, list[CleanedMessage]] = {}
        self.sft_dialogues: list[SFTDialogue] = []
        self.rag_chunks: list[RAGChunk] = []
        self.dpo_pairs: list[dict[str, Any]] = []
        self.analytics_report: FullAnalysisReport = {}
        self.validation_report: dict[str, Any] = {}

    def run_all(self) -> dict[str, Any]:
        """Execute all pipeline stages in sequence."""
        start_time = time.time()
        pipeline_logger.stage_start("pipeline_start")
        pipeline_logger.progress("🚀 STARTING MASTER DATA ENGINEERING & CURATION PIPELINE")

        # Stage 1: Ingestion
        with time_stage("ingestion"):
            pipeline_logger.stage_start("ingestion")
            chats_info, self.raw_messages = merge_multiple_exports(self.raw_export_dirs)
            record_stage_messages("ingestion", len(self.raw_messages))
            pipeline_logger.stage_end("ingestion", success=True, messages=len(self.raw_messages))

        # Stage 2: PII Removal & User Pseudonymization
        with time_stage("pii_anonymization"):
            pipeline_logger.stage_start("pii_anonymization")
            self.cleaned_messages = self.anonymizer.process_batch(self.raw_messages)
            pii_stats = self.anonymizer.get_stats_summary()
            record_stage_messages("pii_anonymization", len(self.cleaned_messages))
            pipeline_logger.stage_end(
                "pii_anonymization",
                success=True,
                phones=pii_stats.get("phones_scrubbed", 0),
                emails=pii_stats.get("emails_scrubbed", 0),
                wallets=pii_stats.get("crypto_wallets_scrubbed", 0),
                api_keys=pii_stats.get("api_keys_scrubbed", 0),
            )

        # Stage 3: Exact & MinHash LSH Deduplication
        with time_stage("deduplication"):
            pipeline_logger.stage_start("deduplication")
            unique_exact, exact_dupes = self.exact_dedup.deduplicate(self.cleaned_messages)
            self.cleaned_messages, lsh_dupes = self.minhash_lsh.deduplicate_messages(unique_exact)
            total_removed = exact_dupes + lsh_dupes
            record_stage_messages("deduplication", total_removed)
            pipeline_logger.stage_end("deduplication", success=True, removed=total_removed)

        # Stage 4: Domain Taxonomy & Technical Keyword Tagging
        with time_stage("taxonomy_tagging"):
            pipeline_logger.stage_start("taxonomy_tagging")
            self.cleaned_messages = self.tagger.tag_batch(self.cleaned_messages)
            record_stage_messages("taxonomy_tagging", len(self.cleaned_messages))
            pipeline_logger.stage_end("taxonomy_tagging", success=True)

        # Stage 5: Conversation Graph DAG & SFT/RAG Extraction
        with time_stage("graph_extraction"):
            pipeline_logger.stage_start("graph_extraction")
            self.cleaned_messages, self.threads = self.dag_builder.build_threads(self.cleaned_messages)
            self.sft_dialogues = self.conv_extractor.extract_sft_dialogues(self.threads)
            self.rag_chunks = self.conv_extractor.extract_rag_chunks(self.threads)
            self.dpo_pairs = self.conv_extractor.extract_dpo_pairs(self.threads)
            pipeline_logger.stage_end(
                "graph_extraction",
                success=True,
                threads=len(self.threads),
                sft_dialogues=len(self.sft_dialogues),
                rag_chunks=len(self.rag_chunks),
                dpo_pairs=len(self.dpo_pairs),
            )

        # Stage 6: Multi-Format Production Exporter
        with time_stage("export"):
            pipeline_logger.stage_start("export")
            # Parquet
            self.parquet_exporter.export_messages(self.cleaned_messages)
            self.parquet_exporter.export_sft_dialogues(self.sft_dialogues)
            self.parquet_exporter.export_rag_chunks(self.rag_chunks)

            # JSONL
            self.jsonl_exporter.export_sharegpt(self.sft_dialogues)
            self.jsonl_exporter.export_alpaca(self.sft_dialogues)
            self.jsonl_exporter.export_openai_chatml(self.sft_dialogues)
            self.rag_exporter.export_rag_jsonl(self.rag_chunks)
            self.dpo_exporter.export_dpo_pairs(self.dpo_pairs)

            # Export sample previews
            settings.SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(settings.SAMPLES_OUTPUT_DIR / "sft_sample_preview.json", "w", encoding="utf-8") as f:
                sample_sft = [d.model_dump() for d in self.sft_dialogues[:15]]
                json.dump(sample_sft, f, ensure_ascii=False, indent=2)

            with open(settings.SAMPLES_OUTPUT_DIR / "rag_sample_preview.json", "w", encoding="utf-8") as f:
                sample_rag = [c.model_dump() for c in self.rag_chunks[:15]]
                json.dump(sample_rag, f, ensure_ascii=False, indent=2)

            with open(settings.SAMPLES_OUTPUT_DIR / "dpo_sample_preview.json", "w", encoding="utf-8") as f:
                json.dump(self.dpo_pairs[:15], f, ensure_ascii=False, indent=2)

            pipeline_logger.stage_end("export", success=True)

        # Stage 7: Analytics, Reports & Validation
        with time_stage("analytics_validation"):
            pipeline_logger.stage_start("analytics_validation")
            analyzer = DeepChatAnalyzer(self.cleaned_messages, sample_limit_for_nlp=100000)
            self.analytics_report = analyzer.run_full_analysis()

            # Generate Reports
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_gen = ReportGenerator(cast(dict[str, Any], self.analytics_report))
            report_gen.export_json(settings.REPORTS_DIR / "analytics_summary.json")
            report_gen.export_markdown(settings.REPORTS_DIR / "DEEP_ANALYTICAL_REPORT.md")

            # Export Benchmark
            bench = BenchmarkRunner()
            bench.export_benchmark_file(settings.REPORTS_DIR / "domain_benchmark_100.json")

            # Validation Suite
            validator = DatasetValidator(settings.OUTPUT_DIR)
            self.validation_report = validator.validate_all()
            with open(settings.REPORTS_DIR / "validation_results.json", "w", encoding="utf-8") as f:
                json.dump(self.validation_report, f, ensure_ascii=False, indent=2)

            pipeline_logger.stage_end("analytics_validation", success=True)

        total_time = time.time() - start_time
        pipeline_logger.stage_end("pipeline_start", success=True, duration_seconds=round(total_time, 2))

        # Print Visual Terminal Summary
        report_gen.print_terminal_summary()

        # Save pipeline execution summary
        exec_summary = {
            "execution_time_seconds": round(total_time, 2),
            "raw_messages_count": len(self.raw_messages),
            "cleaned_messages_count": len(self.cleaned_messages),
            "threads_count": len(self.threads),
            "sft_dialogues_count": len(self.sft_dialogues),
            "rag_chunks_count": len(self.rag_chunks),
            "dpo_pairs_count": len(self.dpo_pairs),
            "pii_stats": pii_stats,
            "validation_passed": self.validation_report.get("overall_passed", False),
        }
        with open(settings.REPORTS_DIR / "pipeline_execution_stats.json", "w", encoding="utf-8") as f:
            json.dump(exec_summary, f, ensure_ascii=False, indent=2)

        return exec_summary
