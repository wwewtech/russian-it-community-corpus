"""
Master Pipeline Orchestrator for IT Community Data Engineering & Curation.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.analytics.engine import DeepChatAnalyzer
from src.analytics.report_generator import ReportGenerator
from src.config import (
    JSONL_OUTPUT_DIR,
    OUTPUT_DIR,
    PARQUET_OUTPUT_DIR,
    RAW_EXPORT_DIRS,
    REPORTS_DIR,
    SAMPLES_OUTPUT_DIR,
)
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
from src.pii.anonymizer import UnifiedPIIAnonymizer
from src.taxonomy.tagger import TechnicalTagger
from src.validation.benchmark import BenchmarkRunner
from src.validation.validator import DatasetValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Pipeline")


class MasterDataPipeline:
    """
    Complete end-to-end data curation pipeline.
    """

    def __init__(self, raw_export_dirs: List[Path] = RAW_EXPORT_DIRS):
        self.raw_export_dirs = raw_export_dirs
        self.anonymizer = UnifiedPIIAnonymizer(enable_ner=True)
        self.exact_dedup = ExactDeduplicator()
        self.minhash_lsh = MinHashLSH()
        self.tagger = TechnicalTagger()
        self.dag_builder = ThreadDAGBuilder()
        self.conv_extractor = ConversationExtractor()
        
        # Exporters
        self.parquet_exporter = ParquetExporter(PARQUET_OUTPUT_DIR)
        self.jsonl_exporter = JSONLExporter(JSONL_OUTPUT_DIR)
        self.rag_exporter = RAGExporter(JSONL_OUTPUT_DIR)
        self.dpo_exporter = DPOExporter(JSONL_OUTPUT_DIR)

        # Output states
        self.raw_messages: List[NormalizedMessage] = []
        self.cleaned_messages: List[CleanedMessage] = []
        self.threads: Dict[int, List[CleanedMessage]] = {}
        self.sft_dialogues: List[SFTDialogue] = []
        self.rag_chunks: List[RAGChunk] = []
        self.dpo_pairs: List[Dict[str, Any]] = []
        self.analytics_report: Dict[str, Any] = {}
        self.validation_report: Dict[str, Any] = {}

    def run_all(self) -> Dict[str, Any]:
        """Execute all pipeline stages in sequence."""
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("🚀 STARTING MASTER DATA ENGINEERING & CURATION PIPELINE")
        logger.info("=" * 70)

        # Stage 1: Ingestion
        t0 = time.time()
        logger.info("[Stage 1/7] Ingesting and Merging Raw Telegram Exports...")
        chats_info, self.raw_messages = merge_multiple_exports(self.raw_export_dirs)
        t_ingest = time.time() - t0
        logger.info(f"✅ Ingestion complete: {len(self.raw_messages):,} messages loaded in {t_ingest:.2f}s")

        # Stage 2: PII Removal & User Pseudonymization
        t0 = time.time()
        logger.info("[Stage 2/7] Executing Dual-Layer PII Scrubbing (Regex + NER) & Anonymization...")
        self.cleaned_messages = self.anonymizer.process_batch(self.raw_messages)
        pii_stats = self.anonymizer.get_stats_summary()
        t_pii = time.time() - t0
        logger.info(f"✅ PII Scrubbing complete in {t_pii:.2f}s. Redacted {pii_stats.get('phones_scrubbed', 0)} phones, "
                    f"{pii_stats.get('emails_scrubbed', 0)} emails, {pii_stats.get('crypto_wallets_scrubbed', 0)} wallets, "
                    f"{pii_stats.get('api_keys_scrubbed', 0)} API keys.")

        # Stage 3: Exact & MinHash LSH Deduplication
        t0 = time.time()
        logger.info("[Stage 3/7] Deduplicating messages (Exact Hashing + MinHash LSH)...")
        unique_exact, exact_dupes = self.exact_dedup.deduplicate(self.cleaned_messages)
        self.cleaned_messages, lsh_dupes = self.minhash_lsh.deduplicate_messages(unique_exact)
        t_dedup = time.time() - t0
        logger.info(f"✅ Deduplication complete in {t_dedup:.2f}s. Removed {exact_dupes + lsh_dupes:,} duplicate/spam messages.")

        # Stage 4: Domain Taxonomy & Technical Keyword Tagging
        t0 = time.time()
        logger.info("[Stage 4/7] Classifying IT Domains and Extracting Technical Tags...")
        self.cleaned_messages = self.tagger.tag_batch(self.cleaned_messages)
        t_tag = time.time() - t0
        logger.info(f"✅ Tagging complete in {t_tag:.2f}s.")

        # Stage 5: Conversation Graph DAG & SFT/RAG Extraction
        t0 = time.time()
        logger.info("[Stage 5/7] Reconstructing Conversation Thread DAGs and Extracting SFT/RAG datasets...")
        self.cleaned_messages, self.threads = self.dag_builder.build_threads(self.cleaned_messages)
        self.sft_dialogues = self.conv_extractor.extract_sft_dialogues(self.threads)
        self.rag_chunks = self.conv_extractor.extract_rag_chunks(self.threads)
        self.dpo_pairs = self.conv_extractor.extract_dpo_pairs(self.threads)
        t_graph = time.time() - t0
        logger.info(f"✅ Extraction complete in {t_graph:.2f}s: {len(self.threads):,} threads, "
                    f"{len(self.sft_dialogues):,} SFT dialogues, {len(self.rag_chunks):,} RAG chunks, "
                    f"{len(self.dpo_pairs):,} DPO pairs.")

        # Stage 6: Multi-Format Production Exporter
        t0 = time.time()
        logger.info("[Stage 6/7] Exporting Datasets to Apache Parquet, ShareGPT, Alpaca, ChatML, and RAG JSONL...")
        # Parquet
        p_msgs = self.parquet_exporter.export_messages(self.cleaned_messages)
        p_sft = self.parquet_exporter.export_sft_dialogues(self.sft_dialogues)
        p_rag = self.parquet_exporter.export_rag_chunks(self.rag_chunks)
        
        # JSONL
        j_sharegpt = self.jsonl_exporter.export_sharegpt(self.sft_dialogues)
        j_alpaca = self.jsonl_exporter.export_alpaca(self.sft_dialogues)
        j_chatml = self.jsonl_exporter.export_openai_chatml(self.sft_dialogues)
        j_rag = self.rag_exporter.export_rag_jsonl(self.rag_chunks)
        j_dpo = self.dpo_exporter.export_dpo_pairs(self.dpo_pairs)

        # Export sample previews
        SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(SAMPLES_OUTPUT_DIR / "sft_sample_preview.json", "w", encoding="utf-8") as f:
            sample_sft = [d.model_dump() for d in self.sft_dialogues[:15]]
            json.dump(sample_sft, f, ensure_ascii=False, indent=2)

        with open(SAMPLES_OUTPUT_DIR / "rag_sample_preview.json", "w", encoding="utf-8") as f:
            sample_rag = [c.model_dump() for c in self.rag_chunks[:15]]
            json.dump(sample_rag, f, ensure_ascii=False, indent=2)

        with open(SAMPLES_OUTPUT_DIR / "dpo_sample_preview.json", "w", encoding="utf-8") as f:
            json.dump(self.dpo_pairs[:15], f, ensure_ascii=False, indent=2)

        t_export = time.time() - t0
        logger.info(f"✅ Export complete in {t_export:.2f}s.")

        # Stage 7: Analytics, Reports & Validation
        t0 = time.time()
        logger.info("[Stage 7/7] Running Deep Analytics Engine and Validation Suite...")
        analyzer = DeepChatAnalyzer(self.cleaned_messages, sample_limit_for_nlp=100000)
        self.analytics_report = analyzer.run_full_analysis()

        # Generate Reports
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_gen = ReportGenerator(self.analytics_report)
        report_gen.export_json(REPORTS_DIR / "analytics_summary.json")
        report_gen.export_markdown(REPORTS_DIR / "DEEP_ANALYTICAL_REPORT.md")

        # Export Benchmark
        bench = BenchmarkRunner()
        bench.export_benchmark_file(REPORTS_DIR / "domain_benchmark_100.json")

        # Validation Suite
        validator = DatasetValidator(OUTPUT_DIR)
        self.validation_report = validator.validate_all()
        with open(REPORTS_DIR / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump(self.validation_report, f, ensure_ascii=False, indent=2)

        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info("=" * 70)

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
        with open(REPORTS_DIR / "pipeline_execution_stats.json", "w", encoding="utf-8") as f:
            json.dump(exec_summary, f, ensure_ascii=False, indent=2)

        return exec_summary
