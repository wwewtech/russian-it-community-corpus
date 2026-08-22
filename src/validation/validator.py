"""
Dataset Validation Suite for verifying data integrity, schema conformance, and zero-PII leak.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.pii.regex_scrubber import RegexPIIScrubber

logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Validates exported Parquet and JSONL datasets for structural integrity and PII safety.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.regex_scrubber = RegexPIIScrubber()

    def validate_all(self) -> dict[str, Any]:
        """Run all validation checks and return test results dictionary."""
        results = {
            "parquet_files": self.validate_parquet_files(),
            "jsonl_files": self.validate_jsonl_files(),
            "pii_leakage_audit": self.audit_pii_leakage(),
            "sft_turn_conformance": self.validate_sft_turn_structures(),
        }

        all_passed = (
            results["parquet_files"]["passed"]
            and results["jsonl_files"]["passed"]
            and results["pii_leakage_audit"]["passed"]
            and results["sft_turn_conformance"]["passed"]
        )
        results["overall_passed"] = all_passed
        logger.info(f"Dataset validation completed. Status: {'PASSED' if all_passed else 'FAILED'}")
        return results

    def validate_parquet_files(self) -> dict[str, Any]:
        """Check all Parquet files for existence, valid schema, and row counts."""
        parquet_dir = self.dataset_dir / "parquet"
        files_to_check = [
            "full_clean_messages.parquet",
            "sft_dialogues.parquet",
            "rag_knowledge_base.parquet",
        ]

        details = {}
        all_ok = True

        for fname in files_to_check:
            fpath = parquet_dir / fname
            if not fpath.exists():
                details[fname] = {"exists": False, "rows": 0, "error": "File not found"}
                all_ok = False
                continue

            try:
                table = pq.read_table(fpath)
                row_count = table.num_rows
                col_count = table.num_columns
                size_mb = round(fpath.stat().st_size / (1024 * 1024), 2)
                details[fname] = {
                    "exists": True,
                    "rows": row_count,
                    "columns": col_count,
                    "size_mb": size_mb,
                    "status": "VALID" if row_count > 0 else "EMPTY",
                }
                if row_count == 0:
                    all_ok = False
            except Exception as e:
                details[fname] = {"exists": True, "error": str(e), "status": "CORRUPT"}
                all_ok = False

        return {"passed": all_ok, "details": details}

    def validate_jsonl_files(self) -> dict[str, Any]:
        """Check JSONL files for line-by-line JSON validity."""
        jsonl_dir = self.dataset_dir / "jsonl"
        files_to_check = [
            "sft_sharegpt_format.jsonl",
            "sft_alpaca_format.jsonl",
            "sft_openai_messages.jsonl",
            "rag_chunks_kb.jsonl",
            "dpo_preference_pairs.jsonl",
        ]

        details = {}
        all_ok = True

        for fname in files_to_check:
            fpath = jsonl_dir / fname
            if not fpath.exists():
                details[fname] = {"exists": False, "lines": 0, "error": "File not found"}
                all_ok = False
                continue

            try:
                valid_lines = 0
                corrupt_lines = 0
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            valid_lines += 1
                        except json.JSONDecodeError:
                            corrupt_lines += 1

                size_mb = round(fpath.stat().st_size / (1024 * 1024), 2)
                is_valid = corrupt_lines == 0 and valid_lines > 0
                details[fname] = {
                    "exists": True,
                    "valid_lines": valid_lines,
                    "corrupt_lines": corrupt_lines,
                    "size_mb": size_mb,
                    "status": "VALID" if is_valid else "INVALID",
                }
                if not is_valid:
                    all_ok = False
            except Exception as e:
                details[fname] = {"exists": True, "error": str(e), "status": "CORRUPT"}
                all_ok = False

        return {"passed": all_ok, "details": details}

    def audit_pii_leakage(self, sample_lines: int = 10000) -> dict[str, Any]:
        """Sample exported datasets and audit for unmasked PII."""
        jsonl_path = self.dataset_dir / "jsonl" / "sft_openai_messages.jsonl"
        if not jsonl_path.exists():
            return {"passed": False, "error": "sft_openai_messages.jsonl not found"}

        leaks = {
            "unmasked_phones": 0,
            "unmasked_emails": 0,
            "unmasked_api_keys": 0,
            "unmasked_crypto": 0,
        }

        checked_lines = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if checked_lines >= sample_lines:
                    break
                checked_lines += 1
                try:
                    data = json.loads(line)
                    text_blob = " ".join(m.get("content", "") for m in data.get("messages", []))

                    # Audit regexes
                    clean_res, reg_stats = self.regex_scrubber.scrub(text_blob)
                    leaks["unmasked_phones"] += reg_stats.get("phones", 0)
                    leaks["unmasked_emails"] += reg_stats.get("emails", 0)
                    leaks["unmasked_api_keys"] += reg_stats.get("api_keys", 0)
                    leaks["unmasked_crypto"] += reg_stats.get("crypto_wallets", 0)
                except Exception:
                    pass

        total_leaks = sum(leaks.values())
        return {
            "passed": total_leaks == 0,
            "sample_lines_checked": checked_lines,
            "leaks_found": leaks,
            "total_leak_count": total_leaks,
        }

    def validate_sft_turn_structures(self) -> dict[str, Any]:
        """Validate that SFT dialogues follow alternating user/assistant turns."""
        jsonl_path = self.dataset_dir / "jsonl" / "sft_openai_messages.jsonl"
        if not jsonl_path.exists():
            return {"passed": False, "error": "sft_openai_messages.jsonl not found"}

        conforming_dialogues = 0
        non_conforming = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    msgs = data.get("messages", [])
                    # Skip system prompt
                    dialogue_msgs = [m for m in msgs if m.get("role") != "system"]

                    # Check alternation: user -> assistant -> user -> assistant...
                    roles = [m.get("role") for m in dialogue_msgs]
                    valid_roles = True
                    for i in range(len(roles) - 1):
                        if roles[i] == roles[i + 1]:
                            valid_roles = False
                            break
                    if roles and roles[0] != "user":
                        valid_roles = False
                    if roles and roles[-1] != "assistant":
                        valid_roles = False

                    if valid_roles:
                        conforming_dialogues += 1
                    else:
                        non_conforming += 1
                except Exception:
                    non_conforming += 1

        return {
            "passed": non_conforming == 0,
            "conforming_dialogues": conforming_dialogues,
            "non_conforming_dialogues": non_conforming,
        }
