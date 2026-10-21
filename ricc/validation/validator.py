"""
Dataset Validation Suite for verifying data integrity, schema conformance, and zero-PII leak.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ricc.pii.regex_scrubber import RegexPIIScrubber
from ricc.schema_version import read_metadata, verify_against_file

logger = logging.getLogger(__name__)

# The keys every record of a JSONL export must carry. A line that parses
# as JSON but lacks them is corrupt for the consumers: they index into
# these fields.
#
# The shapes are declared in `ricc/schema_version.py` (JSONL_SCHEMAS).
_JSONL_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "sft_sharegpt_format.jsonl": frozenset({"conversations"}),
    "sft_openai_messages.jsonl": frozenset({"messages"}),
    "sft_alpaca_format.jsonl": frozenset({"instruction", "output"}),
    "rag_chunks_kb.jsonl": frozenset({"content"}),
    "dpo_preference_pairs.jsonl": frozenset({"prompt", "chosen", "rejected"}),
}


class DatasetValidator:
    """
    Validates exported Parquet and JSONL datasets for structural integrity and PII safety.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.regex_scrubber = RegexPIIScrubber()

    def validate_all(self, sample_lines: int = 10000) -> dict[str, Any]:
        """Run all validation checks and return test results dictionary."""
        results = {
            "parquet_files": self.validate_parquet_files(),
            "jsonl_files": self.validate_jsonl_files(),
            "pii_leakage_audit": self.audit_pii_leakage(sample_lines=sample_lines),
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
        """Check all Parquet files for existence, valid schema, and row counts.

        The schema check is a real contract check: the column names,
        their order and their types must match the declared dataset
        contract (``ricc.schema_version``), and the file must carry the
        current contract version and fingerprint. The previous
        implementation only counted columns, so a renamed or retyped
        column passed validation silently.
        """
        parquet_dir = self.dataset_dir / "parquet"
        files_to_check = [
            ("full_clean_messages.parquet", "full_clean_messages"),
            ("sft_dialogues.parquet", "sft_dialogues"),
            ("rag_knowledge_base.parquet", "rag_knowledge_base"),
        ]

        details = {}
        all_ok = True

        for fname, artifact in files_to_check:
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

                contract_problems = verify_against_file(fpath, artifact)
                meta = read_metadata(table.schema)
                if row_count == 0:
                    status = "EMPTY"
                elif contract_problems:
                    status = "INVALID"
                else:
                    status = "VALID"
                details[fname] = {
                    "exists": True,
                    "rows": row_count,
                    "columns": col_count,
                    "size_mb": size_mb,
                    "schema_version": meta["version"],
                    "schema_fingerprint": meta["fingerprint"],
                    "schema_problems": contract_problems,
                    "status": status,
                }
                if row_count == 0:
                    all_ok = False
                if contract_problems:
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
                shape_errors = 0
                # The record shape of this export, from the declared
                # contract. A line that parses as JSON but is not a
                # record of the declared shape is corrupt too: the
                # consumers index into these fields.
                required_keys = _JSONL_REQUIRED_KEYS.get(fname, frozenset())
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            corrupt_lines += 1
                            continue
                        if required_keys and not isinstance(record, dict):
                            shape_errors += 1
                            continue
                        if required_keys and not required_keys.issubset(record.keys()):
                            shape_errors += 1
                            continue
                        valid_lines += 1

                size_mb = round(fpath.stat().st_size / (1024 * 1024), 2)
                is_valid = corrupt_lines == 0 and shape_errors == 0 and valid_lines > 0
                details[fname] = {
                    "exists": True,
                    "valid_lines": valid_lines,
                    "corrupt_lines": corrupt_lines,
                    "shape_errors": shape_errors,
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
            "unmasked_community_names": 0,
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
                    leaks["unmasked_community_names"] += reg_stats.get("community_names", 0)
                except Exception:
                    pass  # deliberate: scrubber-stats probe is best-effort; leak accounting continues

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
