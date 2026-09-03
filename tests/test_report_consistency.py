"""test_report_consistency.py — Sentinel consistency tests for reports and docs.

Verifies that numbers, tables, adapter counts, and benchmark metrics in
human-facing markdown documentation (README.md, reports/HF_MODEL_CARD.md,
reports/LORA_MODEL_ZOO.md, reports/DATASET_AND_ANALYTICS.md,
lora_adapters/SUMMARY.md) are strictly consistent with the machine-readable
sources of truth:
  - reports/lora_zoo_index.json (58 adapters hosted on Hugging Face Hub)
  - reports/pipeline_execution_stats.json (2,816,434 messages, 171,520 SFT dialogues, etc.)
  - lora_adapters/registry.json (56 local adapters cloned on disk)
  - reports/heuristic_benchmark_eval.json (50-scenario domain heuristic evaluation matrix)

This test suite directly catches:
  1. Historical '41 vs 58' adapter count drift between cards and registry.
  2. Dataset volume drift (clean messages, SFT dialogues, RAG chunks, DPO pairs).
  3. Benchmark metric discrepancies (heuristic scores, AST parse rates) against JSON truth.
  4. Re-emergence of ANY unauthorized percentage metrics in README's retracted benchmark section.
  5. Missing withdrawal disclaimers for academic benchmarks across documentation.
"""

from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_markdown_table_rows(content: str, min_cols: int = 3) -> list[list[str]]:
    """Extract rows from all markdown tables in content.

    Returns a list of cell lists (stripped, outer pipe delimiters omitted).
    Skips header separator lines like | :--- | :---: |.
    """
    rows: list[list[str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < min_cols:
            continue
        # Skip separator rows like | :--- | :--- |
        if all(re.match(r"^:?-+:?$", p) for p in parts):
            continue
        rows.append(parts)
    return rows


def parse_int_clean(val: str) -> int:
    """Extract the first integer from a string (e.g. '2,816,434 rows (189 MB)' -> 2816434)."""
    m = re.search(r"\d[\d,]*", val)
    if not m:
        raise ValueError(f"No digits found in: {val!r}")
    return int(m.group(0).replace(",", ""))


# --------------------------------------------------------------------------- #
# Consistency verification functions (pure logic, testable with mocks)
# --------------------------------------------------------------------------- #


def verify_hf_model_card(card_text: str, zoo_index: dict) -> None:
    """Verify reports/HF_MODEL_CARD.md against reports/lora_zoo_index.json."""
    total_expected = zoo_index["total_adapters"]
    flagships = set(zoo_index.get("flagships", []))
    flagship_count_expected = len(flagships)
    domain_count_expected = total_expected - flagship_count_expected

    # 1. Header count: e.g. "## 📚 Full Catalog (58 Adapters)"
    m_header = re.search(r"##\s*📚\s*Full\s+Catalog\s*\((\d+)\s+Adapters\)", card_text)
    assert m_header is not None, "HF_MODEL_CARD.md missing '## 📚 Full Catalog (<N> Adapters)' header"
    header_count = int(m_header.group(1))
    assert header_count == total_expected, (
        f"HF_MODEL_CARD.md header claims {header_count} adapters, expected {total_expected}"
    )

    # 2. Prose badge / intro: e.g. "**58 pre-trained adapters** (55 domain adapters + 3 flagship 7B–8B QLoRA)"
    m_prose = re.search(
        r"\*\*(\d+)\s+pre-trained\s+adapters\*\*\s*\((\d+)\s+domain\s+adapters\s*\+\s*(\d+)\s+flagship", card_text
    )
    assert m_prose is not None, (
        "HF_MODEL_CARD.md missing '**<N> pre-trained adapters** (<D> domain adapters + <F> flagship...)'"
    )
    p_total, p_domain, p_flagship = int(m_prose.group(1)), int(m_prose.group(2)), int(m_prose.group(3))
    assert p_total == total_expected, f"HF_MODEL_CARD.md prose total {p_total} does not match expected {total_expected}"
    assert p_domain == domain_count_expected, (
        f"HF_MODEL_CARD.md domain adapters count {p_domain} != expected {domain_count_expected}"
    )
    assert p_flagship == flagship_count_expected, (
        f"HF_MODEL_CARD.md flagship count {p_flagship} != expected {flagship_count_expected}"
    )

    # 3. Table row count and adapter slugs
    # Target table has columns: | # | Adapter Subfolder | Base Model | Hub Link |
    rows = parse_markdown_table_rows(card_text, min_cols=4)
    catalog_rows = [r for r in rows if r[0].isdigit()]
    assert len(catalog_rows) == total_expected, (
        f"HF_MODEL_CARD.md catalog table has {len(catalog_rows)} rows, expected {total_expected}"
    )

    # Verify adapter slugs and sequence
    expected_adapters = zoo_index["adapters"]
    for idx, (row, exp) in enumerate(zip(catalog_rows, expected_adapters, strict=True), start=1):
        row_idx = int(row[0])
        assert row_idx == idx, f"HF_MODEL_CARD.md row numbering mismatch: got {row_idx}, expected {idx}"
        slug = row[1].strip("`")
        assert slug == exp["id"], (
            f"HF_MODEL_CARD.md table row {idx} adapter slug mismatch: got {slug!r}, expected {exp['id']!r}"
        )
        base_model = row[2].strip("`")
        assert base_model == exp["base_model"], (
            f"HF_MODEL_CARD.md table row {idx} base model mismatch: got {base_model!r}, expected {exp['base_model']!r}"
        )


def verify_lora_model_zoo(zoo_text: str, zoo_index: dict) -> None:
    """Verify reports/LORA_MODEL_ZOO.md against reports/lora_zoo_index.json."""
    total_expected = zoo_index["total_adapters"]
    flagships = set(zoo_index.get("flagships", []))
    flagship_count_expected = len(flagships)
    domain_count_expected = total_expected - flagship_count_expected

    # 1. Header count: e.g. "## 📊 Полный каталог адаптеров (58 моделей)"
    m_header = re.search(r"##\s*📊\s*Полный\s+каталог\s+адаптеров\s*\((\d+)\s+моделей\)", zoo_text)
    assert m_header is not None, "LORA_MODEL_ZOO.md missing '## 📊 Полный каталог адаптеров (<N> моделей)' header"
    header_count = int(m_header.group(1))
    assert header_count == total_expected, (
        f"LORA_MODEL_ZOO.md header claims {header_count} models, expected {total_expected}"
    )

    # 2. Prose intro: e.g. "Официальный каталог **58 предварительно обученных LoRA-адаптеров** (55 доменных адаптеров + 3 флагманских QLoRA моделей 7B–8B)"
    m_prose = re.search(
        r"Официальный\s+каталог\s*\*\*(\d+)\s+предварительно\s+обученных\s+LoRA-адаптеров\*\*\s*\((\d+)\s+доменных\s+адаптеров\s*\+\s*(\d+)\s+флагманских",
        zoo_text,
    )
    assert m_prose is not None, "LORA_MODEL_ZOO.md missing intro adapter count pattern"
    p_total, p_domain, p_flagship = int(m_prose.group(1)), int(m_prose.group(2)), int(m_prose.group(3))
    assert p_total == total_expected, (
        f"LORA_MODEL_ZOO.md prose total {p_total} does not match expected {total_expected}"
    )
    assert p_domain == domain_count_expected, (
        f"LORA_MODEL_ZOO.md domain count {p_domain} != expected {domain_count_expected}"
    )
    assert p_flagship == flagship_count_expected, (
        f"LORA_MODEL_ZOO.md flagship count {p_flagship} != expected {flagship_count_expected}"
    )

    # 3. Table rows: columns | # | Идентификатор адаптера | Базовая модель | Локальный каталог | Hugging Face Hub |
    rows = parse_markdown_table_rows(zoo_text, min_cols=5)
    catalog_rows = [r for r in rows if r[0].isdigit()]
    assert len(catalog_rows) == total_expected, (
        f"LORA_MODEL_ZOO.md catalog table has {len(catalog_rows)} rows, expected {total_expected}"
    )

    expected_adapters = zoo_index["adapters"]
    for idx, (row, exp) in enumerate(zip(catalog_rows, expected_adapters, strict=True), start=1):
        row_idx = int(row[0])
        assert row_idx == idx, f"LORA_MODEL_ZOO.md row numbering mismatch: got {row_idx}, expected {idx}"
        slug = row[1].strip("`")
        assert slug == exp["id"], (
            f"LORA_MODEL_ZOO.md table row {idx} adapter slug mismatch: got {slug!r}, expected {exp['id']!r}"
        )
        base_model = row[2].strip("`")
        assert base_model == exp["base_model"], (
            f"LORA_MODEL_ZOO.md table row {idx} base model mismatch: got {base_model!r}, expected {exp['base_model']!r}"
        )


def verify_dataset_and_analytics(analytics_text: str, stats: dict, zoo_index: dict) -> None:
    """Verify reports/DATASET_AND_ANALYTICS.md against pipeline stats and zoo index."""
    # 1. LoRA Zoo mention in section "Empirical Evaluation — Honest Status"
    # "Pre-trained adapters for **58** base models are available in the [LoRA Model Zoo]"
    m_zoo = re.search(r"Pre-trained\s+adapters\s+for\s+\*\*(\d+)\*\*\s+base\s+models\s+are\s+available", analytics_text)
    assert m_zoo is not None, "DATASET_AND_ANALYTICS.md missing 'Pre-trained adapters for **<N>** base models'"
    zoo_count = int(m_zoo.group(1))
    assert zoo_count == zoo_index["total_adapters"], (
        f"DATASET_AND_ANALYTICS.md mentions {zoo_count} adapters, expected {zoo_index['total_adapters']}"
    )

    # 2. Key Corpus Metrics table
    rows = parse_markdown_table_rows(analytics_text, min_cols=3)
    metrics_map: dict[str, int] = {}
    for r in rows:
        metric_name = r[0].replace("*", "").strip()
        val_str = r[1].replace("`", "").replace(",", "").strip()
        if val_str.isdigit():
            metrics_map[metric_name] = int(val_str)

    expected_clean = stats["cleaned_messages_count"]
    expected_sft = stats["sft_dialogues_count"]
    expected_rag = stats["rag_chunks_count"]
    expected_dpo = stats["dpo_pairs_count"]

    assert metrics_map.get("Clean Messages") == expected_clean, (
        f"Clean Messages in Key Metrics table: {metrics_map.get('Clean Messages')}, expected {expected_clean}"
    )
    assert metrics_map.get("SFT Dialogues") == expected_sft, (
        f"SFT Dialogues in Key Metrics table: {metrics_map.get('SFT Dialogues')}, expected {expected_sft}"
    )
    assert metrics_map.get("RAG Knowledge Chunks") == expected_rag, (
        f"RAG Knowledge Chunks in Key Metrics table: {metrics_map.get('RAG Knowledge Chunks')}, expected {expected_rag}"
    )
    assert metrics_map.get("DPO Preference Pairs") == expected_dpo, (
        f"DPO Preference Pairs in Key Metrics table: {metrics_map.get('DPO Preference Pairs')}, expected {expected_dpo}"
    )


def verify_readme_metrics(readme_text: str, stats: dict, local_registry: dict, zoo_index: dict) -> None:
    """Verify README.md numbers against pipeline execution stats, local registry, and Hub zoo index."""
    clean_cnt = stats["cleaned_messages_count"]
    sft_cnt = stats["sft_dialogues_count"]
    rag_cnt = stats["rag_chunks_count"]
    dpo_cnt = stats["dpo_pairs_count"]
    nodes_cnt = stats.get("pii_stats", {}).get("community_names_anonymized", 11)
    local_adapters_cnt = local_registry["adapter_count"]
    hub_adapters_cnt = zoo_index["total_adapters"]

    # 1. Local Datasets and Formats table
    rows = parse_markdown_table_rows(readme_text, min_cols=3)
    file_to_volume: dict[str, str] = {}
    for r in rows:
        file_path = r[0].strip("` ")
        if file_path.startswith("dataset_output/"):
            file_to_volume[file_path] = r[2]

    # Verify full_clean_messages.parquet volume
    v_clean = file_to_volume.get("dataset_output/parquet/full_clean_messages.parquet", "")
    assert parse_int_clean(v_clean) == clean_cnt, (
        f"README.md table full_clean_messages.parquet has {v_clean!r}, expected {clean_cnt}"
    )

    # Verify sft_dialogues.parquet volume
    v_sft = file_to_volume.get("dataset_output/parquet/sft_dialogues.parquet", "")
    assert parse_int_clean(v_sft) == sft_cnt, f"README.md table sft_dialogues.parquet has {v_sft!r}, expected {sft_cnt}"

    # Verify rag_knowledge_base.parquet volume
    v_rag = file_to_volume.get("dataset_output/parquet/rag_knowledge_base.parquet", "")
    assert parse_int_clean(v_rag) == rag_cnt, (
        f"README.md table rag_knowledge_base.parquet has {v_rag!r}, expected {rag_cnt}"
    )

    # Verify sft_openai_messages.jsonl volume
    v_openai = file_to_volume.get("dataset_output/jsonl/sft_openai_messages.jsonl", "")
    assert parse_int_clean(v_openai) == sft_cnt, (
        f"README.md table sft_openai_messages.jsonl has {v_openai!r}, expected {sft_cnt}"
    )

    # Verify sft_sharegpt_format.jsonl volume
    v_sharegpt = file_to_volume.get("dataset_output/jsonl/sft_sharegpt_format.jsonl", "")
    assert parse_int_clean(v_sharegpt) == sft_cnt, (
        f"README.md table sft_sharegpt_format.jsonl has {v_sharegpt!r}, expected {sft_cnt}"
    )

    # Verify rag_chunks_kb.jsonl volume
    v_rag_jsonl = file_to_volume.get("dataset_output/jsonl/rag_chunks_kb.jsonl", "")
    assert parse_int_clean(v_rag_jsonl) == rag_cnt, (
        f"README.md table rag_chunks_kb.jsonl has {v_rag_jsonl!r}, expected {rag_cnt}"
    )

    # Verify dpo_preference_pairs.jsonl volume
    v_dpo = file_to_volume.get("dataset_output/jsonl/dpo_preference_pairs.jsonl", "")
    assert parse_int_clean(v_dpo) == dpo_cnt, (
        f"README.md table dpo_preference_pairs.jsonl has {v_dpo!r}, expected {dpo_cnt}"
    )

    # 2. Overview table check (11 community nodes, 171,520 curated dialogues)
    overview_rows = [r for r in rows if any("community nodes anonymized" in c for c in r)]
    assert overview_rows, "README.md overview table missing community nodes anonymized row"
    assert f"{nodes_cnt} community nodes anonymized" in overview_rows[0][2]

    dialogue_rows = [r for r in rows if any("curated dialogues" in c for c in r)]
    assert dialogue_rows, "README.md overview table missing curated dialogues row"
    assert parse_int_clean(dialogue_rows[0][2]) == sft_cnt

    # 3. LoRA Model Zoo adapter counts in README.md:
    # Must explicitly state both 58 (on Hub) and 56 (locally on disk), matching respective sources of truth.
    m_lora_tip = re.search(
        r"LoRA\s+Model\s+Zoo.*?(\d+)\s+adapters\s+on\s+Hugging\s+Face\s+Hub.*?(\d+)\s+adapters\s+cloned\s+locally",
        readme_text,
    )
    assert m_lora_tip is not None, "README.md tip missing dual Hub / local adapter counts"
    assert int(m_lora_tip.group(1)) == hub_adapters_cnt, (
        f"README.md tip Hub count {m_lora_tip.group(1)} != expected {hub_adapters_cnt}"
    )
    assert int(m_lora_tip.group(2)) == local_adapters_cnt, (
        f"README.md tip local count {m_lora_tip.group(2)} != expected {local_adapters_cnt}"
    )

    m_lora_hub = re.search(
        r"Model\s+Hub\s*\((\d+)\s+LoRA\s+Adapters\s+on\s+Hub\s*/\s*(\d+)\s+Local\s+Adapters\)", readme_text
    )
    assert m_lora_hub is not None, "README.md missing Model Hub dual adapter count heading"
    assert int(m_lora_hub.group(1)) == hub_adapters_cnt, (
        f"README.md Model Hub claims {m_lora_hub.group(1)} on Hub, expected {hub_adapters_cnt}"
    )
    assert int(m_lora_hub.group(2)) == local_adapters_cnt, (
        f"README.md Model Hub claims {m_lora_hub.group(2)} local, expected {local_adapters_cnt}"
    )

    m_lora_sec = re.search(
        r"###\s*LoRA\s+Model\s+Zoo\s*\((\d+)\s+Local\s+Adapters\s*/\s*(\d+)\s+Adapters\s+on\s+Hugging\s+Face\s+Hub\)",
        readme_text,
    )
    assert m_lora_sec is not None, "README.md section missing dual adapter count"
    assert int(m_lora_sec.group(1)) == local_adapters_cnt, (
        f"README.md section claims {m_lora_sec.group(1)} local, expected {local_adapters_cnt}"
    )
    assert int(m_lora_sec.group(2)) == hub_adapters_cnt, (
        f"README.md section claims {m_lora_sec.group(2)} on Hub, expected {hub_adapters_cnt}"
    )

    m_tree = re.search(
        r"LORA_MODEL_ZOO\.md\s+#\s+Catalog\s+of\s+(\d+)\s+LoRA\s+Adapters\s+on\s+Hub\s*\((\d+)\s+local\)",
        readme_text,
    )
    assert m_tree is not None, "README.md repo tree missing dual adapter count comment"
    assert int(m_tree.group(1)) == hub_adapters_cnt, (
        f"README.md tree claims {m_tree.group(1)} on Hub, expected {hub_adapters_cnt}"
    )
    assert int(m_tree.group(2)) == local_adapters_cnt, (
        f"README.md tree claims {m_tree.group(2)} local, expected {local_adapters_cnt}"
    )


def verify_benchmark_metrics_and_withdrawal_status(
    analytics_text: str,
    readme_text: str,
    hf_card_text: str,
    zoo_text: str,
    benchmark_eval: dict,
) -> None:
    """Verify empirical benchmark percentage/heuristic metrics and withdrawal status consistency.

    Guards against:
      1. Drift between reports/heuristic_benchmark_eval.json and reports/DATASET_AND_ANALYTICS.md.
      2. Re-emergence of ANY unauthorized percentage metrics in README's retracted benchmark section.
      3. Silent omission of the benchmark WITHDRAWN notice across README and documentation.
    """
    expected_setups = benchmark_eval["setups"]

    # 1. Parse heuristic evaluation table in reports/DATASET_AND_ANALYTICS.md
    rows = parse_markdown_table_rows(analytics_text, min_cols=3)
    bench_rows = [r for r in rows if any(k in r[0] for k in expected_setups)]
    assert len(bench_rows) == len(expected_setups), (
        f"Expected {len(expected_setups)} benchmark rows in DATASET_AND_ANALYTICS.md, found {len(bench_rows)}"
    )

    for r in bench_rows:
        matched_key = [k for k in expected_setups if k in r[0]][0]
        exp = expected_setups[matched_key]
        score = float(r[1].replace("*", "").strip())
        ast_rate = float(r[2].replace("*", "").replace("%", "").strip())
        assert score == exp["heuristic_score"], (
            f"{matched_key} heuristic score mismatch in DATASET_AND_ANALYTICS.md: got {score}, expected {exp['heuristic_score']}"
        )
        assert ast_rate == exp["ast_parse_rate"], (
            f"{matched_key} AST rate mismatch in DATASET_AND_ANALYTICS.md: got {ast_rate}%, expected {exp['ast_parse_rate']}%"
        )

    # 2. Check withdrawal disclaimers across all 4 documents
    assert (
        "Academic metrics (HumanEval / RuMMLU / PPL / ROUGE) published earlier have been WITHDRAWN" in analytics_text
    ), "DATASET_AND_ANALYTICS.md missing official academic metrics withdrawal callout"

    assert "Benchmark section withdrawn from README" in readme_text, (
        "README.md missing 'Benchmark section withdrawn from README' warning notice"
    )

    assert re.search(r"Academic benchmark numbers.*?are \*\*withdrawn pending re-evaluation\*\*", hf_card_text, re.S), (
        "HF_MODEL_CARD.md missing withdrawal notice"
    )

    assert re.search(r"академические метрики.*?отозваны до переоценки", zoo_text), (
        "LORA_MODEL_ZOO.md missing withdrawal notice"
    )

    # 3. Generalized Anti-Inflation Guard:
    # Verifies that NO percentage scores (e.g. 96.4%, 91.2%, 100%, pass@1) appear anywhere in
    # the Comparative Architectural Evaluation section of README.md, as all capability metrics
    # in this section were officially retracted and marked WITHDRAWN pending a fresh GPU re-run.
    m_bench_sec = re.search(
        r"## Comparative Architectural Evaluation.*?(?=## Quick start)",
        readme_text,
        re.DOTALL,
    )
    assert m_bench_sec is not None, "README.md missing 'Comparative Architectural Evaluation' section"
    bench_section_text = m_bench_sec.group(0)
    percentages_in_bench = re.findall(r"\b\d+(?:\.\d+)?%", bench_section_text)
    assert not percentages_in_bench, (
        f"README.md benchmark section contains unauthorized/retracted percentage metrics: {percentages_in_bench}. "
        "All benchmark metrics in this section are WITHDRAWN pending fresh GPU re-run."
    )


def verify_local_registry_and_summary(registry: dict, summary_text: str, lora_dir: Path) -> None:
    """Verify lora_adapters/registry.json, lora_adapters/SUMMARY.md, and on-disk adapter dirs."""
    adapter_count = registry["adapter_count"]
    adapters = registry["adapters"]
    assert len(adapters) == adapter_count, (
        f"registry.json claims {adapter_count} adapters but lists {len(adapters)} entries"
    )

    # 1. On-disk adapter directories with adapter_config.json
    on_disk_adapters = [d.name for d in lora_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").is_file()]
    assert len(on_disk_adapters) == adapter_count, (
        f"Found {len(on_disk_adapters)} adapter dirs on disk, expected {adapter_count}"
    )
    slugs_in_registry = {a["slug"] for a in adapters}
    assert slugs_in_registry == set(on_disk_adapters), (
        f"Disk vs registry mismatch: diff = {set(on_disk_adapters) ^ slugs_in_registry}"
    )

    # 2. SUMMARY.md header count
    m_total = re.search(r"\*\*Total adapters:\*\*\s*(\d+)", summary_text)
    assert m_total is not None, "SUMMARY.md missing '**Total adapters:** <N>'"
    assert int(m_total.group(1)) == adapter_count, (
        f"SUMMARY.md claims {m_total.group(1)} adapters, expected {adapter_count}"
    )

    # 3. SUMMARY.md table rows count
    summary_rows = parse_markdown_table_rows(summary_text, min_cols=8)
    numbered_rows = [r for r in summary_rows if r[0].isdigit()]
    assert len(numbered_rows) == adapter_count, (
        f"SUMMARY.md table has {len(numbered_rows)} rows, expected {adapter_count}"
    )


# --------------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------------- #


class TestReportConsistency(unittest.TestCase):
    """End-to-end consistency tests verifying actual repository files."""

    @classmethod
    def setUpClass(cls):
        cls.zoo_index_path = REPO_ROOT / "reports" / "lora_zoo_index.json"
        cls.stats_path = REPO_ROOT / "reports" / "pipeline_execution_stats.json"
        cls.registry_path = REPO_ROOT / "lora_adapters" / "registry.json"
        cls.benchmark_eval_path = REPO_ROOT / "reports" / "heuristic_benchmark_eval.json"
        cls.lora_dir = REPO_ROOT / "lora_adapters"

        cls.hf_card_path = REPO_ROOT / "reports" / "HF_MODEL_CARD.md"
        cls.zoo_md_path = REPO_ROOT / "reports" / "LORA_MODEL_ZOO.md"
        cls.analytics_path = REPO_ROOT / "reports" / "DATASET_AND_ANALYTICS.md"
        cls.readme_path = REPO_ROOT / "README.md"
        cls.summary_path = REPO_ROOT / "lora_adapters" / "SUMMARY.md"

        with cls.zoo_index_path.open(encoding="utf-8") as f:
            cls.zoo_index = json.load(f)
        with cls.stats_path.open(encoding="utf-8") as f:
            cls.stats = json.load(f)
        with cls.registry_path.open(encoding="utf-8") as f:
            cls.registry = json.load(f)
        with cls.benchmark_eval_path.open(encoding="utf-8") as f:
            cls.benchmark_eval = json.load(f)

        cls.hf_card_text = cls.hf_card_path.read_text(encoding="utf-8")
        cls.zoo_md_text = cls.zoo_md_path.read_text(encoding="utf-8")
        cls.analytics_text = cls.analytics_path.read_text(encoding="utf-8")
        cls.readme_text = cls.readme_path.read_text(encoding="utf-8")
        cls.summary_text = cls.summary_path.read_text(encoding="utf-8")

    def test_hf_model_card_consistency(self):
        """reports/HF_MODEL_CARD.md matches reports/lora_zoo_index.json (58 adapters, rows, slugs)."""
        verify_hf_model_card(self.hf_card_text, self.zoo_index)

    def test_lora_model_zoo_consistency(self):
        """reports/LORA_MODEL_ZOO.md matches reports/lora_zoo_index.json (58 adapters, rows, slugs)."""
        verify_lora_model_zoo(self.zoo_md_text, self.zoo_index)

    def test_dataset_and_analytics_consistency(self):
        """reports/DATASET_AND_ANALYTICS.md matches pipeline_execution_stats.json & zoo index."""
        verify_dataset_and_analytics(self.analytics_text, self.stats, self.zoo_index)

    def test_readme_metrics_consistency(self):
        """README.md dataset numbers and LoRA counts match execution stats, registry, and zoo index."""
        verify_readme_metrics(self.readme_text, self.stats, self.registry, self.zoo_index)

    def test_benchmark_metrics_and_withdrawal_consistency(self):
        """Check heuristic benchmark table matches heuristic_benchmark_eval.json, with withdrawal notices."""
        verify_benchmark_metrics_and_withdrawal_status(
            self.analytics_text,
            self.readme_text,
            self.hf_card_text,
            self.zoo_md_text,
            self.benchmark_eval,
        )

    def test_local_registry_and_summary_consistency(self):
        """lora_adapters/registry.json matches on-disk dirs and lora_adapters/SUMMARY.md (56 adapters)."""
        verify_local_registry_and_summary(self.registry, self.summary_text, self.lora_dir)

    # ----------------------------------------------------------------------- #
    # Negative test suite: proof that tampered files strictly fail assertions
    # ----------------------------------------------------------------------- #

    def test_negative_tampered_hf_card_count_fails(self):
        """Demonstrates that an out-of-sync count (like the historical '41' bug) fails immediately."""
        # Mutate 58 -> 41 in header
        tampered = re.sub(r"\(58 Adapters\)", "(41 Adapters)", self.hf_card_text)
        with self.assertRaises(AssertionError) as ctx:
            verify_hf_model_card(tampered, self.zoo_index)
        self.assertIn("header claims 41 adapters, expected 58", str(ctx.exception))

    def test_negative_tampered_hf_card_rows_fails(self):
        """Demonstrates that dropping or altering a row in HF_MODEL_CARD.md fails."""
        # Cut off the last row
        lines = self.hf_card_text.splitlines()
        truncated_lines = [line for line in lines if not line.startswith("| 58 |")]
        tampered = "\n".join(truncated_lines)
        with self.assertRaises(AssertionError) as ctx:
            verify_hf_model_card(tampered, self.zoo_index)
        self.assertIn("catalog table has 57 rows, expected 58", str(ctx.exception))

    def test_negative_tampered_zoo_md_count_fails(self):
        """Demonstrates that altering adapter count in LORA_MODEL_ZOO.md fails."""
        tampered = re.sub(r"\(58 моделей\)", "(59 моделей)", self.zoo_md_text)
        with self.assertRaises(AssertionError) as ctx:
            verify_lora_model_zoo(tampered, self.zoo_index)
        self.assertIn("header claims 59 models, expected 58", str(ctx.exception))

    def test_negative_tampered_readme_number_fails(self):
        """Demonstrates that altering any single number in README.md tables fails."""
        # Mutate cleaned messages count 2,816,434 -> 2,816,435
        tampered = self.readme_text.replace("2,816,434 rows", "2,816,435 rows")
        with self.assertRaises(AssertionError) as ctx:
            verify_readme_metrics(tampered, self.stats, self.registry, self.zoo_index)
        self.assertIn("expected 2816434", str(ctx.exception))

    def test_negative_tampered_readme_hub_adapter_count_fails(self):
        """Demonstrates that altering Hub adapter count in README.md (58 -> 59) fails."""
        tampered = self.readme_text.replace("58 adapters on Hugging Face Hub", "59 adapters on Hugging Face Hub")
        with self.assertRaises(AssertionError) as ctx:
            verify_readme_metrics(tampered, self.stats, self.registry, self.zoo_index)
        self.assertIn("expected 58", str(ctx.exception))

    def test_negative_tampered_readme_local_adapter_count_fails(self):
        """Demonstrates that altering local adapter count in README.md (56 -> 57) fails."""
        tampered = self.readme_text.replace("56 adapters cloned locally", "57 adapters cloned locally")
        with self.assertRaises(AssertionError) as ctx:
            verify_readme_metrics(tampered, self.stats, self.registry, self.zoo_index)
        self.assertIn("expected 56", str(ctx.exception))

    def test_negative_tampered_heuristic_score_fails(self):
        """Demonstrates that altering heuristic benchmark score (34.5 -> 96.4) in markdown fails."""
        tampered = self.analytics_text.replace(
            "| Domain LoRA (171.5k dialogues) | 34.5 | 72.2% |",
            "| Domain LoRA (171.5k dialogues) | 96.4 | 72.2% |",
        )
        with self.assertRaises(AssertionError) as ctx:
            verify_benchmark_metrics_and_withdrawal_status(
                tampered,
                self.readme_text,
                self.hf_card_text,
                self.zoo_md_text,
                self.benchmark_eval,
            )
        self.assertIn("expected 34.5", str(ctx.exception))

    def test_negative_tampered_ast_rate_fails(self):
        """Demonstrates that altering AST parse rate (72.2% -> 100.0%) in markdown fails."""
        tampered = self.analytics_text.replace(
            "| Domain LoRA (171.5k dialogues) | 34.5 | 72.2% |",
            "| Domain LoRA (171.5k dialogues) | 34.5 | 100.0% |",
        )
        with self.assertRaises(AssertionError) as ctx:
            verify_benchmark_metrics_and_withdrawal_status(
                tampered,
                self.readme_text,
                self.hf_card_text,
                self.zoo_md_text,
                self.benchmark_eval,
            )
        self.assertIn("expected 72.2%", str(ctx.exception))

    def test_negative_tampered_benchmark_json_fails(self):
        """Demonstrates that if the JSON source of truth is updated but docs are not, test fails."""
        tampered_json = copy.deepcopy(self.benchmark_eval)
        tampered_json["setups"]["Domain LoRA (171.5k dialogues)"]["heuristic_score"] = 99.9
        with self.assertRaises(AssertionError) as ctx:
            verify_benchmark_metrics_and_withdrawal_status(
                self.analytics_text,
                self.readme_text,
                self.hf_card_text,
                self.zoo_md_text,
                tampered_json,
            )
        self.assertIn("expected 99.9", str(ctx.exception))

    def test_negative_tampered_readme_inflated_metric_fails(self):
        """Demonstrates that sneaking ANY percentage metric (e.g. 91.2% or 96.4%) into README fails."""
        tampered = self.readme_text.replace(
            "## Comparative Architectural Evaluation (Base vs RAG vs LoRA vs Hybrid)\n\n> [!WARNING]",
            "## Comparative Architectural Evaluation (Base vs RAG vs LoRA vs Hybrid)\n\n"
            "Preliminary pass rate: **91.2%**\n\n> [!WARNING]",
        )
        with self.assertRaises(AssertionError) as ctx:
            verify_benchmark_metrics_and_withdrawal_status(
                self.analytics_text,
                tampered,
                self.hf_card_text,
                self.zoo_md_text,
                self.benchmark_eval,
            )
        self.assertIn("unauthorized/retracted percentage metrics: ['91.2%']", str(ctx.exception))

    def test_negative_missing_withdrawal_notice_fails(self):
        """Demonstrates that removing the WITHDRAWN warning notice from README fails."""
        tampered = self.readme_text.replace(
            "Benchmark section withdrawn from README", "Benchmark section active in README"
        )
        with self.assertRaises(AssertionError) as ctx:
            verify_benchmark_metrics_and_withdrawal_status(
                self.analytics_text,
                tampered,
                self.hf_card_text,
                self.zoo_md_text,
                self.benchmark_eval,
            )
        self.assertIn("Benchmark section withdrawn from README", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
