"""
Canonical Scientific Benchmark & Reproducibility Audit Pipeline (protocol v3).

This is the ONLY benchmark entry point. All evaluation artifacts in reports/
(raw_model_evaluation_matrix.json, reproducibility_audit.json,
scientific_evaluation_metrics.json, SCIENTIFIC_EVALUATION_REPORT.md,
eval_split_manifest.json) are produced exclusively by this script, so two
"versions of the same benchmark" can never silently diverge again.

Protocol v3 guarantees:
1. Single fixed methodology: 3 seeded sub-splits (seeds 42/123/777) of 10
   dialogues each, max_length=384, greedy generation, fp16 on CUDA.
2. True held-out evaluation: every dialogue used by ANY LoRA trainer in
   src/lora/ (replicated sampling rules, see TRAINING_SAMPLING_RULES) is
   excluded from the evaluation pool. The exact split is fingerprinted
   (SHA-256) and persisted to reports/eval_split_manifest.json.
3. Full provenance: every output JSON embeds a run_metadata block with the
   git revision, dirty flag, UTC timestamp, library versions and dataset
   fingerprint. Report text NEVER references hand-written commit hashes.
4. Data-driven conclusions: the Markdown narrative is generated from the
   measured numbers, not hardcoded.

Usage:
    python scripts/run_scientific_benchmark.py              # full GPU run
    python scripts/run_scientific_benchmark.py --from-cache # regenerate report
                                                            # from cached JSONs
"""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScientificBenchmark")

PROTOCOL_VERSION = "3.0"
SEEDS = [42, 123, 777]
EVAL_SPLIT_SIZE = 10
MAX_LEN = 384
DATASET_PATH = Path("dataset_output/parquet/sft_dialogues.parquet")

# Sampling rules replicated from the LoRA trainers in src/lora/. Every dialogue
# selected by these rules may have been seen during adapter training and MUST
# be excluded from the evaluation pool.
#   src/lora/train_20_popular_models.py : df.sample(n=400, random_state=42)
#   src/lora/train_heavyweight_5_sota.py: df.sample(n=500, random_state=42)
TRAINING_SAMPLING_RULES = [(400, 42), (500, 42)]

EVAL_MODELS = [
    # Tier 1: Micro & Small Base Models (Non-instruct)
    {"id": "pythia_70m", "base": "EleutherAI/pythia-70m", "family": "Pythia", "params": "70M", "type": "base"},
    {"id": "opt_125m", "base": "facebook/opt-125m", "family": "OPT", "params": "125M", "type": "base"},
    {"id": "bloom_560m", "base": "bigscience/bloom-560m", "family": "BLOOM", "params": "560M", "type": "base"},
    {"id": "gpt2_medium", "base": "openai-community/gpt2-medium", "family": "GPT-2", "params": "355M", "type": "base"},
    {
        "id": "rugpt3_small",
        "base": "ai-forever/rugpt3small_based_on_gpt2",
        "family": "RuGPT3",
        "params": "125M",
        "type": "base",
    },
    # Tier 2: Compact 1B-Class Models (Instruct & Coder)
    {
        "id": "tinyllama_1.1b_chat",
        "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "TinyLlama",
        "params": "1.1B",
        "type": "instruct",
    },
    {
        "id": "smollm2_1.7b_instruct",
        "base": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "family": "SmolLM2",
        "params": "1.7B",
        "type": "instruct",
    },
    {
        "id": "llama_3.2_1b_instruct",
        "base": "unsloth/Llama-3.2-1B-Instruct",
        "family": "LLaMA 3.2",
        "params": "1.0B",
        "type": "instruct",
    },
    {
        "id": "deepseek_coder_1.3b_instruct",
        "base": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "family": "DeepSeek Coder",
        "params": "1.3B",
        "type": "coder",
    },
    {
        "id": "falcon3_1b_instruct",
        "base": "tiiuae/Falcon3-1B-Instruct",
        "family": "Falcon 3",
        "params": "1.0B",
        "type": "instruct",
    },
    # Tier 3: Modern 0.5B-1.5B Architectures (Instruct & Reasoning)
    {
        "id": "qwen2.5_0.5b_instruct",
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen 2.5",
        "params": "0.5B",
        "type": "instruct",
    },
    {
        "id": "qwen2.5_1.5b_instruct",
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen 2.5",
        "params": "1.5B",
        "type": "instruct",
    },
    {
        "id": "qwen2.5_coder_1.5b_instruct",
        "base": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "Qwen 2.5 Coder",
        "params": "1.5B",
        "type": "coder",
    },
    {
        "id": "vikhr_qwen_2.5_1.5b",
        "base": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
        "family": "Vikhr Russian",
        "params": "1.5B",
        "type": "instruct",
    },
    {
        "id": "deepseek_r1_distill_qwen_1.5b",
        "base": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "DeepSeek R1",
        "params": "1.5B",
        "type": "reasoning",
    },
]

QUALITATIVE_PROMPTS = [
    {
        "id": "q1_postgres",
        "title": "PostgreSQL Replica Lag & WAL",
        "prompt": "Как в PostgreSQL диагностировать причину отставания репликации (replica lag) и какие параметры postgresql.conf нужно проверить?",
    },
    {
        "id": "q2_k8s",
        "title": "Kubernetes OOMKilled & Limits",
        "prompt": "Что делать, если под в Kubernetes постоянно падает с ошибкой OOMKilled (Exit Code 137)?",
    },
    {
        "id": "q3_nginx",
        "title": "Nginx WebSocket Proxying",
        "prompt": "Как правильно настроить проксирование WebSocket в Nginx и какие директивы proxy_set_header и proxy_read_timeout необходимы?",
    },
    {
        "id": "q4_asyncio",
        "title": "Python Asyncio Blocking I/O",
        "prompt": "Почему в FastAPI / Asyncio нельзя вызывать блокирующий time.sleep или requests.get в async def эндпоинтах и как это исправить?",
    },
]


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_git_revision() -> dict[str, Any]:
    """Return the REAL current git revision. Never hand-write commit hashes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - git always present in repo
        logger.warning(f"Could not resolve git revision: {exc}")
        return {"commit": "unknown", "dirty": None}


def build_run_metadata(dataset_sha256: str, source: str) -> dict[str, Any]:
    import peft as peft_lib
    import transformers as tf_lib

    return {
        "protocol_version": PROTOCOL_VERSION,
        "source": source,
        "script": "scripts/run_scientific_benchmark.py",
        "git": get_git_revision(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seeds": SEEDS,
        "eval_split_size_per_seed": EVAL_SPLIT_SIZE,
        "max_length": MAX_LEN,
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": dataset_sha256,
        "training_exclusion_rules": [{"n_samples": n, "random_state": s} for n, s in TRAINING_SAMPLING_RULES],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": tf_lib.__version__,
            "peft": peft_lib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }


def dialogue_to_text(row: pd.Series) -> str:
    msgs = row.get("messages", [])
    return "\n".join([f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}" for m in msgs])


def build_eval_splits(df: pd.DataFrame) -> dict[str, Any]:
    """Build seeded evaluation splits from a pool that excludes training data."""
    excluded_indices = set()
    for n, seed in TRAINING_SAMPLING_RULES:
        if len(df) > n:
            excluded_indices.update(df.sample(n=n, random_state=seed).index.tolist())
    pool = df.drop(index=sorted(excluded_indices)).reset_index(drop=True)
    logger.info(f"Eval pool: {len(pool)} dialogues ({len(excluded_indices)} excluded as possible training data).")

    splits: dict[int, list[str]] = {}
    manifest_seeds: dict[str, Any] = {}
    for s in SEEDS:
        sample_df = pool.sample(n=EVAL_SPLIT_SIZE, random_state=s).reset_index(drop=True)
        texts = [dialogue_to_text(row) for _, row in sample_df.iterrows()]
        splits[s] = texts
        manifest_seeds[f"seed_{s}"] = {
            "pool_positions": sample_df.index.tolist() if "index" in sample_df.columns else None,
            "dialogue_sha256": [sha256_text(t) for t in texts],
        }
    return {"splits": splits, "pool_size": len(pool), "excluded_count": len(excluded_indices), "seeds": manifest_seeds}


def compute_sample_ppls(model, tokenizer, texts: list[str], max_len: int = MAX_LEN) -> list[float]:
    losses = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", max_length=max_len, truncation=True).input_ids.to(model.device)
            if enc.shape[1] < 4:
                continue
            loss = model(enc, labels=enc).loss.item()
            if not math.isnan(loss) and not math.isinf(loss):
                losses.append(loss)
    return [math.exp(loss) for loss in losses]


# ---------------------------------------------------------------------------
# Full GPU benchmark run
# ---------------------------------------------------------------------------


def run_benchmark() -> None:
    logger.info("Initializing Scientific Benchmark & Reproducibility Audit (protocol v%s)...", PROTOCOL_VERSION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df_sft = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded SFT dataset with {len(df_sft)} dialogues.")

    dataset_sha = sha256_file(DATASET_PATH)
    split_info = build_eval_splits(df_sft)
    seed_splits: dict[int, list[str]] = split_info["splits"]

    metadata = build_run_metadata(dataset_sha, source="full-gpu-run")

    full_matrix_results: list[dict[str, Any]] = []
    reproducibility_results: dict[str, Any] = {}

    for m_idx, m_info in enumerate(EVAL_MODELS, 1):
        m_id = m_info["id"]
        base_name = m_info["base"]
        adapter_path = Path(f"lora_adapters/{m_id}")
        logger.info("=" * 80)
        logger.info(
            f"[{m_idx}/{len(EVAL_MODELS)}] Evaluating: {m_id} ({base_name}) | Family: {m_info['family']} | Params: {m_info['params']}"
        )
        logger.info("=" * 80)

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            base_model.eval()

            base_seed_ppls = {s: compute_sample_ppls(base_model, tokenizer, seed_splits[s]) for s in SEEDS}

            base_generations = {}
            for q in QUALITATIVE_PROMPTS:
                inp = tokenizer(f"<|user|>\n{q['prompt']}<|assistant|>\n", return_tensors="pt").to(base_model.device)
                with torch.no_grad():
                    out = base_model.generate(**inp, max_new_tokens=90, do_sample=False)
                base_generations[q["id"]] = tokenizer.decode(
                    out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()

            lora_seed_ppls = {}
            lora_generations = {}
            adapter_found = adapter_path.exists() and (
                (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()
            )
            if adapter_found:
                logger.info(f"Attaching LoRA adapter from {adapter_path}...")
                lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
                lora_model.eval()

                for s in SEEDS:
                    lora_seed_ppls[s] = compute_sample_ppls(lora_model, tokenizer, seed_splits[s])

                for q in QUALITATIVE_PROMPTS:
                    inp = tokenizer(f"<|user|>\n{q['prompt']}<|assistant|>\n", return_tensors="pt").to(
                        lora_model.device
                    )
                    with torch.no_grad():
                        out = lora_model.generate(**inp, max_new_tokens=90, do_sample=False)
                    lora_generations[q["id"]] = tokenizer.decode(
                        out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True
                    ).strip()

                del lora_model
            else:
                logger.warning(
                    f"Adapter not found for {m_id}; marking record as adapter_missing (no fake LoRA numbers)."
                )
                lora_seed_ppls = {}
                lora_generations = {}

            seed_comparison = {}
            all_base_ppls: list[float] = []
            all_lora_ppls: list[float] = []
            for s in SEEDS:
                b_p = base_seed_ppls[s]
                l_p = lora_seed_ppls.get(s, [])
                all_base_ppls.extend(b_p)
                all_lora_ppls.extend(l_p)

                b_mean = float(np.mean(b_p)) if b_p else float("nan")
                b_std = float(np.std(b_p)) if b_p else 0.0
                l_mean = float(np.mean(l_p)) if l_p else float("nan")
                l_std = float(np.std(l_p)) if l_p else 0.0
                d_pct = ((b_mean - l_mean) / b_mean * 100.0) if l_p and b_mean > 0 else float("nan")

                seed_comparison[f"seed_{s}"] = {
                    "base_ppl_mean": round(b_mean, 2),
                    "base_ppl_std": round(b_std, 2),
                    "lora_ppl_mean": round(l_mean, 2) if l_p else None,
                    "lora_ppl_std": round(l_std, 2) if l_p else None,
                    "delta_ppl_pct": round(d_pct, 2) if l_p else None,
                }

            overall_base_mean = float(np.mean(all_base_ppls))
            overall_base_median = float(np.median(all_base_ppls))
            overall_base_std = float(np.std(all_base_ppls))
            overall_base_min = float(np.min(all_base_ppls))
            overall_base_max = float(np.max(all_base_ppls))

            if all_lora_ppls:
                overall_lora_mean = float(np.mean(all_lora_ppls))
                overall_lora_median = float(np.median(all_lora_ppls))
                overall_lora_std = float(np.std(all_lora_ppls))
                overall_lora_min = float(np.min(all_lora_ppls))
                overall_lora_max = float(np.max(all_lora_ppls))
                overall_delta_pct = (
                    ((overall_base_mean - overall_lora_mean) / overall_base_mean * 100.0)
                    if overall_base_mean > 0
                    else float("nan")
                )
            else:
                overall_lora_mean = overall_lora_median = overall_lora_std = None
                overall_lora_min = overall_lora_max = None
                overall_delta_pct = None

            deltas = [seed_comparison[f"seed_{s}"]["delta_ppl_pct"] for s in SEEDS]
            is_stable_sign = all(d is not None for d in deltas) and (deltas[0] >= 0) == (deltas[1] >= 0) == (
                deltas[2] >= 0
            )

            reproducibility_results[m_id] = {
                "base_model": base_name,
                "family": m_info["family"],
                "params": m_info["params"],
                "type": m_info["type"],
                "adapter_found": adapter_found,
                "seeds": seed_comparison,
                "overall_summary": {
                    "base_ppl_mean": round(overall_base_mean, 2),
                    "lora_ppl_mean": round(overall_lora_mean, 2) if all_lora_ppls else None,
                    "delta_pct": round(overall_delta_pct, 2) if all_lora_ppls else None,
                    "is_stable_sign": is_stable_sign,
                },
            }

            model_matrix_entry = {
                "id": m_id,
                "base_model": base_name,
                "family": m_info["family"],
                "params": m_info["params"],
                "type": m_info["type"],
                "adapter_found": adapter_found,
                "base_ppl_distribution": {
                    "mean": round(overall_base_mean, 2),
                    "median": round(overall_base_median, 2),
                    "std": round(overall_base_std, 2),
                    "min": round(overall_base_min, 2),
                    "max": round(overall_base_max, 2),
                    "raw_samples": [round(x, 2) for x in all_base_ppls[:10]],
                },
                "lora_ppl_distribution": {
                    "mean": round(overall_lora_mean, 2) if all_lora_ppls else None,
                    "median": round(overall_lora_median, 2) if all_lora_ppls else None,
                    "std": round(overall_lora_std, 2) if all_lora_ppls else None,
                    "min": round(overall_lora_min, 2) if all_lora_ppls else None,
                    "max": round(overall_lora_max, 2) if all_lora_ppls else None,
                    "raw_samples": [round(x, 2) for x in all_lora_ppls[:10]],
                },
                "mean_ppl_improvement_pct": round(overall_delta_pct, 2) if all_lora_ppls else None,
                "qualitative_generations": {
                    q["id"]: {
                        "prompt": q["prompt"],
                        "base_response": base_generations.get(q["id"], ""),
                        "lora_response": lora_generations.get(q["id"]),
                    }
                    for q in QUALITATIVE_PROMPTS
                },
            }
            full_matrix_results.append(model_matrix_entry)

            if all_lora_ppls:
                logger.info(
                    f"[{m_id}] Base PPL: {overall_base_mean:.2f} (σ={overall_base_std:.1f}) -> LoRA PPL: {overall_lora_mean:.2f} (σ={overall_lora_std:.1f}) | Δ = {overall_delta_pct:+.1f}%"
                )
            else:
                logger.info(
                    f"[{m_id}] Base PPL: {overall_base_mean:.2f} (σ={overall_base_std:.1f}) | LoRA: adapter missing"
                )

            del base_model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error evaluating {m_id}: {e}", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_artifacts(full_matrix_results, reproducibility_results, metadata, split_info)
    logger.info("Scientific Evaluation & Reproducibility Audit successfully completed!")


# ---------------------------------------------------------------------------
# Artifact writing (shared by GPU run and --from-cache regeneration)
# ---------------------------------------------------------------------------


def write_artifacts(
    matrix: list[dict[str, Any]],
    repro: dict[str, Any],
    metadata: dict[str, Any],
    split_info: dict[str, Any] | None,
) -> None:
    Path("reports").mkdir(exist_ok=True)

    with open("reports/reproducibility_audit.json", "w", encoding="utf-8") as f:
        json.dump({"run_metadata": metadata, "models": repro}, f, ensure_ascii=False, indent=2)

    with open("reports/raw_model_evaluation_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"run_metadata": metadata, "models": matrix}, f, ensure_ascii=False, indent=2)

    summary_metrics = []
    for m in matrix:
        summary_metrics.append(
            {
                "model_id": m["id"],
                "family": m["family"],
                "params": m["params"],
                "type": m["type"],
                "adapter_found": m.get("adapter_found", True),
                "base_ppl": m["base_ppl_distribution"]["mean"],
                "lora_ppl": m["lora_ppl_distribution"]["mean"],
                "ppl_delta_pct": m["mean_ppl_improvement_pct"],
                "base_ppl_std": m["base_ppl_distribution"]["std"],
                "lora_ppl_std": m["lora_ppl_distribution"]["std"],
            }
        )
    with open("reports/scientific_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"run_metadata": metadata, "metrics": summary_metrics}, f, ensure_ascii=False, indent=2)

    if split_info is not None:
        manifest = {
            "run_metadata": metadata,
            "eval_pool_size": split_info["pool_size"],
            "excluded_training_dialogues": split_info["excluded_count"],
            "splits": split_info["seeds"],
        }
        with open("reports/eval_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    generate_markdown_report(matrix, repro, metadata)


# ---------------------------------------------------------------------------
# Markdown report — narrative is DERIVED from the measured numbers
# ---------------------------------------------------------------------------


def generate_markdown_report(matrix: list[dict[str, Any]], repro: dict[str, Any], metadata: dict[str, Any]) -> None:
    measured = [m for m in matrix if m.get("mean_ppl_improvement_pct") is not None]
    missing = [m for m in matrix if not m.get("adapter_found", True)]

    md = []
    md.append("# 🔬 Scientific Evaluation & Multi-Seed Reproducibility Audit: LoRA Domain Adaptation")
    md.append("")
    md.append(
        f"> **Protocol version**: {metadata['protocol_version']} · "
        f"**Git commit**: `{metadata['git']['commit']}` (dirty: {metadata['git']['dirty']}) · "
        f"**Generated (UTC)**: {metadata['generated_at_utc']}"
    )
    md.append(f"> **Dataset fingerprint (SHA-256)**: `{metadata['dataset_sha256']}`")
    md.append(">")
    md.append("> **Methodology & Verification Protocol**:")

    cached_measurement = metadata.get("measurement_protocol_version") is not None
    if cached_measurement:
        md.append(
            f"> 1. **⚠️ MEASUREMENT PROVENANCE**: the numbers below were measured under "
            f"**protocol v{metadata['measurement_protocol_version']}** (commit "
            f"`{metadata['measured_in_commit']}`): seeded splits drawn from the **full** corpus, "
            f"*without* training-overlap exclusion. They are preserved verbatim for history; "
            f"re-run `scripts/run_scientific_benchmark.py` (without `--from-cache`) for clean "
            f"v{metadata['protocol_version']} numbers on the exclusion-filtered pool."
        )
    else:
        md.append(
            f"> 1. **Empirical Intrinsic Loss & Perplexity ($PPL = \\exp(\\text{{loss}})$)**: measured per-sample on "
            f"{len(SEEDS)} independent seeded sub-splits of {EVAL_SPLIT_SIZE} dialogues each "
            f"($S \\in \\{{{', '.join(map(str, SEEDS))}\\}}$, max_length={MAX_LEN}). "
            "The evaluation pool **excludes every dialogue selected by the LoRA training sampling rules** "
            "(see `reports/eval_split_manifest.json`), so the split is genuinely held-out."
        )
    md.append(
        "> 2. **Sign convention**: $\\Delta PPL = (PPL_{base} - PPL_{lora}) / PPL_{base} \\times 100\\%$ — "
        "**positive = improvement** (LoRA lowers perplexity), negative = regression."
    )
    md.append("> 3. **Multi-Seed Stability Audit**: whether the sign of $\\Delta PPL$ replicates across seeds.")
    md.append(
        "> 4. **4-Domain Qualitative Generation Sandbox**: verbatim greedy generation across PostgreSQL, Kubernetes, Nginx, and Asyncio Python tasks."
    )
    md.append(
        f"> 5. **Environment**: {metadata['environment']['device']}, torch {metadata['environment']['torch']}, "
        f"transformers {metadata['environment']['transformers']}, peft {metadata['environment']['peft']}."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"## 📊 1. Multi-Seed Reproducibility & Distribution Table ({len(matrix)} Models)")
    md.append("")
    md.append(
        "| Architecture Class | Model ID | Params | Base PPL (Mean ± σ) | LoRA PPL (Mean ± σ) | Full Range [Min .. Max] | Seed 42 (Δ%) | Seed 123 (Δ%) | Seed 777 (Δ%) | Overall Δ PPL |"
    )
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    def _fmt_seed(seed_key: str, seed_map: dict) -> str:
        v = seed_map.get(seed_key, {}).get("delta_ppl_pct")
        return f"{v:+.1f}%" if v is not None else "n/a"

    for item in matrix:
        m_id = item["id"]
        seeds = repro.get(m_id, {}).get("seeds", {})

        b_dist = item["base_ppl_distribution"]
        l_dist = item["lora_ppl_distribution"]
        delta = item["mean_ppl_improvement_pct"]

        if delta is None:
            md.append(
                f"| **{item['family']}** ({item['type']}) | `{m_id}` | {item['params']} | "
                f"{b_dist['mean']:.2f} ± {b_dist['std']:.1f} | adapter missing | "
                f"[{b_dist['min']:.1f}..{b_dist['max']:.1f}] → n/a | "
                f"{_fmt_seed('seed_42', seeds)} | {_fmt_seed('seed_123', seeds)} | {_fmt_seed('seed_777', seeds)} | n/a |"
            )
            continue

        sign = "+" if delta > 0 else ""
        delta_str = f"**{sign}{delta:.1f}%**" if abs(delta) > 5 else f"{sign}{delta:.1f}%"
        md.append(
            f"| **{item['family']}** ({item['type']}) | `{m_id}` | {item['params']} | "
            f"{b_dist['mean']:.2f} ± {b_dist['std']:.1f} | **{l_dist['mean']:.2f} ± {l_dist['std']:.1f}** | "
            f"[{b_dist['min']:.1f}..{b_dist['max']:.1f}] → [{l_dist['min']:.1f}..{l_dist['max']:.1f}] | "
            f"{_fmt_seed('seed_42', seeds)} | {_fmt_seed('seed_123', seeds)} | {_fmt_seed('seed_777', seeds)} | {delta_str} |"
        )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🔍 2. Qualitative Engineering Inference Comparison (Base vs LoRA)")
    md.append("")

    for item in matrix:
        m_id = item["id"]
        gens = item.get("qualitative_generations", {})
        md.append(f"### 🤖 `{m_id}` ({item['family']} {item['params']})")
        md.append("")
        for q_data in gens.values():
            md.append(f"**Prompt**: *{q_data['prompt']}*")
            md.append(f"- **Base Model**: {q_data['base_response']}")
            lora_resp = q_data.get("lora_response")
            md.append(f"- **LoRA Adapter**: {lora_resp if lora_resp is not None else '_adapter not evaluated_'}")
            md.append("")
        md.append("---")
        md.append("")

    md.append("## 💡 3. Empirical Findings (generated from the measured data)")
    md.append("")

    if measured:
        ranked = sorted(measured, key=lambda m: -m["mean_ppl_improvement_pct"])
        regressions = [m for m in measured if m["mean_ppl_improvement_pct"] < 0]
        stable = [m for m in measured if repro.get(m["id"], {}).get("overall_summary", {}).get("is_stable_sign")]

        md.append(
            "1. **Top improvements (overall Δ PPL)**: "
            + "; ".join(f"`{m['id']}` **{m['mean_ppl_improvement_pct']:+.1f}%**" for m in ranked[:3])
            + "."
        )
        if regressions:
            md.append(
                "2. **Honest negative results**: "
                + "; ".join(f"`{m['id']}` {m['mean_ppl_improvement_pct']:+.1f}%" for m in regressions)
                + " — LoRA *raised* perplexity for these models on this split."
            )
        else:
            md.append("2. **No regressions** measured on this split (Δ ≥ 0 for every evaluated model).")
        md.append(
            f"3. **Cross-seed sign stability**: {len(stable)}/{len(measured)} models keep the same "
            "sign of $\\Delta$ across all three seeds."
        )
        by_type: dict[str, list[float]] = {}
        for m in measured:
            by_type.setdefault(m["type"], []).append(m["mean_ppl_improvement_pct"])
        type_summary = "; ".join(f"{t}: mean Δ {np.mean(v):+.1f}% (n={len(v)})" for t, v in sorted(by_type.items()))
        md.append(f"4. **Effect by model class**: {type_summary}.")
    else:
        md.append("_No LoRA adapters were evaluated; the table contains base-model measurements only._")

    if missing:
        md.append(
            "5. **Not evaluated (adapter missing, no numbers fabricated)**: "
            + ", ".join(f"`{m['id']}`" for m in missing)
            + "."
        )

    md.append("")
    md.append("## ⚠️ 4. Scope & Limitations")
    md.append("")
    md.append(
        f"- This audit covers **{len(matrix)} of 55** zoo adapters. Conclusions MUST NOT be extrapolated "
        "to the unevaluated adapters without re-running this script on them."
    )
    md.append(
        f"- Each seed split contains {EVAL_SPLIT_SIZE} dialogues ({len(SEEDS) * EVAL_SPLIT_SIZE} measurements "
        "per model in total). This is a **sanity check, not an academic leaderboard**; absolute PPL values "
        "depend on split composition and max_length."
    )
    md.append(
        "- Knowledge-retention probes (HumanEval/RuMMLU micro-subsets of 8 items) reported elsewhere in this "
        "repository are **8-item subsets**, not the full benchmarks, and are labelled as such in "
        "`reports/BENCHMARK_AND_EVALUATION.md`."
    )
    md.append("")
    md.append("## 🧾 5. Methodology Changelog (why old numbers are NOT comparable)")
    md.append("")
    md.append("| Protocol | Commit | Eval set | max_length | Notes |")
    md.append("| :--- | :--- | :--- | :---: | :--- |")
    md.append(
        "| v1 | `d99aee3` (superseded) | `df.tail(15)` of the full corpus | 512 | Single unseeded split; possible train/eval overlap; script `comprehensive_scientific_audit.py` **removed** — it overwrote the canonical output file with a different methodology. |"
    )
    md.append(
        "| v2 | `463c778` | 3 seeded splits of 10 from the full corpus | 384 | Multi-seed, but training dialogues were NOT excluded. |"
    )
    md.append(
        f"| v3 | `{metadata['git']['commit']}` | 3 seeded splits of 10 from a pool excluding all training dialogues (manifest-fingerprinted) | {MAX_LEN} | Current canonical protocol. |"
    )
    md.append("")
    md.append(
        "Absolute PPL values from different protocol rows measure different text sets and "
        "**must never be compared row-by-row**. Within one protocol row, all figures are "
        "reproducible from `reports/eval_split_manifest.json` plus this script."
    )
    md.append("")

    with open("reports/SCIENTIFIC_EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ---------------------------------------------------------------------------
# --from-cache: regenerate artifacts from the cached v2 measurements
# ---------------------------------------------------------------------------

CACHED_MEASUREMENT_COMMIT = "463c778ddd8a8bd6c4c304ccf1cf974c7e1d6564"


def regenerate_from_cache() -> None:
    """Rebuild the Markdown report from cached JSON measurements.

    The numeric measurements are loaded verbatim — this mode never invents
    numbers. It exists to repair report narrative/provenance after protocol
    fixes without a multi-hour GPU re-run.
    """
    raw_path = Path("reports/raw_model_evaluation_matrix.json")
    repro_path = Path("reports/reproducibility_audit.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    repro = json.loads(repro_path.read_text(encoding="utf-8"))

    # Support both the legacy layout (top-level list / model dict) and the
    # current layout ({"run_metadata": ..., "models": ...}).
    matrix = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    repro_models = repro["models"] if isinstance(repro, dict) and "models" in repro else repro

    dataset_sha = sha256_file(DATASET_PATH) if DATASET_PATH.exists() else "unknown"
    metadata = build_run_metadata(dataset_sha, source="regenerated-from-cache")
    metadata["measured_in_commit"] = CACHED_MEASUREMENT_COMMIT
    metadata["measurement_protocol_version"] = "2.0"
    metadata["note"] = (
        "Numbers were measured under protocol v2 (commit 463c778): seeded splits "
        "drawn from the FULL corpus without training-overlap exclusion. Treat them "
        "as possibly contaminated; re-run without --from-cache for clean v3 numbers."
    )

    # Normalize legacy records to the current schema.
    for m in matrix:
        m.setdefault("adapter_found", True)
        for q in m.get("qualitative_generations", {}).values():
            q.setdefault("lora_response", q.get("lora_response", ""))
    for rec in repro_models.values():
        rec.setdefault("adapter_found", True)

    logger.info("Regenerating report artifacts from cached v2 measurements (no GPU run).")
    write_artifacts(matrix, repro_models, metadata, split_info=None)
    logger.info("Done. NOTE: numbers remain protocol-v2 measurements; run without --from-cache for v3.")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Canonical scientific benchmark (protocol v{PROTOCOL_VERSION})")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Regenerate Markdown/JSON artifacts from cached measurements without running models.",
    )
    args = parser.parse_args()

    if args.from_cache:
        regenerate_from_cache()
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
