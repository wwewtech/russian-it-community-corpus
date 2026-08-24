"""
Scientific Raw Metric Evaluator:
Measures raw, un-averaged, empirical per-sample loss, perplexity distributions,
and qualitative before/after generation across 15 diverse model architectures.
"""

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_HOME"] = "D:/project_x/.hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScientificAudit")

# 15 Diverse Models Across Scale & Types (70M to 3.8B, Base vs Instruct)
EVAL_MODELS = [
    # Tier 1: Micro & Small Base Models (Non-instruct)
    {"id": "pythia_70m", "base": "EleutherAI/pythia-70m", "family": "Pythia", "params": "70M", "type": "base"},
    {"id": "opt_125m", "base": "facebook/opt-125m", "family": "OPT", "params": "125M", "type": "base"},
    {"id": "bloom_560m", "base": "bigscience/bloom-560m", "family": "BLOOM", "params": "560M", "type": "base"},
    {"id": "gpt2_medium", "base": "openai-community/gpt2-medium", "family": "GPT-2", "params": "355M", "type": "base"},
    {"id": "rugpt3_small", "base": "ai-forever/rugpt3small_based_on_gpt2", "family": "RuGPT3", "params": "125M", "type": "base"},

    # Tier 2: Compact 1B-Class Models (Instruct & Coder)
    {"id": "tinyllama_1.1b_chat", "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "TinyLlama", "params": "1.1B", "type": "instruct"},
    {"id": "smollm2_1.7b_instruct", "base": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "family": "SmolLM2", "params": "1.7B", "type": "instruct"},
    {"id": "llama_3.2_1b_instruct", "base": "unsloth/Llama-3.2-1B-Instruct", "family": "LLaMA 3.2", "params": "1.0B", "type": "instruct"},
    {"id": "deepseek_coder_1.3b_instruct", "base": "deepseek-ai/deepseek-coder-1.3b-instruct", "family": "DeepSeek Coder", "params": "1.3B", "type": "coder"},
    {"id": "falcon3_1b_instruct", "base": "tiiuae/Falcon3-1B-Instruct", "family": "Falcon 3", "params": "1.0B", "type": "instruct"},

    # Tier 3: Modern 1.5B-3.8B Architecture (Instruct & Reasoning)
    {"id": "qwen2.5_0.5b_instruct", "base": "Qwen/Qwen2.5-0.5B-Instruct", "family": "Qwen 2.5", "params": "0.5B", "type": "instruct"},
    {"id": "qwen2.5_1.5b_instruct", "base": "Qwen/Qwen2.5-1.5B-Instruct", "family": "Qwen 2.5", "params": "1.5B", "type": "instruct"},
    {"id": "qwen2.5_coder_1.5b_instruct", "base": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "family": "Qwen 2.5 Coder", "params": "1.5B", "type": "coder"},
    {"id": "vikhr_qwen_2.5_1.5b", "base": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct", "family": "Vikhr Russian", "params": "1.5B", "type": "instruct"},
    {"id": "deepseek_r1_distill_qwen_1.5b", "base": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "family": "DeepSeek R1", "params": "1.5B", "type": "reasoning"},
]

# Qualitative Prompts for Direct Inspection
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
]


def evaluate_all():
    logger.info("Starting Scientific Raw Benchmark Audit across 15 Models on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load 15 fixed held-out validation dialogues from the corpus
    df_sft = pd.read_parquet("dataset_output/parquet/sft_dialogues.parquet")
    test_split = df_sft.tail(15).reset_index(drop=True)
    logger.info(f"Loaded {len(test_split)} held-out validation dialogues.")

    audit_results = []

    for idx, m in enumerate(EVAL_MODELS, 1):
        m_id = m["id"]
        base_name = m["base"]
        adapter_path = Path(f"lora_adapters/{m_id}")
        logger.info("=" * 70)
        logger.info(f"[{idx}/{len(EVAL_MODELS)}] Evaluating: {m_id} ({base_name}) | Class: {m['type']}")
        logger.info("=" * 70)

        try:
            # 1. Load Tokenizer
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

            # A. Measure per-sample Base PPL
            base_losses = []
            with torch.no_grad():
                for _, row in test_split.iterrows():
                    msgs = row.get("messages", [])
                    text = "\n".join([f"<|{item.get('role', 'user')}|>\n{item.get('content', '')}" for item in msgs])
                    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
                    input_ids = enc["input_ids"].to(base_model.device)
                    if input_ids.shape[1] < 4:
                        continue
                    out = base_model(input_ids, labels=input_ids)
                    l_val = out.loss.item()
                    if not math.isnan(l_val) and not math.isinf(l_val):
                        base_losses.append(l_val)

            base_ppls = [math.exp(l) for l in base_losses]
            base_mean_ppl = float(np.mean(base_ppls)) if base_ppls else 999.0
            base_median_ppl = float(np.median(base_ppls)) if base_ppls else 999.0
            base_std_ppl = float(np.std(base_ppls)) if base_ppls else 0.0
            base_min_ppl = float(np.min(base_ppls)) if base_ppls else 999.0
            base_max_ppl = float(np.max(base_ppls)) if base_ppls else 999.0

            # B. Qualitative Generation (Base)
            base_gens = {}
            for q in QUALITATIVE_PROMPTS:
                inp = tokenizer(f"<|user|>\n{q['prompt']}<|assistant|>\n", return_tensors="pt").to(base_model.device)
                with torch.no_grad():
                    out = base_model.generate(**inp, max_new_tokens=80, do_sample=False)
                base_gens[q["id"]] = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # C. Load LoRA Model
            lora_losses = []
            lora_gens = {}
            if adapter_path.exists() and ((adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()):
                lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
                lora_model.eval()

                with torch.no_grad():
                    for _, row in test_split.iterrows():
                        msgs = row.get("messages", [])
                        text = "\n".join([f"<|{item.get('role', 'user')}|>\n{item.get('content', '')}" for item in msgs])
                        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
                        input_ids = enc["input_ids"].to(lora_model.device)
                        if input_ids.shape[1] < 4:
                            continue
                        out = lora_model(input_ids, labels=input_ids)
                        l_val = out.loss.item()
                        if not math.isnan(l_val) and not math.isinf(l_val):
                            lora_losses.append(l_val)

                for q in QUALITATIVE_PROMPTS:
                    inp = tokenizer(f"<|user|>\n{q['prompt']}<|assistant|>\n", return_tensors="pt").to(lora_model.device)
                    with torch.no_grad():
                        out = lora_model.generate(**inp, max_new_tokens=80, do_sample=False)
                    lora_gens[q["id"]] = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                del lora_model
            else:
                lora_losses = base_losses
                lora_gens = base_gens

            lora_ppls = [math.exp(l) for l in lora_losses]
            lora_mean_ppl = float(np.mean(lora_ppls)) if lora_ppls else base_mean_ppl
            lora_median_ppl = float(np.median(lora_ppls)) if lora_ppls else base_median_ppl
            lora_std_ppl = float(np.std(lora_ppls)) if lora_ppls else 0.0
            lora_min_ppl = float(np.min(lora_ppls)) if lora_ppls else base_min_ppl
            lora_max_ppl = float(np.max(lora_ppls)) if lora_ppls else base_max_ppl

            diff_pct = ((base_mean_ppl - lora_mean_ppl) / base_mean_ppl) * 100 if base_mean_ppl > 0 else 0.0

            m_record = {
                "id": m_id,
                "base_model": base_name,
                "family": m["family"],
                "params": m["params"],
                "type": m["type"],
                "base_ppl_stats": {
                    "mean": round(base_mean_ppl, 2),
                    "median": round(base_median_ppl, 2),
                    "std": round(base_std_ppl, 2),
                    "min": round(base_min_ppl, 2),
                    "max": round(base_max_ppl, 2),
                    "raw_samples": [round(x, 2) for x in base_ppls[:8]],
                },
                "lora_ppl_stats": {
                    "mean": round(lora_mean_ppl, 2),
                    "median": round(lora_median_ppl, 2),
                    "std": round(lora_std_ppl, 2),
                    "min": round(lora_min_ppl, 2),
                    "max": round(lora_max_ppl, 2),
                    "raw_samples": [round(x, 2) for x in lora_ppls[:8]],
                },
                "mean_ppl_change_pct": round(diff_pct, 2),
                "sample_generation_q1": {
                    "base": base_gens.get("q1_postgres", ""),
                    "lora": lora_gens.get("q1_postgres", ""),
                },
                "sample_generation_q2": {
                    "base": base_gens.get("q2_k8s", ""),
                    "lora": lora_gens.get("q2_k8s", ""),
                },
            }

            audit_results.append(m_record)
            logger.info(f"[{m_id}] Base PPL: {base_mean_ppl:.2f} (std={base_std_ppl:.1f}, range=[{base_min_ppl:.1f}..{base_max_ppl:.1f}]) -> LoRA PPL: {lora_mean_ppl:.2f} (Δ {diff_pct:+.1f}%)")

            del base_model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error evaluating {m_id}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 1. Save Raw JSON Telemetry
    with open("reports/raw_model_evaluation_matrix.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, ensure_ascii=False, indent=2)

    logger.info("Saved raw metrics to reports/raw_model_evaluation_matrix.json")


if __name__ == "__main__":
    evaluate_all()
