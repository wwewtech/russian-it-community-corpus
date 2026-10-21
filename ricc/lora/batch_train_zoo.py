"""
Extended Multi-Model LoRA Zoo (50+ Models) for Russian IT Community Corpus.
Trains and uploads domain PEFT LoRA adapters across 50+ open-source LLMs on NVIDIA GeForce RTX 3060.

Two budgets are available via ``--profile``:

* ``pilot`` — the tiny smoke budget (400 dialogues / 15 steps) used to validate
  the pipeline end to end.
* ``full`` — production budgets per model size tier (up to 20k dialogues /
  2000 steps), with a 2% holdout eval split, step-level training logs
  (``training_log.jsonl``), automatic checkpoints and resume support for
  multi-hour runs, and an aggregated run report in ``reports/``.

``--hours N`` replaces the step budget with a wall-clock budget per model
(graceful stop + save), ``--holdout-tail N`` reserves the last N dialogues of
the corpus for honest post-training benchmarks (see
``scripts/run_lora_benchmark.py``), and ``scripts/night_run_full_corpus.py``
chains several time-boxed full-corpus runs plus benchmarks in one command.
"""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

from ricc.bootstrap import setup_runtime_env

setup_runtime_env()

import huggingface_hub
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ricc.lora.policy_gate import (
    CANONICAL_HOLDOUT_TAIL,
    compute_coverage,
    evaluate_compliance,
    required_steps_for_epoch,
    trainable_dialogues,
)
from ricc.ml_runtime import set_deterministic_seed, with_retries
from ricc.observability import record_classified_ml_error, record_lora_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoRAZoo44")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"

ZOO_44_MODELS = [
    # --- 1. Qwen 2.5 General & Coder Family (6 models) ---
    {
        "id": "qwen2.5_0.5b_instruct",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen 2.5",
        "params": "0.5B",
        "desc": "Ultra-lightweight edge LLM with high Russian language proficiency",
    },
    {
        "id": "qwen2.5_1.5b_instruct",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen 2.5",
        "params": "1.5B",
        "desc": "Compact high-efficiency assistant for local edge deployment",
    },
    {
        "id": "qwen2.5_3b_instruct",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "family": "Qwen 2.5",
        "params": "3.0B",
        "desc": "Balanced reasoning and domain knowledge model",
    },
    {
        "id": "qwen2.5_coder_0.5b_instruct",
        "model_name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "family": "Qwen 2.5 Coder",
        "params": "0.5B",
        "desc": "Specialized coding and programming assistant adapter",
    },
    {
        "id": "qwen2.5_coder_1.5b_instruct",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "Qwen 2.5 Coder",
        "params": "1.5B",
        "desc": "High-speed coding assistant with deep code syntax mastery",
    },
    {
        "id": "qwen2.5_coder_3b_instruct",
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "family": "Qwen 2.5 Coder",
        "params": "3.0B",
        "desc": "Mid-sized code generation and debugging model",
    },
    {
        "id": "qwen2.5_math_1.5b_instruct",
        "model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "family": "Qwen 2.5 Math",
        "params": "1.5B",
        "desc": "Quantitative reasoning and algorithm solver",
    },
    # --- 2. DeepSeek Reasoning & Coder Series (2 models) ---
    {
        "id": "deepseek_r1_distill_qwen_1.5b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "DeepSeek R1",
        "params": "1.5B",
        "desc": "Chain-of-thought reasoning distilled model for architectural tasks",
    },
    {
        "id": "deepseek_coder_1.3b_instruct",
        "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "family": "DeepSeek Coder",
        "params": "1.3B",
        "desc": "Classic compact DeepSeek coding model",
    },
    # --- 3. Meta LLaMA 3.2 Open Weights (2 models) ---
    {
        "id": "llama_3.2_1b_instruct",
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "family": "Llama 3.2",
        "params": "1.0B",
        "desc": "Meta Llama 3.2 compact instruction model",
    },
    {
        "id": "llama_3.2_3b_instruct",
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "family": "Llama 3.2",
        "params": "3.0B",
        "desc": "Meta Llama 3.2 high-capacity lightweight model",
    },
    # --- 4. SmolLM2 Series (3 models) ---
    {
        "id": "smollm2_135m_instruct",
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "family": "SmolLM2",
        "params": "135M",
        "desc": "Micro-model for embedded hardware, IoT and instant inference",
    },
    {
        "id": "smollm2_360m_instruct",
        "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "family": "SmolLM2",
        "params": "360M",
        "desc": "Lightweight sub-half-billion instruction tuned model",
    },
    {
        "id": "smollm2_1.7b_instruct",
        "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "family": "SmolLM2",
        "params": "1.7B",
        "desc": "Fast instruction-tuned small language model",
    },
    # --- 5. SmolLM v1 Series (3 models) ---
    {
        "id": "smollm_135m_instruct",
        "model_name": "HuggingFaceTB/SmolLM-135M-Instruct",
        "family": "SmolLM v1",
        "params": "135M",
        "desc": "Original SmolLM 135M instruction model",
    },
    {
        "id": "smollm_360m_instruct",
        "model_name": "HuggingFaceTB/SmolLM-360M-Instruct",
        "family": "SmolLM v1",
        "params": "360M",
        "desc": "Original SmolLM 360M instruction model",
    },
    {
        "id": "smollm_1.7b_instruct",
        "model_name": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "family": "SmolLM v1",
        "params": "1.7B",
        "desc": "Original SmolLM 1.7B instruction model",
    },
    # --- 6. SOTA Russian NLP (3 models) ---
    {
        "id": "vikhr_qwen_2.5_0.5b",
        "model_name": "Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct",
        "family": "Vikhr Russian NLP",
        "params": "0.5B",
        "desc": "Russian-specialized adaptation on Qwen 2.5 architecture",
    },
    {
        "id": "vikhr_qwen_2.5_1.5b",
        "model_name": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
        "family": "Vikhr Russian NLP",
        "params": "1.5B",
        "desc": "Enhanced Russian vocabulary and morphology model",
    },
    {
        "id": "vikhr_llama_3.2_1b",
        "model_name": "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct",
        "family": "Vikhr Russian NLP",
        "params": "1.0B",
        "desc": "Llama 3.2 adapted for Russian technical discourse",
    },
    # --- 7. Sber AI RuGPT3 Generative Family (3 models) ---
    {
        "id": "rugpt3_small",
        "model_name": "ai-forever/rugpt3small_based_on_gpt2",
        "family": "Sber AI",
        "params": "125M",
        "desc": "Classic Russian GPT-2 small generative model",
    },
    {
        "id": "rugpt3_medium",
        "model_name": "ai-forever/rugpt3medium_based_on_gpt2",
        "family": "Sber AI",
        "params": "350M",
        "desc": "Russian GPT-2 medium generative architecture",
    },
    {
        "id": "rugpt3_large",
        "model_name": "ai-forever/rugpt3large_based_on_gpt2",
        "family": "Sber AI",
        "params": "760M",
        "desc": "Russian GPT-2 large 760M generative foundation model",
    },
    # --- 8. Google Gemma Series (2 models) ---
    {
        "id": "gemma_2_2b_it",
        "model_name": "unsloth/gemma-2-2b-it",
        "family": "Gemma 2",
        "params": "2.0B",
        "desc": "Google Gemma 2 high-precision instruction model",
    },
    {
        "id": "gemma_1.1_2b_it",
        "model_name": "google/gemma-1.1-2b-it",
        "family": "Gemma 1.1",
        "params": "2.0B",
        "desc": "Google Gemma 1.1 lightweight instruction model",
    },
    # --- 9. Microsoft Phi Family (4 models) ---
    {
        "id": "phi_3.5_mini_instruct",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "family": "Phi 3.5",
        "params": "3.8B",
        "desc": "Microsoft Phi 3.5 mini with deep synthetic reasoning",
    },
    {
        "id": "phi_3_mini_4k_instruct",
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "family": "Phi 3",
        "params": "3.8B",
        "desc": "Microsoft Phi-3 4k context instruction model",
    },
    {
        "id": "phi_2",
        "model_name": "microsoft/phi-2",
        "family": "Phi 2",
        "params": "2.7B",
        "desc": "Microsoft Phi-2 high reasoning small language model",
    },
    {
        "id": "phi_1_5",
        "model_name": "microsoft/phi-1_5",
        "family": "Phi 1.5",
        "params": "1.3B",
        "desc": "Microsoft Phi-1.5 Python and reasoning model",
    },
    # --- 10. IBM Granite & Edge Architectures (4 models) ---
    {
        "id": "granite_3b_code_instruct",
        "model_name": "ibm-granite/granite-3b-code-instruct",
        "family": "IBM Granite",
        "params": "3.0B",
        "desc": "IBM Granite code generation and enterprise assistant",
    },
    {
        "id": "internlm2_5_1_8b_chat",
        "model_name": "internlm/internlm2_5-1_8b-chat",
        "family": "InternLM 2.5",
        "params": "1.8B",
        "desc": "InternLM 2.5 lightweight reasoning and tool-use model",
    },
    {
        "id": "minicpm_2b_dpo",
        "model_name": "openbmb/MiniCPM-2B-dpo-bf16",
        "family": "MiniCPM",
        "params": "2.0B",
        "desc": "OpenBMB MiniCPM compact edge language model",
    },
    {
        "id": "tinyllama_1.1b_chat",
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "TinyLlama",
        "params": "1.1B",
        "desc": "Classic compact architecture with fast throughput",
    },
    # --- 11. Meta OPT Open Family (2 models) ---
    {
        "id": "opt_1.3b",
        "model_name": "facebook/opt-1.3b",
        "family": "Meta OPT",
        "params": "1.3B",
        "desc": "Meta Open Pre-trained Transformer 1.3B",
    },
    {
        "id": "opt_2.7b",
        "model_name": "facebook/opt-2.7b",
        "family": "Meta OPT",
        "params": "2.7B",
        "desc": "Meta Open Pre-trained Transformer 2.7B",
    },
    # --- 12. EleutherAI Pythia SOTA Research Series (2 models) ---
    {
        "id": "pythia_1.4b",
        "model_name": "EleutherAI/pythia-1.4b-deduped",
        "family": "EleutherAI Pythia",
        "params": "1.4B",
        "desc": "EleutherAI Pythia 1.4B scientific research checkpoint",
    },
    {
        "id": "pythia_2.8b",
        "model_name": "EleutherAI/pythia-2.8b-deduped",
        "family": "EleutherAI Pythia",
        "params": "2.8B",
        "desc": "EleutherAI Pythia 2.8B language model",
    },
    # --- 13. Stability AI StableLM Series (2 models) ---
    {
        "id": "stablelm_2_1_6b_chat",
        "model_name": "stabilityai/stablelm-2-1_6b-chat",
        "family": "Stability AI",
        "params": "1.6B",
        "desc": "Stability AI StableLM 2 1.6B chat model",
    },
    {
        "id": "stablelm_2_zephyr_1_6b",
        "model_name": "stabilityai/stablelm-2-zephyr-1_6b",
        "family": "Stability AI",
        "params": "1.6B",
        "desc": "Stability AI Zephyr aligned 1.6B model",
    },
    # --- 14. Cerebras & BigScience BLOOM (3 models) ---
    {
        "id": "cerebras_gpt_1.3b",
        "model_name": "cerebras/Cerebras-GPT-1.3B",
        "family": "Cerebras GPT",
        "params": "1.3B",
        "desc": "Cerebras CS-2 trained compute-optimal 1.3B model",
    },
    {
        "id": "cerebras_gpt_2.7b",
        "model_name": "cerebras/Cerebras-GPT-2.7B",
        "family": "Cerebras GPT",
        "params": "2.7B",
        "desc": "Cerebras CS-2 trained 2.7B model",
    },
    {
        "id": "bloom_1b7",
        "model_name": "bigscience/bloom-1b7",
        "family": "BigScience BLOOM",
        "params": "1.7B",
        "desc": "BigScience multilingual BLOOM 1.7B foundation model",
    },
    # --- 15. Base Baseline ---
    {
        "id": "russian_it_lora",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen Baseline",
        "params": "0.5B",
        "desc": "Primary Russian IT baseline LoRA adapter",
    },
    # --- 16. Additional Expanded Architectures (10 models) ---
    {
        "id": "qwen1.5_0.5b_chat",
        "model_name": "Qwen/Qwen1.5-0.5B-Chat",
        "family": "Qwen 1.5",
        "params": "0.5B",
        "desc": "Classic Qwen 1.5 compact chat architecture",
    },
    {
        "id": "qwen1.5_1.8b_chat",
        "model_name": "Qwen/Qwen1.5-1.8B-Chat",
        "family": "Qwen 1.5",
        "params": "1.8B",
        "desc": "Qwen 1.5 1.8B mid-sized dialogue model",
    },
    {
        "id": "opt_350m",
        "model_name": "facebook/opt-350m",
        "family": "Meta OPT",
        "params": "350M",
        "desc": "Meta OPT 350M lightweight transformer",
    },
    {
        "id": "opt_125m",
        "model_name": "facebook/opt-125m",
        "family": "Meta OPT",
        "params": "125M",
        "desc": "Meta OPT 125M ultra-lightweight edge checkpoint",
    },
    {
        "id": "pythia_410m",
        "model_name": "EleutherAI/pythia-410m-deduped",
        "family": "EleutherAI Pythia",
        "params": "410M",
        "desc": "EleutherAI Pythia 410M research checkpoint",
    },
    {
        "id": "pythia_70m",
        "model_name": "EleutherAI/pythia-70m-deduped",
        "family": "EleutherAI Pythia",
        "params": "70M",
        "desc": "EleutherAI Pythia 70M micro-parameter model",
    },
    {
        "id": "bloom_560m",
        "model_name": "bigscience/bloom-560m",
        "family": "BigScience BLOOM",
        "params": "560M",
        "desc": "BigScience multilingual BLOOM 560M model",
    },
    {
        "id": "falcon_rw_1b",
        "model_name": "tiiuae/falcon-rw-1b",
        "family": "TII Falcon",
        "params": "1.0B",
        "desc": "Technology Innovation Institute Falcon 1B RefinedWeb",
    },
    {
        "id": "gpt2_medium",
        "model_name": "openai-community/gpt2-medium",
        "family": "OpenAI GPT-2",
        "params": "355M",
        "desc": "Classic GPT-2 Medium 355M generative foundation",
    },
    {
        "id": "gpt2_large",
        "model_name": "openai-community/gpt2-large",
        "family": "OpenAI GPT-2",
        "params": "774M",
        "desc": "Classic GPT-2 Large 774M generative foundation",
    },
]

# --- Training budgets ------------------------------------------------------ #
# PILOT is a SMOKE profile: 15 steps over 400 dialogues. It exists to prove
# that a base model, a dataset and the trainer fit together on the local GPU.
# It is NOT a training run and must never be published as a final adapter.
PILOT_BUDGET: dict[str, int] = {"max_samples": 400, "steps": 15}
FULL_PROFILE_EVAL_FRAC = 0.02

# The "full" profile is derived from the DATASET, not from a hardcoded tier.
# The previous tier budgets (600-2000 steps) covered 2.77-9.24 % of the
# corpus, so no run using them could ever satisfy the binding policy.
# The step count for a full epoch is now computed by policy_gate from the
# actual number of trainable dialogues; see budget_for() below.
FULL_TIER_BUDGETS: dict[str, dict[str, int]] = {
    "tiny": {"max_samples": 200000, "steps": 0},
    "small": {"max_samples": 200000, "steps": 0},
    "mid": {"max_samples": 200000, "steps": 0},
    "large": {"max_samples": 200000, "steps": 0},
}
_DATASET_SHA256_CACHE: dict[str, str] = {}


def param_tier(params: str) -> str:
    """Map a params label such as ``135M`` / ``1.5B`` / ``3.8B`` to a budget tier."""
    text = params.strip().upper()
    try:
        value = float(text[:-1])
    except ValueError:
        return "small"
    if text.endswith("M"):
        value /= 1000.0
    if value <= 0.5:
        return "tiny"
    if value <= 1.2:
        return "small"
    if value <= 2.3:
        return "mid"
    return "large"


def dataset_sha256(path: str) -> str:
    """sha256 of the training file (cached per process) for provenance manifests."""
    cached = _DATASET_SHA256_CACHE.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    _DATASET_SHA256_CACHE[path] = digest.hexdigest()
    return digest.hexdigest()


class _TimeLimitCallback(TrainerCallback):
    """Stop training gracefully once a wall-clock budget is exhausted."""

    def __init__(self, budget_sec: float) -> None:
        self.budget_sec = budget_sec
        self.started_at: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.monotonic()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.started_at is not None and time.monotonic() - self.started_at >= self.budget_sec:
            control.should_training_stop = True


def detect_target_modules(model: torch.nn.Module) -> list[str]:
    """Dynamically detect attention projection linear module names for LoRA.

    Only module types PEFT can wrap are considered (Linear/Conv1d and
    transformers' Conv1D), and single-letter targets must match the leaf
    name exactly. Substring matching used to pick up non-linear modules
    whose *path* merely contained the letter — Phi-3.5's ``activation_fn``
    (a ``SiLUActivation``) matched the bare ``"v"`` target and crashed
    PEFT with "Target module SiLUActivation() is not supported".
    """
    module_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and type(module).__name__ != "Conv1D":
            continue
        leaf = name.rsplit(".", 1)[-1]
        for target in [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "query_key_value",
            "Wqkv",
            "to_q",
            "to_v",
            "c_attn",
            "q",
            "v",
            "k",
            "out_proj",
        ]:
            if leaf == target or (len(target) > 1 and target in name):
                module_names.add(leaf)
                break
    if not module_names:
        module_names = {"q_proj", "v_proj"}
    return sorted(module_names)


def _dataset_dialogue_count(dataset_path: str | Path) -> int:
    """Number of dialogues in the SFT JSONL corpus (0 when unavailable).

    Used to derive the full-profile step budget from the dataset instead of
    a hardcoded tier table. Counts non-empty lines without loading the
    corpus into memory.
    """
    path = Path(dataset_path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def train_and_upload_single_model(
    meta: dict[str, Any],
    dataset_path: str = "dataset_output/jsonl/sft_openai_messages.jsonl",
    base_output_dir: Path = Path("lora_adapters"),
    max_steps: int = 20,
    max_samples: int = 400,
    max_seq_length: int = 512,
    seed: int = 42,
    upload: bool = True,
    force: bool = False,
    eval_frac: float = 0.0,
    keep_checkpoints: bool = False,
    holdout_tail: int = 0,
    time_budget_hours: float | None = None,
    save_total_limit: int = 10,
) -> dict[str, Any]:
    model_id = meta["id"]
    model_name = meta["model_name"]
    out_dir = base_output_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and (out_dir / "adapter_model.safetensors").exists() and (out_dir / "adapter_config.json").exists():
        logger.info(f"⏩ Adapter for {model_name} ({model_id}) already exists at {out_dir}. Skipping training...")
        return {
            "id": model_id,
            "model_name": model_name,
            "family": meta["family"],
            "params": meta["params"],
            "description": meta["desc"],
            "status": "SKIPPED",
            "train_loss": None,
            "training_time_sec": 0,
            "adapter_dir": str(out_dir),
        }

    # Interrupted long runs leave checkpoint-* folders without a final adapter:
    # resume from the latest one instead of restarting from scratch.
    resume_from: Path | None = None
    existing_checkpoints = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    if existing_checkpoints and not force:
        resume_from = existing_checkpoints[-1]
        logger.info(f"↻ Resuming {model_id} from {resume_from.name}")

    start_time = time.time()
    logger.info(f"=== Starting LoRA Training: {model_name} ({meta['family']} · {meta['params']}) ===")
    set_deterministic_seed(seed)

    try:
        tokenizer: Any = cast(
            Any,
            with_retries(
                lambda: cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True, trust_remote_code=True),
                what=f"tokenizer load {model_name}",
            ),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|endoftext|>"

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model: Any = with_retries(
            lambda: AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            ),
            what=f"model load {model_name}",
        )

        grad_ckpt: Any = getattr(model, "gradient_checkpointing_enable", None)
        if callable(grad_ckpt):
            grad_ckpt()

        # Remote-code models (Phi-3.5's hub modeling_phi3.py) still call the
        # Cache.get_usable_length API that transformers 4.57 removed whenever
        # use_cache is on — including the eval forward, which crashed a
        # 1.5-hour run just before its first checkpoint. Disable the KV cache
        # for training/eval entirely (no cache is used by the Trainer anyway).
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False

        target_mods = detect_target_modules(model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_mods,
            bias="none",
        )
        peft_model = get_peft_model(model, peft_config)

        # --force cleanup is deferred until the base model loaded and the LoRA
        # wrapper was constructed successfully: a run that fails earlier (bad
        # download, unsupported target modules, ...) must NOT destroy the
        # previously trained adapter. Phi-3.5's SiLUActivation crash did
        # exactly that before target detection was fixed.
        # --force deletes the FINISHED adapter, but the checkpoints are
        # kept: they are the only way to continue a long run instead of
        # starting it again. The canonical full run is 21 420 steps
        # (~25 h on a 3B model); discarding the checkpoints would throw
        # away hours of work every time the command is re-issued.
        if force and out_dir.exists() and any(out_dir.iterdir()):
            checkpoints = [p for p in out_dir.glob("checkpoint-*") if p.is_dir()]
            for stale in out_dir.iterdir():
                if stale in checkpoints:
                    continue
                if stale.is_dir():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
            if checkpoints:
                logger.info(
                    "--force: cleared the finished adapter at %s, keeping %d checkpoint(s) for resume",
                    out_dir,
                    len(checkpoints),
                )
            else:
                logger.info("--force: cleared existing adapter at %s for retraining", out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        raw_ds = load_dataset("json", data_files=dataset_path, split="train")
        dataset_total = len(raw_ds)
        if holdout_tail > 0 and dataset_total > holdout_tail:
            raw_ds = raw_ds.select(range(dataset_total - holdout_tail))
        if len(raw_ds) > max_samples:
            raw_ds = raw_ds.select(range(max_samples))

        def format_dialogue(example: dict[str, Any], _tokenizer: Any = tokenizer) -> dict[str, Any]:
            msgs = example.get("messages", [])
            if not msgs:
                return cast("dict[str, Any]", {"input_ids": [], "labels": []})
            # Standard SFT: mask the user prompt with -100 so the loss is
            # computed only over the assistant's response tokens. Training
            # on prompt tokens (the old `labels = input_ids.copy()`
            # behaviour) teaches the model to parrot the user instead of
            # answering — see the v12.0.4 prompt-loss-masking fix in
            # ricc/lora/train_lora.py, mirrored here.
            try:
                txt = _tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                parts = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"[{role.upper()}]: {content}")
                txt = "\n".join(parts)
            tok = _tokenizer(txt, max_length=max_seq_length, truncation=True, padding=False)
            labels = list(tok["input_ids"])
            if len(msgs) >= 2 and msgs[-1].get("role") in ("assistant", "gpt"):
                try:
                    prompt_text = _tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs[:-1]]
                    prompt_text = "\n".join(prompt_parts) + "\nassistant:"
                tokenized_prompt = _tokenizer(prompt_text, max_length=max_seq_length, truncation=True, padding=False)
                prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels))
                labels[:prompt_len] = [-100] * prompt_len
            tok["labels"] = labels
            return cast("dict[str, Any]", tok)

        tokenized_ds = raw_ds.map(format_dialogue, remove_columns=raw_ds.column_names)

        eval_ds: Any = None
        if eval_frac > 0:
            split = tokenized_ds.train_test_split(test_size=eval_frac, seed=seed)
            tokenized_ds = split["train"]
            eval_ds = split["test"]
            if len(eval_ds) > 256:
                eval_ds = eval_ds.select(range(256))

        total_tokens = int(sum(len(ids) for ids in tokenized_ds["input_ids"]))
        avg_len = total_tokens / max(1, len(tokenized_ds))

        if max_steps >= 5000:
            # Save more often than we evaluate: a crash in the eval forward
            # used to lose every step since the last checkpoint (save runs
            # after eval), and hours of training with it.
            log_every, eval_every, save_every, enable_saves = 25, 1000, 500, True
        else:
            log_every = max(1, max_steps // 20)
            eval_every = max(1, max_steps // 10)
            save_every = max(1, max_steps // 4)
            enable_saves = max_steps >= 100

        train_args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=log_every,
            save_strategy="steps" if enable_saves else "no",
            save_steps=save_every,
            # The checkpoints are the resume points of a multi-hour run, so
            # more than two are kept: with the canonical 21 420 steps and
            # a checkpoint every 500 steps, a limit of 2 would delete every
            # resume point within a thousand steps.
            save_total_limit=save_total_limit,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=eval_every,
            report_to="none",
            seed=seed,
        )

        callbacks: list[TrainerCallback] | None = None
        if time_budget_hours is not None:
            callbacks = [_TimeLimitCallback(time_budget_hours * 3600.0)]

        trainer = Trainer(
            model=peft_model,
            args=train_args,
            train_dataset=tokenized_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
            callbacks=callbacks,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_res = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
        train_loss = float(train_res.training_loss)
        actual_steps = int(trainer.state.global_step)
        tokens_seen = int(actual_steps * 8 * avg_len)
        # Coverage is measured against the dialogues the trainer ACTUALLY saw.
        # The previous denominator (dataset_total) included the held-out tail,
        # so a run covering 101% of the real training set reported "100%"
        # while a run covering exactly one real epoch reported "98.93%".
        train_dialogues = len(tokenized_ds)
        coverage_pct = compute_coverage(actual_steps, train_dialogues)
        required_steps = required_steps_for_epoch(train_dialogues)
        peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

        eval_loss: float | None = None
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                eval_loss = float(entry["eval_loss"])
                break
        eval_perplexity = math.exp(eval_loss) if eval_loss is not None and eval_loss < 50 else None

        with (out_dir / "training_log.jsonl").open("w", encoding="utf-8") as log_file:
            for entry in trainer.state.log_history:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        peft_model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        if not keep_checkpoints:
            for cp in out_dir.glob("checkpoint-*"):
                if cp.is_dir():
                    shutil.rmtree(cp)

        readme_path = out_dir / "README.md"
        readme_path.write_text(
            f"""# Russian IT Community Corpus — LoRA Adapter
## Base Model: `{model_name}` ({meta["family"]} · {meta["params"]})

Fine-tuned on the **RICC** corpus (2.82M messages, 171.5k SFT dialogues).
""",
            encoding="utf-8",
        )

        elapsed = time.time() - start_time
        tokens_per_sec = round(tokens_seen / elapsed, 1) if elapsed > 0 else 0.0
        logger.info(
            f"Finished {model_name} in {elapsed:.1f}s | Loss: {train_loss:.4f}"
            + (f" | Eval: {eval_loss:.4f}" if eval_loss is not None else "")
            + f" | ~{tokens_per_sec:.0f} tok/s"
        )

        # Provenance manifest: its presence marks an adapter retrained with the
        # prompt-masking objective. Legacy adapters (no training_meta.json)
        # predate the masking fix and need --force retraining to acquire it.
        (out_dir / "training_meta.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "base_model": model_name,
                    "dataset": dataset_path,
                    "dataset_sha256": dataset_sha256(dataset_path),
                    "max_steps": max_steps,
                    "actual_steps": actual_steps,
                    "required_steps_for_full_epoch": required_steps,
                    "time_budget_hours": time_budget_hours,
                    "max_seq_length": max_seq_length,
                    "max_samples": max_samples,
                    "samples_used": len(tokenized_ds),
                    "train_dialogues": train_dialogues,
                    "dataset_total_dialogues": dataset_total,
                    "holdout_tail_dialogues": holdout_tail,
                    "dialogues_seen_approx": actual_steps * 8,
                    "corpus_coverage_pct": coverage_pct,
                    "eval_frac": eval_frac,
                    "eval_samples": len(eval_ds) if eval_ds is not None else 0,
                    "seed": seed,
                    "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": target_mods},
                    "prompt_masking": True,
                    "train_loss": round(train_loss, 4),
                    "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
                    "eval_perplexity": round(eval_perplexity, 2) if eval_perplexity is not None else None,
                    "tokens_seen_approx": tokens_seen,
                    "tokens_per_sec": tokens_per_sec,
                    "peak_vram_bytes": peak_vram,
                    "resumed_from": resume_from.name if resume_from is not None else None,
                    "training_time_sec": round(elapsed, 1),
                    "generator": "ricc/lora/batch_train_zoo.py",
                    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # --- Policy gate -----------------------------------------------------
        # A time-boxed or partial run is NOT a final adapter. It must be
        # labelled PARTIAL and must not be uploaded to the public zoo.
        compliance = evaluate_compliance(out_dir)
        partial = time_budget_hours is not None or not compliance.compliant
        run_status = "PARTIAL" if partial else "SUCCESS"
        if partial:
            logger.warning(
                f"{model_id}: run is PARTIAL — {'; '.join(compliance.reasons) or 'wall-clock budget hit'}. "
                "It will not be uploaded as a final adapter."
            )

        record_lora_result(
            model=model_name,
            adapter=model_id,
            status="success",
            duration_sec=elapsed,
            loss=train_loss,
            steps=actual_steps,
            vram_bytes=peak_vram,
        )

        # Auto-upload to Hugging Face Model Zoo (opt-out with --no-upload).
        # The policy gate runs first: a partial adapter is never published.
        if upload and partial:
            logger.warning(f"HF upload skipped for {model_id}: adapter does not satisfy the training policy")
        elif upload:
            try:
                api = huggingface_hub.HfApi(token=HF_TOKEN)
                with_retries(
                    lambda: api.upload_folder(
                        folder_path=str(out_dir),
                        path_in_repo=model_id,
                        repo_id=MODEL_REPO_ID,
                        repo_type="model",
                        ignore_patterns=["checkpoint-*"],
                    ),
                    what=f"HF upload {model_id}",
                )
                logger.info(f"Uploaded {model_id} directly to HF Hub: {MODEL_REPO_ID}")
            except Exception as upload_err:
                logger.warning(f"HF upload skipped for {model_id}: {upload_err}")
        else:
            logger.info(f"HF upload disabled for {model_id} (--no-upload)")

        del peft_model
        del model
        del trainer
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "id": model_id,
            "model_name": model_name,
            "family": meta["family"],
            "params": meta["params"],
            "description": meta["desc"],
            "status": run_status,
            "policy_compliant": compliance.compliant,
            "policy_reasons": list(compliance.reasons),
            "required_steps_for_full_epoch": required_steps,
            "train_dialogues": train_dialogues,
            "train_loss": round(train_loss, 4),
            "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
            "steps": actual_steps,
            "planned_steps": max_steps,
            "corpus_coverage_pct": coverage_pct,
            "samples_used": len(tokenized_ds),
            "tokens_seen_approx": tokens_seen,
            "tokens_per_sec": tokens_per_sec,
            "peak_vram_bytes": peak_vram,
            "training_time_sec": round(elapsed, 1),
            "adapter_dir": str(out_dir),
        }

    except Exception as e:
        typed = record_classified_ml_error(e, model_name=model_name, adapter_id=model_id)
        record_lora_result(model=model_name, adapter=model_id, status="failure")
        logger.error(f"Failed training {model_name}: {typed}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "id": model_id,
            "model_name": model_name,
            "family": meta["family"],
            "params": meta["params"],
            "description": meta["desc"],
            "status": "FAILED",
            "error": str(e),
        }


def _fmt_metric(result: dict[str, Any], key: str, pattern: str) -> str:
    value = result.get(key)
    return pattern.format(value) if isinstance(value, (int, float)) else "—"


def update_run_report(results: list[dict[str, Any]], report_dir: Path = Path("reports")) -> None:
    """Crash-safe aggregate report for the current zoo run.

    Deliberately writes its own files and never touches the canonical catalog
    (``reports/lora_zoo_index.json`` / ``reports/LORA_MODEL_ZOO.md``), which is
    regenerated from the Hub file tree by ``scripts/regenerate_model_catalog.py``.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "lora_zoo_training_run.json"
    md_path = report_dir / "LORA_ZOO_TRAINING_REPORT.md"

    merged: dict[str, Any] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            merged = dict(loaded.get("models", {}))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {json_path}; starting a fresh run report")
    for r in results:
        merged[r["id"]] = r

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {"generated_at_utc": generated_at, "models": merged}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# LoRA Zoo Training Run Report",
        "",
        f"Generated {generated_at} · {len(merged)} adapters in the run log.",
        "",
        "| Adapter | Base model | Status | Train loss | Eval loss | Steps | Samples | Time | tok/s |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_id in sorted(merged):
        r = merged[model_id]
        md_lines.append(
            f"| `{model_id}` | `{r.get('model_name', '?')}` | {r.get('status', '?')} "
            f"| {_fmt_metric(r, 'train_loss', '{:.4f}')} | {_fmt_metric(r, 'eval_loss', '{:.4f}')} "
            f"| {r.get('steps', '—')} | {r.get('samples_used', '—')} "
            f"| {_fmt_metric(r, 'training_time_sec', '{:.0f}s')} "
            f"| {_fmt_metric(r, 'tokens_per_sec', '{:.0f}')} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"Run report updated: {len(merged)} adapters -> {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="full",
        help=(
            "Training budget: 'full' (default) runs one complete epoch over the whole corpus, "
            "derived from the dataset size; 'pilot' is a 15-step SMOKE profile that must "
            "never be published as a final adapter."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_output/jsonl/sft_openai_messages.jsonl",
        help="SFT JSONL corpus used for training (its size drives the full-profile budget)",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Optimizer steps per model (overrides the profile budget)"
    )
    parser.add_argument(
        "--samples", type=int, default=None, help="Max SFT dialogues per model (overrides the profile budget)"
    )
    parser.add_argument(
        "--eval-frac", type=float, default=None, help="Holdout fraction for eval (full profile default: 0.02)"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Wall-clock training budget per model in hours: stop gracefully and save when reached",
    )
    parser.add_argument(
        "--holdout-tail",
        type=int,
        default=CANONICAL_HOLDOUT_TAIL,
        help=(
            "Reserve the last N dialogues as a benchmark holdout (excluded from training). "
            f"Default {CANONICAL_HOLDOUT_TAIL} (the canonical held-out tail). Use 0 to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducible runs")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated adapter ids to train (default: all models in the zoo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lora_adapters"),
        help="Adapter output root (default: lora_adapters)",
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip Hugging Face upload")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for the aggregate run report (default: reports)",
    )
    parser.add_argument(
        "--no-report",
        "--no-catalogs",
        action="store_true",
        dest="no_report",
        help="Do not update the aggregate run report (alias: --no-catalogs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain adapters that already exist (the checkpoints are kept, so a long run can be continued)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=10,
        help=(
            "How many checkpoints to keep per adapter. The checkpoints are the resume "
            "points of a multi-hour run, so the default keeps more than the two that "
            "were kept before (which deleted every resume point within a thousand steps)."
        ),
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep per-model checkpoint folders after a successful run",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=None,
        help='Per-model budget overrides: {"default": {"steps": N, "samples": N}, "<adapter_id>": {...}}',
    )
    args = parser.parse_args()

    models = ZOO_44_MODELS
    if args.only:
        wanted = {m.strip() for m in args.only.split(",") if m.strip()}
        known = {m["id"] for m in ZOO_44_MODELS}
        unknown = wanted - known
        if unknown:
            parser.error(f"unknown adapter id(s): {', '.join(sorted(unknown))}")
        models = [m for m in ZOO_44_MODELS if m["id"] in wanted]
        if not models:
            parser.error("--only matched no models")

    profile_file_data: dict[str, Any] = {}
    if args.profile_file is not None:
        profile_file_data = cast(dict[str, Any], json.loads(args.profile_file.read_text(encoding="utf-8")))
    default_overrides: dict[str, Any] = dict(profile_file_data.get("default", {}))

    def budget_for(meta: dict[str, Any]) -> tuple[int, int, float]:
        if args.profile == "pilot":
            steps, samples, eval_frac = PILOT_BUDGET["steps"], PILOT_BUDGET["max_samples"], 0.0
        else:
            # The full profile is derived from the dataset: one complete epoch
            # over the trainable dialogues, capped by --holdout-tail. This
            # replaces the old hardcoded tier budgets (600-2000 steps), which
            # covered only 2.77-9.24 % of the corpus and therefore could not
            # satisfy the binding training policy.
            tier = param_tier(meta["params"])
            eval_frac = FULL_PROFILE_EVAL_FRAC
            total = _dataset_dialogue_count(args.dataset) if args.dataset else 0
            holdout = args.holdout_tail if args.holdout_tail is not None else CANONICAL_HOLDOUT_TAIL
            trainable = trainable_dialogues(total, holdout_tail=holdout, eval_frac=eval_frac)
            steps = required_steps_for_epoch(trainable)
            samples = FULL_TIER_BUDGETS[tier]["max_samples"]
            if total:
                logger.info(
                    f"Full profile for {meta['id']}: {total} dialogues - {holdout} holdout - "
                    f"{eval_frac:.3%} eval = {trainable} trainable -> {steps} steps for one epoch"
                )
        overrides: dict[str, Any] = {**default_overrides, **dict(profile_file_data.get(meta["id"], {}))}
        steps = int(overrides.get("steps", steps))
        samples = int(overrides.get("samples", samples))
        eval_frac = float(overrides.get("eval_frac", eval_frac))
        if args.steps is not None:
            steps = args.steps
        elif args.hours is not None:
            steps = 10_000_000
        if args.samples is not None:
            samples = args.samples
        if args.eval_frac is not None:
            eval_frac = args.eval_frac
        return steps, samples, eval_frac

    results: list[dict[str, Any]] = []
    run_start = time.time()
    for idx, meta in enumerate(models, 1):
        steps, samples, eval_frac = budget_for(meta)
        logger.info(
            f"[{idx}/{len(models)}] {meta['model_name']} ({meta['params']}): "
            f"steps={steps}, samples={samples}, eval_frac={eval_frac}"
        )
        res = train_and_upload_single_model(
            meta,
            dataset_path=args.dataset,
            base_output_dir=args.output_dir,
            max_steps=steps,
            max_samples=samples,
            seed=args.seed,
            upload=not args.no_upload,
            force=args.force,
            eval_frac=eval_frac,
            keep_checkpoints=args.keep_checkpoints,
            holdout_tail=args.holdout_tail,
            time_budget_hours=args.hours,
            save_total_limit=args.save_total_limit,
        )
        results.append(res)
        if not args.no_report:
            update_run_report(results, report_dir=args.report_dir)
        if idx < len(models):
            avg_sec = (time.time() - run_start) / idx
            eta_hours = avg_sec * (len(models) - idx) / 3600
            logger.info(
                f"⏱ avg {avg_sec / 60:.1f} min/model · ETA for {len(models) - idx} remaining: ~{eta_hours:.1f} h"
            )

    logger.info(f"LoRA Zoo run finished: {len(models)} model(s) processed.")

    failed_ids = [str(r.get("id")) for r in results if r.get("status") == "FAILED"]
    if failed_ids:
        logger.error(f"{len(failed_ids)} model(s) failed: {', '.join(failed_ids)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
