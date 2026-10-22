"""Model catalog and training-budget helpers for the LoRA zoo."""

import hashlib
from pathlib import Path
from typing import Any

ZOO_44_MODELS: list[dict[str, Any]] = [
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
    # --- Flagship 7B-8B models (4-bit QLoRA) -------------------------------
    # These do not fit on a 12 GB consumer card in fp16, so they are
    # quantized to 4 bits (NF4). The capability used to live in
    # a separate copy of the training
    # loop; it is now a flag on the one trainer.
    {
        "id": "heavyweight_qwen2.5_coder_7b",
        "model_name": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "family": "Qwen 2.5 Heavyweight",
        "params": "7B",
        "desc": "Flagship 7B coding model, 4-bit QLoRA for a 12 GB consumer card",
        "load_in_4bit": True,
    },
    {
        "id": "heavyweight_deepseek_r1_7b",
        "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "family": "DeepSeek R1 Heavyweight",
        "params": "7B",
        "desc": "Flagship 7B reasoning model, 4-bit QLoRA for a 12 GB consumer card",
        "load_in_4bit": True,
    },
    {
        "id": "heavyweight_llama3.1_8b",
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "family": "Llama 3.1 Heavyweight",
        "params": "8B",
        "desc": "Flagship 8B general model, 4-bit QLoRA for a 12 GB consumer card",
        "load_in_4bit": True,
    },
    # --- Models that were only reachable through the legacy scripts -------
    # These adapters exist on disk, but their base models were not in this
    # zoo. With the models added here, every adapter in `lora_adapters/` is
    # covered by the single trainer.
    {
        "id": "qwen2_0.5b_instruct",
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "family": "Qwen 2",
        "params": "0.5B",
        "desc": "First-generation Qwen 2 instruct model",
    },
    {
        "id": "qwen2_1.5b_instruct",
        "model_name": "Qwen/Qwen2-1.5B-Instruct",
        "family": "Qwen 2",
        "params": "1.5B",
        "desc": "First-generation Qwen 2 1.5B instruct model",
    },
    {
        "id": "falcon3_1b_instruct",
        "model_name": "tiiuae/Falcon3-1B-Instruct",
        "family": "TII Falcon 3",
        "params": "1.0B",
        "desc": "Falcon 3 1B instruct model",
    },
    {
        "id": "falcon3_3b_instruct",
        "model_name": "tiiuae/Falcon3-3B-Instruct",
        "family": "TII Falcon 3",
        "params": "3.0B",
        "desc": "Falcon 3 3B instruct model",
    },
    {
        "id": "tiny_starcoder_py",
        "model_name": "bigcode/tiny_starcoder_py",
        "family": "BigCode StarCoder",
        "params": "164M",
        "desc": "Tiny Python code model from the StarCoder family",
    },
]

# --- Training budgets ------------------------------------------------------ #
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
