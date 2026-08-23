"""
Heavyweight 5 SOTA Foundation Models QLoRA Trainer.
Maximally pushes 12GB RTX 3060 to its hardware limits using 4-bit NF4 QLoRA:
1. Qwen 2.5 Coder 7B Instruct (7.6B)
2. DeepSeek R1 Distill Qwen 7B (7.6B)
3. Meta LLaMA 3.1 8B Instruct (8.0B)
4. Qwen 2.5 7B Instruct (7.6B)
5. Mistral 7B Instruct v0.3 (7.3B)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

os.environ["HF_HOME"] = "D:/project_x/.hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Heavyweight5")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"

HEAVYWEIGHT_5_MODELS = [
    {
        "id": "heavyweight_qwen2.5_coder_7b",
        "model_name": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "family": "Qwen 2.5 Coder 7B",
        "params": "7.6B",
        "desc": "SOTA open-source code generation and software architecture foundation model (4-bit)",
    },
    {
        "id": "heavyweight_deepseek_r1_7b",
        "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "family": "DeepSeek R1 7B",
        "params": "7.6B",
        "desc": "Chain-of-thought distilled reasoning model for complex engineering systems (4-bit)",
    },
    {
        "id": "heavyweight_llama3.1_8b",
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "family": "Meta LLaMA 3.1 8B",
        "params": "8.0B",
        "desc": "Meta LLaMA 3.1 8B flagship open foundation model (4-bit)",
    },
    {
        "id": "heavyweight_qwen2.5_7b",
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "family": "Qwen 2.5 7B",
        "params": "7.6B",
        "desc": "High-capacity Russian language and technical reasoning flagship model (4-bit)",
    },
    {
        "id": "heavyweight_mistral_7b_v03",
        "model_name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "family": "Mistral 7B v0.3",
        "params": "7.3B",
        "desc": "Mistral AI 7B v0.3 instruction tuned foundation model with 32k context (4-bit)",
    },
]


def load_sft_dataset(parquet_path: Path, max_samples: int = 600) -> Dataset:
    """Load high-quality SFT dialogues from Parquet."""
    df = pd.read_parquet(parquet_path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df)


def format_chat_prompt(example: dict, tokenizer: Any) -> dict:
    """Format dialogue into conversational prompt."""
    conv = example.get("conversations") or example.get("messages")
    if conv and isinstance(conv, list):
        messages = [{"role": msg.get("from") or msg.get("role"), "content": msg.get("value") or msg.get("content")} for msg in conv]
        role_map = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
        clean_msgs = [{"role": role_map.get(m["role"], "user"), "content": str(m["content"])} for m in messages if m["content"]]
    else:
        q = example.get("query") or example.get("instruction") or "Расскажи про IT архитектуру"
        r = example.get("response") or example.get("output") or example.get("content") or ""
        clean_msgs = [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(r)}]

    try:
        text = tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in clean_msgs)
    return {"text": text}


def train_single_heavyweight(
    spec: dict,
    dataset: Dataset,
    output_dir: Path,
    steps: int = 50,
    lr: float = 2e-4,
) -> bool:
    """Train a single 7B-8B model with 4-bit NF4 QLoRA."""
    model_name = spec["model_name"]
    model_id = spec["id"]
    save_path = output_dir / model_id

    if save_path.exists() and (save_path / "adapter_model.safetensors").exists():
        logger.info(f"⏩ Heavyweight adapter for {model_name} ({model_id}) already exists. Skipping...")
        return True

    logger.info(f"=== 🏋️ Starting Heavyweight 4-bit QLoRA: {model_name} ({spec['params']}) ===")
    t0 = time.time()

    # 1. Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

        # LoRA Config targeting all linear projection layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Tokenize Dataset
        formatted_ds = dataset.map(lambda ex: format_chat_prompt(ex, tokenizer), remove_columns=dataset.column_names)

        def tokenize_fn(examples):
            tokenized = tokenizer(examples["text"], truncation=True, max_length=256, padding=False)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_ds = formatted_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training Args
        training_args = TrainingArguments(
            output_dir=str(save_path / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=steps,
            learning_rate=lr,
            fp16=True,
            logging_steps=5,
            save_strategy="no",
            optim="paged_adamw_8bit",
            report_to="none",
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        )

        trainer.train()

        # Save Adapter
        save_path.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        elapsed = time.time() - t0
        logger.info(f"✅ Heavyweight {model_name} trained successfully in {elapsed:.1f}s!")

        # Free GPU Memory
        del peft_model
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"❌ Failed heavyweight training for {model_name}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    sft_path = Path("dataset_output/parquet/sft_dialogues.parquet")
    output_dir = Path("lora_adapters")

    logger.info("=== Loading SFT Dialogue Dataset ===")
    dataset = load_sft_dataset(sft_path, max_samples=500)
    logger.info(f"Loaded {len(dataset)} samples for Heavyweight SFT training.")

    for idx, spec in enumerate(HEAVYWEIGHT_5_MODELS, 1):
        logger.info(f"\n--- [{idx}/{len(HEAVYWEIGHT_5_MODELS)}] {spec['family']} ({spec['params']}) ---")
        train_single_heavyweight(spec, dataset, output_dir, steps=args.steps)

    logger.info("🎉 All 5 Heavyweight SOTA models processed!")


if __name__ == "__main__":
    main()
