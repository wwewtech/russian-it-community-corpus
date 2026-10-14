"""
Local LoRA Fine-Tuning Pipeline for Russian IT Community Dataset.
Optimized for NVIDIA GeForce RTX 3060 (12GB VRAM).
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, cast

from src.bootstrap import setup_runtime_env

setup_runtime_env(pytorch_alloc_conf=True)

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.ml_runtime import set_deterministic_seed, with_retries
from src.observability import record_classified_ml_error, record_lora_result

logger = logging.getLogger(__name__)


def train_lora(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path: str = "dataset_output/jsonl/sft_openai_messages.jsonl",
    output_dir: str = "lora_adapters/russian_it_lora",
    max_steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    seed: int = 42,
) -> Path:
    """
    Execute LoRA domain adaptation on RTX 3060 with PEFT.
    """
    seed_state = set_deterministic_seed(seed)
    print(f"🎲 Deterministic seed: {seed_state}")
    print(
        f"🚀 Initializing LoRA Fine-Tuning on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"📦 Base Model: {model_name}")
    print(f"📚 Dataset: {dataset_path}")

    # 1. Load Tokenizer & Model (network calls wrapped with bounded retry)
    tokenizer: Any = cast(
        Any,
        with_retries(
            lambda: cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True),
            what=f"tokenizer load {model_name}",
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model: Any = with_retries(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        ),
        what=f"model load {model_name}",
    )

    # Enable gradient checkpointing for VRAM saving on RTX 3060
    grad_ckpt: Any = getattr(base_model, "gradient_checkpointing_enable", None)
    if callable(grad_ckpt):
        grad_ckpt()
    model: Any = base_model

    # 2. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = cast(Any, get_peft_model)(model, peft_config)
    trainable_params: Any = getattr(model, "print_trainable_parameters", None)
    if callable(trainable_params):
        trainable_params()

    # 3. Load & Format Dataset.
    # The previous version silently truncated to the first 1000 rows
    # (`if len(raw_dataset) > 1000: raw_dataset = raw_dataset.select(range(1000))`),
    # which contradicted the README claim of full-corpus SFT and produced
    # noticeably under-fitted adapters. We now train on the whole JSONL; pass
    # `--max-samples N` if you need a quick smoke run.
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"📚 Loaded {len(raw_dataset):,} SFT examples from {dataset_path}")

    def format_chatml(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages", [])
        if not messages:
            return {"input_ids": [], "labels": []}

        # Standard SFT: mask the prompt with -100 so CrossEntropyLoss is only
        # computed over the assistant's response tokens, not the user's input.
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tokenized = tokenizer(full_text, max_length=max_seq_length, truncation=True, padding=False)
                labels = list(tokenized["input_ids"])

                if len(messages) >= 2 and messages[-1].get("role") in ("assistant", "gpt"):
                    prompt_text = tokenizer.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=True
                    )
                    tokenized_prompt = tokenizer(prompt_text, max_length=max_seq_length, truncation=True, padding=False)
                    prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels))
                    labels[:prompt_len] = [-100] * prompt_len

                tokenized["labels"] = labels
                return cast("dict[str, Any]", tokenized)
            except Exception as e:
                logger.debug(f"Tokenizer chat_template failed, using fallback: {e}")

        # Fallback for models without a built-in chat template:
        prompt_parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[:-1]]
        prompt_text = "\n".join(prompt_parts) + "\nassistant:"
        full_text = prompt_text + " " + str(messages[-1].get("content", ""))
        tokenized = tokenizer(full_text, max_length=max_seq_length, truncation=True, padding=False)
        labels = list(tokenized["input_ids"])
        tokenized_prompt = tokenizer(prompt_text, max_length=max_seq_length, truncation=True, padding=False)
        prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return cast("dict[str, Any]", tokenized)

    tokenized_dataset = raw_dataset.map(format_chatml, remove_columns=raw_dataset.column_names)

    # 4. Training Arguments.
    # The original config saved only ONE checkpoint (at `max_steps`), which
    # meant a crash on step 99 / 100 destroyed the run. We now save an
    # intermediate checkpoint every 100 steps and keep the last three so a
    # partial run is still recoverable.
    save_every = min(100, max(1, max_steps // 5))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    # 5. Execute Training
    print("⚡ Starting LoRA training loop...")
    adapter_id = Path(output_dir).name
    t0 = time.time()
    try:
        trainer.train()
    except Exception as exc:
        typed = record_classified_ml_error(exc, model_name=model_name, adapter_id=adapter_id)
        record_lora_result(model=model_name, adapter=adapter_id, status="failure")
        logger.error(f"Training failed for {model_name}: {typed}")
        raise
    duration = time.time() - t0
    train_loss = float(trainer.state.log_history[-1]["train_loss"]) if trainer.state.log_history else None
    vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    # 6. Save Adapter
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    record_lora_result(
        model=model_name,
        adapter=adapter_id,
        status="success",
        duration_sec=duration,
        loss=train_loss,
        steps=max_steps,
        vram_bytes=vram_bytes,
    )
    print(f"🎉 LoRA adapter successfully saved to {out_path}!")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducible runs")
    args = parser.parse_args()
    train_lora(model_name=args.model, max_steps=args.steps, seed=args.seed)
