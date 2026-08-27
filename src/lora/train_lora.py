"""
Local LoRA Fine-Tuning Pipeline for Russian IT Community Dataset.
Optimized for NVIDIA GeForce RTX 3060 (12GB VRAM).
"""

import argparse
import logging
from pathlib import Path

from src.bootstrap import setup_runtime_env

setup_runtime_env(pytorch_alloc_conf=True)

import torch  # noqa: E402
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def train_lora(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path: str = "dataset_output/jsonl/sft_openai_messages.jsonl",
    output_dir: str = "lora_adapters/russian_it_lora",
    max_steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
):
    """
    Execute LoRA domain adaptation on RTX 3060 with PEFT.
    """
    print(
        f"🚀 Initializing LoRA Fine-Tuning on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"📦 Base Model: {model_name}")
    print(f"📚 Dataset: {dataset_path}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Enable gradient checkpointing for VRAM saving on RTX 3060
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load & Format Dataset
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    if len(raw_dataset) > 1000:
        raw_dataset = raw_dataset.select(range(1000))

    def format_chatml(example):
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, max_length=max_seq_length, truncation=True, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(format_chatml, remove_columns=raw_dataset.column_names)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=max_steps,
        save_total_limit=1,
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
    trainer.train()

    # 6. Save Adapter
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"🎉 LoRA adapter successfully saved to {out_path}!")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    train_lora(model_name=args.model, max_steps=args.steps)
