"""
Extended 20 Popular LLMs LoRA Fine-Tuning & Model Hub Synchronizer.
Trains PEFT LoRA domain adapters on NVIDIA GeForce RTX 3060 and uploads to Hugging Face Model Hub.
"""

import contextlib
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import huggingface_hub
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train20Popular")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"

POPULAR_20_MODELS = [
    {
        "id": "falcon3_1b_instruct",
        "model_name": "tiiuae/Falcon3-1B-Instruct",
        "family": "Falcon 3",
        "params": "1.0B",
        "quant": False,
        "desc": "TII Falcon 3 compact high-efficiency instruction model",
    },
    {
        "id": "falcon3_3b_instruct",
        "model_name": "tiiuae/Falcon3-3B-Instruct",
        "family": "Falcon 3",
        "params": "3.0B",
        "quant": False,
        "desc": "TII Falcon 3 3B balanced reasoning instruction model",
    },
    {
        "id": "falcon3_7b_instruct",
        "model_name": "tiiuae/Falcon3-7B-Instruct",
        "family": "Falcon 3 7B",
        "params": "7.0B",
        "quant": True,
        "desc": "TII Falcon 3 7B flagship architecture (4-bit QLoRA)",
    },
    {
        "id": "phi_1_5",
        "model_name": "microsoft/phi-1_5",
        "family": "Phi 1.5",
        "params": "1.3B",
        "quant": False,
        "desc": "Microsoft Phi-1.5 Python & algorithmic reasoning model",
    },
    {
        "id": "phi_2",
        "model_name": "microsoft/phi-2",
        "family": "Phi 2",
        "params": "2.7B",
        "quant": False,
        "desc": "Microsoft Phi-2 high-reasoning compact language model",
    },
    {
        "id": "phi_3_mini_4k_instruct",
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "family": "Phi 3",
        "params": "3.8B",
        "quant": False,
        "desc": "Microsoft Phi-3 4k context instruction tuned model",
    },
    {
        "id": "opt_125m",
        "model_name": "facebook/opt-125m",
        "family": "Meta OPT",
        "params": "125M",
        "quant": False,
        "desc": "Meta OPT 125M lightweight transformer baseline",
    },
    {
        "id": "opt_350m",
        "model_name": "facebook/opt-350m",
        "family": "Meta OPT",
        "params": "350M",
        "quant": False,
        "desc": "Meta OPT 350M generative language model",
    },
    {
        "id": "opt_1.3b",
        "model_name": "facebook/opt-1.3b",
        "family": "Meta OPT",
        "params": "1.3B",
        "quant": False,
        "desc": "Meta OPT 1.3B dense causal language model",
    },
    {
        "id": "opt_2.7b",
        "model_name": "facebook/opt-2.7b",
        "family": "Meta OPT",
        "params": "2.7B",
        "quant": False,
        "desc": "Meta OPT 2.7B generative foundation model",
    },
    {
        "id": "internlm2_5_1_8b_chat",
        "model_name": "internlm/internlm2_5-1_8b-chat",
        "family": "InternLM 2.5",
        "params": "1.8B",
        "quant": False,
        "desc": "InternLM 2.5 lightweight reasoning and chat model",
    },
    {
        "id": "cerebras_gpt_1.3b",
        "model_name": "cerebras/Cerebras-GPT-1.3B",
        "family": "Cerebras-GPT",
        "params": "1.3B",
        "quant": False,
        "desc": "Cerebras-GPT 1.3B Chinchilla-optimal compute model",
    },
    {
        "id": "cerebras_gpt_2.7b",
        "model_name": "cerebras/Cerebras-GPT-2.7B",
        "family": "Cerebras-GPT",
        "params": "2.7B",
        "quant": False,
        "desc": "Cerebras-GPT 2.7B compute-optimal foundation model",
    },
    {
        "id": "falcon_rw_1b",
        "model_name": "tiiuae/falcon-rw-1b",
        "family": "Falcon RW",
        "params": "1.0B",
        "quant": False,
        "desc": "TII Falcon RefinedWeb 1B causal model",
    },
    {
        "id": "olmo_1b_instruct",
        "model_name": "allenai/OLMo-1B-0724-Instruct",
        "family": "AllenAI OLMo",
        "params": "1.0B",
        "quant": False,
        "desc": "AllenAI OLMo truly open foundation instruction model",
    },
    {
        "id": "qwen2_0.5b_instruct",
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "family": "Qwen 2",
        "params": "0.5B",
        "quant": False,
        "desc": "Alibaba Qwen 2 0.5B multilingual instruction model",
    },
    {
        "id": "qwen2_1.5b_instruct",
        "model_name": "Qwen/Qwen2-1.5B-Instruct",
        "family": "Qwen 2",
        "params": "1.5B",
        "quant": False,
        "desc": "Alibaba Qwen 2 1.5B compact multilingual assistant",
    },
    {
        "id": "heavyweight_qwen2.5_7b",
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "family": "Qwen 2.5 7B",
        "params": "7.6B",
        "quant": True,
        "desc": "Qwen 2.5 7B flagship Russian IT discourse model (4-bit)",
    },
    {
        "id": "heavyweight_mistral_7b_v03",
        "model_name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "family": "Mistral 7B v0.3",
        "params": "7.3B",
        "quant": True,
        "desc": "Mistral 7B v0.3 flagship 32k context foundation model (4-bit)",
    },
    {
        "id": "gemma_1.1_2b_it",
        "model_name": "google/gemma-1.1-2b-it",
        "family": "Gemma 1.1",
        "params": "2.0B",
        "quant": False,
        "desc": "Google Gemma 1.1 2B instruction tuned model",
    },
]


def load_conversations(parquet_path: Path, max_samples: int = 400) -> Dataset:
    """Load structured multi-turn SFT dialogues from Parquet."""
    df = pd.read_parquet(parquet_path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df)


def train_single_model(cfg: dict[str, Any], sft_dataset: Dataset, api: huggingface_hub.HfApi) -> bool:
    adapter_id = cfg["id"]
    model_name = cfg["model_name"]
    output_dir = Path(f"lora_adapters/{adapter_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_exist = (output_dir / "adapter_model.safetensors").exists() or (output_dir / "adapter_model.bin").exists()
    if weights_exist:
        logger.info(f"[{adapter_id}] Weights already exist locally. Uploading to Hub...")
        try:
            api.upload_folder(
                folder_path=str(output_dir),
                path_in_repo=adapter_id,
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                ignore_patterns=["checkpoint-*"],
            )
            logger.info(f"[{adapter_id}] Successfully verified & uploaded to HF!")
            return True
        except Exception as e:
            logger.warning(f"[{adapter_id}] Upload check failed: {e}")
            return True

    logger.info("=" * 70)
    logger.info(f"🚀 Training LoRA Adapter: {adapter_id} ({model_name}) | Family: {cfg['family']}")
    logger.info("=" * 70)

    try:
        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

        # 2. Model Loading
        if cfg.get("quant", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

        if hasattr(model, "gradient_checkpointing_enable"):
            with contextlib.suppress(Exception):
                model.gradient_checkpointing_enable()

        # 3. LoRA Configuration
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Fallback for GPT2/OPT/Falcon
        if "gpt" in model_name.lower() or "opt" in model_name.lower() or "cerebras" in model_name.lower():
            target_modules = ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )

        peft_model = get_peft_model(model, peft_config)

        # 4. Tokenization Function
        def format_chat(example, _tokenizer=tokenizer):
            msgs = example.get("messages", [])
            text_blocks = []
            for m in msgs:
                r = m.get("role", "user")
                c = m.get("content", "")
                text_blocks.append(f"<|{r}|>\n{c}")
            full_text = "\n".join(text_blocks)
            tokens = _tokenizer(full_text, truncation=True, max_length=512, padding=False)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized_ds = sft_dataset.map(format_chat, remove_columns=sft_dataset.column_names)

        # 5. Training
        training_args = TrainingArguments(
            output_dir=f"tmp_checkpoints/{adapter_id}",
            max_steps=50,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
        )

        logger.info(f"Training {adapter_id} for 50 steps on RTX 3060...")
        trainer.train()

        # 6. Save Adapter & Tokenizer
        peft_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Save metadata card
        meta = {
            "adapter_id": adapter_id,
            "base_model": model_name,
            "family": cfg["family"],
            "params": cfg["params"],
            "dataset": "wwewtech/russian-it-community-corpus",
            "trained_on": "NVIDIA GeForce RTX 3060 (12GB)",
            "lora_r": 8,
            "lora_alpha": 16,
            "max_steps": 50,
        }
        with open(output_dir / "adapter_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Successfully saved {adapter_id} locally at {output_dir}!")

        # 7. Upload to Hugging Face
        logger.info(f"Uploading {adapter_id} to Hugging Face Model Hub: {MODEL_REPO_ID}...")
        api.upload_folder(
            folder_path=str(output_dir),
            path_in_repo=adapter_id,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            ignore_patterns=["checkpoint-*"],
        )
        logger.info(f"🎉 Successfully uploaded {adapter_id} to Hugging Face Hub!")

        # Cleanup memory
        del peft_model
        del model
        del trainer
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        logger.error(f"❌ Failed training/uploading {adapter_id}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def main():
    api = huggingface_hub.HfApi(token=HF_TOKEN)
    parquet_path = Path("dataset_output/parquet/sft_dialogues.parquet")
    if not parquet_path.exists():
        logger.error(f"Dataset file {parquet_path} not found.")
        return

    logger.info("Loading SFT dialogues dataset for training...")
    sft_ds = load_conversations(parquet_path, max_samples=400)
    logger.info(f"Loaded {len(sft_ds)} training dialogues.")

    successful = 0
    for idx, cfg in enumerate(POPULAR_20_MODELS, 1):
        logger.info(f"\n>>> Processing model [{idx}/{len(POPULAR_20_MODELS)}]: {cfg['id']} <<<")
        ok = train_single_model(cfg, sft_ds, api)
        if ok:
            successful += 1

    logger.info("=" * 70)
    logger.info(f"🎉 COMPLETED BATCH TRAINING: {successful}/{len(POPULAR_20_MODELS)} models processed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
