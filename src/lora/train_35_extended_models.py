"""
Train Additional 35 Popular Foundation Models LoRA Adapters.
Expands the Russian IT Community LoRA Model Zoo from 55 to 90 models.
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_HOME"] = "D:/project_x/.hf_cache"
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
logger = logging.getLogger("Train35Extra")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"

MODELS_35 = [
    # 1. H2O Danube Series (3 models)
    {"id": "h2o_danube_1_8b_chat", "model_name": "h2oai/h2o-danube-1.8b-chat", "family": "H2O Danube", "params": "1.8B", "quant": False},
    {"id": "h2o_danube2_1_8b_chat", "model_name": "h2oai/h2o-danube2-1.8b-chat", "family": "H2O Danube 2", "params": "1.8B", "quant": False},
    {"id": "h2o_danube3_500m_chat", "model_name": "h2oai/h2o-danube3-500m-chat", "family": "H2O Danube 3", "params": "500M", "quant": False},

    # 2. Apple OpenELM Series (3 models)
    {"id": "openelm_270m_instruct", "model_name": "apple/OpenELM-270M-Instruct", "family": "Apple OpenELM", "params": "270M", "quant": False},
    {"id": "openelm_450m_instruct", "model_name": "apple/OpenELM-450M-Instruct", "family": "Apple OpenELM", "params": "450M", "quant": False},
    {"id": "openelm_1_1b_instruct", "model_name": "apple/OpenELM-1_1B-Instruct", "family": "Apple OpenELM", "params": "1.1B", "quant": False},

    # 3. Cerebras-GPT Micro & Compact (3 models)
    {"id": "cerebras_gpt_111m", "model_name": "cerebras/Cerebras-GPT-111M", "family": "Cerebras-GPT", "params": "111M", "quant": False},
    {"id": "cerebras_gpt_256m", "model_name": "cerebras/Cerebras-GPT-256M", "family": "Cerebras-GPT", "params": "256M", "quant": False},
    {"id": "cerebras_gpt_590m", "model_name": "cerebras/Cerebras-GPT-590M", "family": "Cerebras-GPT", "params": "590M", "quant": False},

    # 4. BigCode & StarCoder (3 models)
    {"id": "starcoder_1b", "model_name": "bigcode/starcoderbase-1b", "family": "BigCode StarCoder", "params": "1.0B", "quant": False},
    {"id": "starcoder_3b", "model_name": "bigcode/starcoderbase-3b", "family": "BigCode StarCoder", "params": "3.0B", "quant": False},
    {"id": "tiny_starcoder_py", "model_name": "bigcode/tiny_starcoder_py", "family": "BigCode StarCoder", "params": "164M", "quant": False},

    # 5. Salesforce CodeGen (2 models)
    {"id": "codegen_350m_multi", "model_name": "Salesforce/codegen-350M-multi", "family": "Salesforce CodeGen", "params": "350M", "quant": False},
    {"id": "codegen_2b_multi", "model_name": "Salesforce/codegen-2B-multi", "family": "Salesforce CodeGen", "params": "2.0B", "quant": False},

    # 6. DeciCoder & DeepSeek Code (2 models)
    {"id": "decicoder_1b", "model_name": "Deci/DeciCoder-1b", "family": "Deci DeciCoder", "params": "1.0B", "quant": False},
    {"id": "deepseek_coder_6_7b", "model_name": "unsloth/deepseek-coder-6.7b-instruct-bnb-4bit", "family": "DeepSeek Coder 6.7B", "params": "6.7B", "quant": True},

    # 7. Sber AI mGPT Multilingual (1 model)
    {"id": "sber_mgpt", "model_name": "ai-forever/mGPT", "family": "Sber AI mGPT", "params": "1.3B", "quant": False},

    # 8. InternLM 2 & Chat (2 models)
    {"id": "internlm2_chat_1_8b", "model_name": "internlm/internlm2-chat-1_8b", "family": "InternLM 2", "params": "1.8B", "quant": False},
    {"id": "internlm_chat_7b", "model_name": "unsloth/internlm-chat-7b-bnb-4bit", "family": "InternLM 7B", "params": "7.0B", "quant": True},

    # 9. OpenBMB MiniCPM (2 models)
    {"id": "minicpm_1b_sft", "model_name": "openbmb/MiniCPM-1B-sft-bf16", "family": "MiniCPM", "params": "1.0B", "quant": False},
    {"id": "minicpm_2b_sft", "model_name": "openbmb/MiniCPM-2B-sft-bf16", "family": "MiniCPM", "params": "2.0B", "quant": False},

    # 10. Flagship QLoRA 4-bit Foundation Models (14 models)
    {"id": "heavyweight_deepseek_r1_distill_llama_8b", "model_name": "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit", "family": "DeepSeek R1 LLaMA", "params": "8.0B", "quant": True},
    {"id": "heavyweight_qwen2.5_coder_14b", "model_name": "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit", "family": "Qwen 2.5 Coder 14B", "params": "14.0B", "quant": True},
    {"id": "heavyweight_qwen2.5_14b", "model_name": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", "family": "Qwen 2.5 14B", "params": "14.0B", "quant": True},
    {"id": "heavyweight_llama3_8b_instruct", "model_name": "unsloth/llama-3-8b-Instruct-bnb-4bit", "family": "Meta LLaMA 3", "params": "8.0B", "quant": True},
    {"id": "heavyweight_llama3_8b", "model_name": "unsloth/llama-3-8b-bnb-4bit", "family": "Meta LLaMA 3 Base", "params": "8.0B", "quant": True},
    {"id": "heavyweight_saiga_llama3_8b", "model_name": "IlyaGusev/saiga_llama3_8b", "family": "Saiga Russian LLaMA 3", "params": "8.0B", "quant": True},
    {"id": "heavyweight_vikhr_7b_instruct", "model_name": "Vikhrmodels/Vikhr-7B-instruct_0.1", "family": "Vikhr Russian 7B", "params": "7.0B", "quant": True},
    {"id": "heavyweight_baichuan2_7b", "model_name": "baichuan-inc/Baichuan2-7B-Chat", "family": "Baichuan 2 7B", "params": "7.0B", "quant": True},
    {"id": "heavyweight_mistral_7b_v01", "model_name": "unsloth/mistral-7b-bnb-4bit", "family": "Mistral 7B v0.1", "params": "7.3B", "quant": True},
    {"id": "heavyweight_mistral_7b_instruct_v02", "model_name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", "family": "Mistral 7B v0.2", "params": "7.3B", "quant": True},
    {"id": "heavyweight_zephyr_7b_beta", "model_name": "HuggingFaceH4/zephyr-7b-beta", "family": "HuggingFace Zephyr", "params": "7.3B", "quant": True},
    {"id": "heavyweight_openchat_3_5", "model_name": "openchat/openchat_3.5", "family": "OpenChat 3.5", "params": "7.0B", "quant": True},
    {"id": "heavyweight_starling_lm_7b", "model_name": "berkeley-nest/Starling-LM-7B-alpha", "family": "Berkeley Starling", "params": "7.0B", "quant": True},
    {"id": "heavyweight_solar_10_7b", "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0", "family": "Upstage SOLAR", "params": "10.7B", "quant": True},
]


def load_conversations(parquet_path: Path, max_samples: int = 350) -> Dataset:
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
        logger.info(f"[{adapter_id}] Weights exist locally. Uploading to Hub...")
        try:
            api.upload_folder(
                folder_path=str(output_dir),
                path_in_repo=adapter_id,
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                ignore_patterns=["checkpoint-*"],
            )
            logger.info(f"[{adapter_id}] Successfully uploaded to HF!")
            return True
        except Exception as e:
            logger.warning(f"[{adapter_id}] Upload check failed: {e}")
            return True

    logger.info("=" * 70)
    logger.info(f"🚀 Training LoRA Adapter: {adapter_id} ({model_name}) | Family: {cfg['family']}")
    logger.info("=" * 70)

    try:
        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
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
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        # 3. LoRA Configuration
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if any(k in model_name.lower() for k in ["gpt", "opt", "cerebras", "codegen", "starcoder", "mgpt"]):
            target_modules = ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "wte", "wpe"]

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
        def format_chat(example):
            msgs = example.get("messages", [])
            text_blocks = []
            for m in msgs:
                r = m.get("role", "user")
                c = m.get("content", "")
                text_blocks.append(f"<|{r}|>\n{c}")
            full_text = "\n".join(text_blocks)
            tokens = tokenizer(full_text, truncation=True, max_length=512, padding=False)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized_ds = sft_dataset.map(format_chat, remove_columns=sft_dataset.column_names)

        # 5. Training
        training_args = TrainingArguments(
            output_dir=f"tmp_checkpoints/{adapter_id}",
            max_steps=40,
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

        logger.info(f"Training {adapter_id} for 40 steps on RTX 3060...")
        trainer.train()

        # 6. Save Adapter & Tokenizer
        peft_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        meta = {
            "adapter_id": adapter_id,
            "base_model": model_name,
            "family": cfg["family"],
            "params": cfg["params"],
            "dataset": "wwewtech/russian-it-community-corpus",
            "trained_on": "NVIDIA GeForce RTX 3060 (12GB)",
            "lora_r": 8,
            "lora_alpha": 16,
            "max_steps": 40,
        }
        with open(output_dir / "adapter_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Saved {adapter_id} locally at {output_dir}!")

        # 7. Upload to Hugging Face
        logger.info(f"Uploading {adapter_id} to Hugging Face: {MODEL_REPO_ID}...")
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

    logger.info("Loading SFT dialogues dataset for training 35 extra models...")
    sft_ds = load_conversations(parquet_path, max_samples=350)
    logger.info(f"Loaded {len(sft_ds)} training dialogues.")

    successful = 0
    for idx, cfg in enumerate(MODELS_35, 1):
        logger.info(f"\n>>> Processing model [{idx}/{len(MODELS_35)}]: {cfg['id']} <<<")
        ok = train_single_model(cfg, sft_ds, api)
        if ok:
            successful += 1

    logger.info("=" * 70)
    logger.info(f"🎉 COMPLETED EXTRA BATCH TRAINING: {successful}/{len(MODELS_35)} models processed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
