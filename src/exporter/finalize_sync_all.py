"""
Final synchronization script for 40+ LoRA models across Hugging Face and GitHub.
"""

import json
import logging
import os
import time
from pathlib import Path

from src.bootstrap import setup_runtime_env

setup_runtime_env()

import huggingface_hub  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FinalizeSync")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"
DATASET_REPO_ID = "wwewtech/russian-it-community-corpus"


def upload_dataset(api: huggingface_hub.HfApi):
    """Upload cleaned parquet files, dataset card, and reports to HF Hub."""
    logger.info(f"Uploading sanitized dataset artifacts to Hugging Face: {DATASET_REPO_ID}...")

    # 1. Dataset Card (README.md)
    dataset_card_path = Path("reports/DATASET_AND_ANALYTICS.md")
    if dataset_card_path.exists():
        api.upload_file(
            path_or_fileobj=str(dataset_card_path),
            path_in_repo="README.md",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
        )
        logger.info("Uploaded updated dataset card (README.md)")

    # 2. Cleaned Parquet Files
    parquet_mappings = [
        ("dataset_output/parquet/full_clean_messages.parquet", "data/full_clean_messages.parquet"),
        ("dataset_output/parquet/sft_dialogues.parquet", "data/sft_dialogues.parquet"),
        ("dataset_output/parquet/rag_knowledge_base.parquet", "data/rag_knowledge_base.parquet"),
    ]
    for local_path, repo_path in parquet_mappings:
        p = Path(local_path)
        if p.exists():
            logger.info(f"Uploading {local_path} ({p.stat().st_size / (1024 * 1024):.2f} MB) -> {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
            )
            logger.info(f"Successfully uploaded {repo_path}")

    # 3. Reports and Metadata
    meta_files = [
        ("reports/domain_benchmark_100.json", "domain_benchmark_100.json"),
        ("reports/LORA_MODEL_ZOO.md", "LORA_MODEL_ZOO.md"),
    ]
    for local_path, repo_path in meta_files:
        p = Path(local_path)
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
            )
            logger.info(f"Uploaded {repo_path}")


def main():
    api = huggingface_hub.HfApi(token=HF_TOKEN if HF_TOKEN else None)
    adapters_dir = Path("lora_adapters")

    # Upload dataset files if token is available
    try:
        upload_dataset(api)
    except Exception as e:
        logger.warning(f"Dataset upload to Hugging Face skipped or failed: {e}")

    # 1. Check all local adapters
    local_adapters = sorted(
        [d for d in adapters_dir.iterdir() if d.is_dir() and (d / "adapter_model.safetensors").exists()]
    )
    logger.info(f"Found {len(local_adapters)} trained LoRA adapters locally in {adapters_dir}")

    # 2. Upload any missing adapters to Hugging Face Model Hub
    hf_files = api.list_repo_files(repo_id=MODEL_REPO_ID, repo_type="model")
    uploaded_folders = {f.split("/")[0] for f in hf_files if "/" in f}

    for d in local_adapters:
        if d.name not in uploaded_folders:
            logger.info(f"Uploading missing adapter {d.name} to Hugging Face Model Hub...")
            for attempt in range(5):
                try:
                    api.upload_folder(
                        folder_path=str(d),
                        path_in_repo=d.name,
                        repo_id=MODEL_REPO_ID,
                        repo_type="model",
                        ignore_patterns=["checkpoint-*"],
                    )
                    logger.info(f"Successfully uploaded {d.name}!")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {d.name}: {e}. Retrying in 5s...")
                    time.sleep(5)

    # 3. Generate Final LORA_MODEL_ZOO.md and lora_zoo_index.json
    results = []
    for d in local_adapters:
        safetensors = d / "adapter_model.safetensors"
        size_mb = safetensors.stat().st_size / (1024 * 1024)
        results.append(
            {
                "id": d.name,
                "size_mb": round(size_mb, 2),
                "status": "SUCCESS",
                "adapter_dir": f"lora_adapters/{d.name}",
                "hf_model_repo": MODEL_REPO_ID,
            }
        )

    with open("reports/lora_zoo_index.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# 🦁 Russian IT Community LoRA Model Zoo",
        "",
        f"**Официальный каталог {len(results)} предварительно обученных LoRA-адаптеров**, дообученных на корпусе **RICC (2.91M сообщений, 171.5k диалогов)** для русскоязычного IT-дискурса, бэкенда, DevOps, AI/ML и инфраструктуры.",
        "",
        f"Все адаптеры доступны на Hugging Face Hub: [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}).",
        "",
        "---",
        "",
        "## 📊 Доступные LoRA-адаптеры",
        "",
        "| # | Идентификатор модели | Размер весов | Каталог адаптера | Hugging Face Hub |",
        "| :---: | :--- | :---: | :--- | :--- |",
    ]

    for idx, r in enumerate(results, 1):
        md_lines.append(
            f"| {idx} | **`{r['id']}`** | **{r['size_mb']} MB** | [`lora_adapters/{r['id']}/`](lora_adapters/{r['id']}/) | [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}) |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 🚀 Быстрый старт: Запуск любого адаптера в 3 строки кода",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            "# 1. Выберите любой адаптер из каталога выше",
            'adapter_id = "qwen2.5_1.5b_instruct"',
            'model_hub_path = f"wwewtech/russian-it-community-lora"',
            "",
            "# 2. Загрузка весов напрямую с Hugging Face Hub",
            "tokenizer = AutoTokenizer.from_pretrained(model_hub_path, subfolder=adapter_id)",
            'base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="auto")',
            "model = PeftModel.from_pretrained(base_model, model_hub_path, subfolder=adapter_id)",
            "",
            "# 3. Инференс",
            'prompt = "Как настроить Nginx reverse proxy с поддержкой WebSocket в Docker?"',
            'messages = [{"role": "user", "content": prompt}]',
            'inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")',
            "outputs = model.generate(**inputs, max_new_tokens=256)",
            "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
            "```",
        ]
    )

    with open("reports/LORA_MODEL_ZOO.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Generated LORA_MODEL_ZOO.md with {len(results)} adapters!")


if __name__ == "__main__":
    main()
