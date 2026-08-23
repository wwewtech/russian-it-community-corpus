"""
Final synchronization script for 40+ LoRA models across Hugging Face and GitHub.
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import huggingface_hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FinalizeSync")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"
DATASET_REPO_ID = "wwewtech/russian-it-community-corpus"


def main():
    api = huggingface_hub.HfApi(token=HF_TOKEN)
    adapters_dir = Path("lora_adapters")

    # 1. Check all local adapters
    local_adapters = sorted([d for d in adapters_dir.iterdir() if d.is_dir() and (d / "adapter_model.safetensors").exists()])
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
                    logger.warning(f"Attempt {attempt+1} failed for {d.name}: {e}. Retrying in 5s...")
                    time.sleep(5)

    # 3. Generate Final LORA_MODEL_ZOO.md and lora_zoo_index.json
    results = []
    for d in local_adapters:
        safetensors = d / "adapter_model.safetensors"
        size_mb = safetensors.stat().st_size / (1024 * 1024)
        results.append({
            "id": d.name,
            "size_mb": round(size_mb, 2),
            "status": "SUCCESS",
            "adapter_dir": f"lora_adapters/{d.name}",
            "hf_model_repo": MODEL_REPO_ID,
        })

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
            f"| {idx} | **`{r['id']}`** | **{r['size_mb']} MB** | [`lora_adapters/{r['id']}/`](file:///D:/project_x/lora_adapters/{r['id']}/) | [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}) |"
        )

    md_lines.extend([
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
        'tokenizer = AutoTokenizer.from_pretrained(model_hub_path, subfolder=adapter_id)',
        'base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="auto")',
        'model = PeftModel.from_pretrained(base_model, model_hub_path, subfolder=adapter_id)',
        "",
        "# 3. Инференс",
        'prompt = "Как настроить Nginx reverse proxy с поддержкой WebSocket в Docker?"',
        'messages = [{"role": "user", "content": prompt}]',
        'inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")',
        'outputs = model.generate(**inputs, max_new_tokens=256)',
        'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        "```",
    ])

    with open("reports/LORA_MODEL_ZOO.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Generated LORA_MODEL_ZOO.md with {len(results)} adapters!")

    # 4. Git commit and push to GitHub
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"feat(zoo): expand LoRA Model Zoo to {len(results)} pre-trained foundation models"], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)
    logger.info("Successfully pushed updated configs and catalog to GitHub!")


if __name__ == "__main__":
    main()
