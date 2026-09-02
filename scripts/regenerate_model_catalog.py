"""Regenerate the LoRA Model Zoo catalog and the HF model card from live Hub data.

Ground truth = actual adapter folders in wwewtech/russian-it-community-lora.
Run:  python scripts/regenerate_model_catalog.py
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO = "wwewtech/russian-it-community-lora"
API = f"https://huggingface.co/api/{'models'}/{REPO}"
FLAGSHIPS = {"heavyweight_qwen2.5_coder_7b", "heavyweight_deepseek_r1_7b", "heavyweight_llama3.1_8b"}


def fetch_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def main():
    data = fetch_json(API)
    subfolders = sorted(
        {s["rfilename"].split("/")[0] for s in data.get("siblings", []) if "/" in s["rfilename"]} - {"models"}
    )

    def fetch_base(sub: str) -> dict:
        with urllib.request.urlopen(
            f"https://huggingface.co/{REPO}/resolve/main/{sub}/adapter_config.json", timeout=30
        ) as raw:
            base = json.load(raw).get("base_model_name_or_path") or "?"
        return {"id": sub, "base": base}

    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(fetch_base, subfolders))

    total = len(rows)
    flagship_count = sum(1 for r in rows if r["id"] in FLAGSHIPS)
    domain_count = total - flagship_count
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d")

    # 1. Generate HF_MODEL_CARD.md (for Hugging Face Hub Model Card)
    hf_lines = [
        "---",
        "license: mit",
        "language:",
        "- ru",
        "- en",
        "library_name: peft",
        "tags:",
        "- russian",
        "- lora",
        "- qlora",
        "- peft",
        "- sft",
        "- text-generation",
        "- russian-nlp",
        "---",
        "",
        "# 🦁 Russian IT Community LoRA Model Zoo",
        "",
        f"**{total} pre-trained adapters** ({domain_count} domain adapters + {flagship_count} flagship 7B–8B QLoRA), "
        "fine-tuned on the RICC corpus (2.91M messages, 171.5k curated SFT dialogues) "
        "for Russian-language IT discourse: backend, DevOps, AI/ML, infrastructure.",
        "",
        f"> Catalog regenerated from the Hub file tree on {generated_at}. "
        "Source of truth: the `siblings` listing of this repository.",
        "",
        "## ⚡ Quick Start: 3-Line Inference",
        "",
        "```python",
        "import torch",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        'model_id = "Qwen/Qwen2.5-1.5B-Instruct"          # any base model from the catalog',
        'adapter_id = "wwewtech/russian-it-community-lora"',
        'subfolder = "qwen2.5_1.5b_instruct"              # choose from the catalog below',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        'base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")',
        "model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)",
        "",
        'inputs = tokenizer("<|user|>\\nКак настроить репликацию PostgreSQL?\\n<|assistant|>\\n", return_tensors="pt").to(model.device)',
        "outputs = model.generate(**inputs, max_new_tokens=256)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
        f"## 📚 Full Catalog ({total} Adapters)",
        "",
        "| # | Adapter Subfolder | Base Model | Hub Link |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for i, r in enumerate(rows, 1):
        hf_lines.append(
            f"| {i:02d} | `{r['id']}` | `{r['base']}` "
            f"| [`{r['id']}/`](https://huggingface.co/{REPO}/tree/main/{r['id']}) |"
        )
    hf_lines += [
        "",
        "## 🥇 Flagship QLoRA Models (7B–8B)",
        "",
        "Full-precision copies also live under [`models/`](https://huggingface.co/"
        f"{REPO}/tree/main/models): `models/heavyweight_qwen2.5_coder_7b`, "
        "`models/heavyweight_deepseek_r1_7b`, `models/heavyweight_llama3.1_8b`.",
        "",
        "## 📓 Training Data & Evaluation Status",
        "",
        "- Training corpus: [RICC SFT Dialogues](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus) "
        "(171,520 multi-turn dialogues).",
        "- Academic benchmark numbers (HumanEval / RuMMLU / PPL) published earlier are **withdrawn pending "
        "re-evaluation**: the harness had answer-parsing and column-mapping defects that produced implausible "
        "values (see repo commit history). Enterprise scenario scores are rubric-based heuristics, not "
        "capability measurements.",
        "",
    ]

    card = Path("reports/HF_MODEL_CARD.md")
    card.write_text("\n".join(hf_lines), encoding="utf-8")

    # 2. Generate LORA_MODEL_ZOO.md (for local GitHub repository documentation)
    zoo_lines = [
        "# 🦁 Russian IT Community LoRA Model Zoo & Local Adapter Catalog",
        "",
        f"Официальный каталог **{total} предварительно обученных LoRA-адаптеров** ({domain_count} доменных адаптеров + {flagship_count} флагманских QLoRA моделей 7B–8B), "
        "дообученных на корпусе **RICC (2.82M очищенных сообщений, 171.5k диалогов)** для русскоязычного IT-дискурса, бэкенда, DevOps, AI/ML и системного администрирования.",
        "",
        "Все адаптеры доступны как локально в каталоге [`lora_adapters/`](../lora_adapters/), так и на Hugging Face Hub: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora).",
        "",
        "---",
        "",
        "## 💻 Локальный запуск через CLI и Python",
        "",
        "### 1. Запуск интерактивного терминала (Inference CLI):",
        "```bash",
        "# Запуск Qwen 2.5 1.5B с доменным LoRA-адаптером и локальным RAG",
        "python src/inference.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter qwen2.5_1.5b_instruct",
        "",
        "# Запуск флагманской 7B модели в 4-битном режиме (VRAM <= 6 GB)",
        "python src/inference.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --adapter heavyweight_qwen2.5_coder_7b",
        "```",
        "",
        "### 2. Запуск в Python коде:",
        "```python",
        "import torch",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        "# 1. Загрузка базовой модели",
        'model_id = "Qwen/Qwen2.5-1.5B-Instruct"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        'base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")',
        "",
        "# 2. Подключение локального адаптера",
        'model = PeftModel.from_pretrained(base_model, "lora_adapters/qwen2.5_1.5b_instruct")',
        "",
        "# 3. Генерация ответа",
        'prompt = "Как настроить Nginx reverse proxy с поддержкой WebSocket в Docker?"',
        'messages = [{"role": "user", "content": prompt}]',
        'inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")',
        "outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
        "---",
        "",
        f"## 📊 Полный каталог адаптеров ({total} моделей)",
        "",
        "| # | Идентификатор адаптера | Базовая модель | Локальный каталог | Hugging Face Hub |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]
    for i, r in enumerate(rows, 1):
        zoo_lines.append(
            f"| {i:02d} | `{r['id']}` | `{r['base']}` "
            f"| [`lora_adapters/{r['id']}/`](../lora_adapters/{r['id']}/) "
            f"| [`{r['id']}/`](https://huggingface.co/{REPO}/tree/main/{r['id']}) |"
        )
    zoo_lines += [
        "",
        "---",
        "",
        "## ⚙️ Аппаратные требования и воспроизводимость",
        "",
        "- **Потребление VRAM при обучении:** ~4.35 GB на NVIDIA GeForce RTX 3060 (12 GB) с gradient accumulation = 4, batch size = 1.",
        "- **Флагманские 7B–8B модели:** используют 4-битное квантование BitsAndBytes (NF4) для инференса в пределах 6 GB VRAM.",
        "- **Статус метрик:** ранее опубликованные академические метрики (HumanEval / RuMMLU / PPL) **отозваны до переоценки** — аудит тестового контура выявил дефекты парсинга ответов и маппинга колонок (см. историю коммитов). Оценки enterprise-сценариев являются эвристиками на основе рубрик, а не измерениями способностей.",
        "",
    ]

    out_md = Path("reports/LORA_MODEL_ZOO.md")
    out_md.write_text("\n".join(zoo_lines), encoding="utf-8")

    index = {
        "repo_id": REPO,
        "total_adapters": total,
        "flagships": sorted(FLAGSHIPS),
        "adapters": [{"id": r["id"], "base_model": r["base"]} for r in rows],
        "generated_at_utc": generated_at,
    }
    Path("reports/lora_zoo_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_md}, {card}, lora_zoo_index.json; total adapters: {total}")


if __name__ == "__main__":
    main()
