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
        {s["rfilename"].split("/")[0] for s in data.get("siblings", []) if "/" in s["rfilename"]}
        - {"models"}
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

    lines = [
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
        "base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=\"auto\")",
        "model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)",
        "",
        'inputs = tokenizer("<|user|>\\nКак настроить репликацию PostgreSQL?<|assistant|>\\n", return_tensors="pt").to(model.device)',
        "outputs = model.generate(**inputs, max_new_tokens=256)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
        f"## 📚 Full Catalog ({total} Adapters)",
        "",
        "| # | Adapter Subfolder | Base Model | Link |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i:02d} | `{r['id']}` | `{r['base']}` "
            f"| [`{r['id']}/`](https://huggingface.co/{REPO}/tree/main/{r['id']}) |"
        )
    lines += [
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
        "- Academic benchmark numbers (HumanEval / RuMMLU / PPL) published earlier are **withdrawn pending re-evaluation**: "
        "the harness had answer-parsing and column-mapping defects that produced implausible values "
        "(see repo commit history). Enterprise scenario scores are rubric-based heuristics, not capability measurements.",
        "",
    ]

    out_md = Path("reports/LORA_MODEL_ZOO.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    card = Path("reports/HF_MODEL_CARD.md")
    card.write_text("\n".join(lines), encoding="utf-8")

    index = {"repo_id": REPO, "total_adapters": total, "flagships": sorted(FLAGSHIPS),
             "adapters": [{"id": r["id"], "base_model": r["base"]} for r in rows],
             "generated_at_utc": generated_at}
    Path("reports/lora_zoo_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_md}, {card}, lora_zoo_index.json; total adapters: {total}")


if __name__ == "__main__":
    main()
