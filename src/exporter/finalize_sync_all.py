"""
Final synchronization script for 40+ LoRA models across Hugging Face and GitHub.

The module exposes three layers:

* :func:`compute_lora_zoo_index` — pure function that walks a local
  ``lora_adapters/`` directory and returns a serialisable list of adapter
  records. Covered by :mod:`tests.test_finalize_sync_all`.
* :func:`build_lora_zoo_markdown` — pure function that renders the
  Markdown catalogue (no I/O).
* :func:`upload_dataset` / :func:`upload_missing_adapters` — I/O wrappers
  around the ``huggingface_hub.HfApi`` client, mocked in tests.
* :func:`main` — argparse / env-var entry point used by
  ``scripts/sync_to_hub.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from src.bootstrap import setup_runtime_env

setup_runtime_env()

import huggingface_hub  # noqa: E402  (deferred after setup_runtime_env)

logger = logging.getLogger("FinalizeSync")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_REPO_ID = "wwewtech/russian-it-community-lora"
DATASET_REPO_ID = "wwewtech/russian-it-community-corpus"


def _local_adapters(adapters_dir: Path) -> list[Path]:
    """Return adapter directories that have a ``adapter_model.safetensors`` file."""
    return sorted(d for d in adapters_dir.iterdir() if d.is_dir() and (d / "adapter_model.safetensors").exists())


def compute_lora_zoo_index(
    adapters_dir: Path,
    model_repo_id: str = MODEL_REPO_ID,
) -> list[dict]:
    """Build a serialisable list of adapter records from the local ``lora_adapters/`` tree.

    Each record is suitable for both ``reports/lora_zoo_index.json`` and for
    rendering in :func:`build_lora_zoo_markdown`. Pure (no I/O beyond
    ``Path.stat``), so it is fully covered by unit tests.
    """
    results: list[dict] = []
    for d in _local_adapters(Path(adapters_dir)):
        safetensors = d / "adapter_model.safetensors"
        size_mb = round(safetensors.stat().st_size / (1024 * 1024), 2)
        results.append(
            {
                "id": d.name,
                "size_mb": size_mb,
                "status": "SUCCESS",
                "adapter_dir": f"lora_adapters/{d.name}",
                "hf_model_repo": model_repo_id,
            }
        )
    return results


def build_lora_zoo_markdown(
    zoo_index: list[dict],
    *,
    model_repo_id: str = MODEL_REPO_ID,
    corpus_size_label: str = "2.91M сообщений, 171.5k диалогов",
) -> str:
    """Render the LoRA-Zoo catalogue as a Markdown string.

    Pure function: no I/O, no datetime.now(). All values are either
    arguments or derived from the ``zoo_index`` list.
    """
    lines: list[str] = [
        "# 🦁 Russian IT Community LoRA Model Zoo",
        "",
        f"**Официальный каталог {len(zoo_index)} предварительно обученных LoRA-адаптеров**, "
        f"дообученных на корпусе **RICC ({corpus_size_label})** для русскоязычного IT-дискурса, "
        "бэкенда, DevOps, AI/ML и инфраструктуры.",
        "",
        f"Все адаптеры доступны на Hugging Face Hub: [`{model_repo_id}`](https://huggingface.co/{model_repo_id}).",
        "",
        "---",
        "",
        "## 📊 Доступные LoRA-адаптеры",
        "",
        "| # | Идентификатор модели | Размер весов | Каталог адаптера | Hugging Face Hub |",
        "| :---: | :--- | :---: | :--- | :--- |",
    ]
    for idx, r in enumerate(zoo_index, 1):
        lines.append(
            f"| {idx} | **`{r['id']}`** | **{r['size_mb']} MB** "
            f"| [`lora_adapters/{r['id']}/`](lora_adapters/{r['id']}/) "
            f"| [`{model_repo_id}`](https://huggingface.co/{model_repo_id}) |"
        )
    lines.extend(
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
    return "\n".join(lines)


def upload_dataset(
    api: huggingface_hub.HfApi,
    *,
    dataset_repo_id: str = DATASET_REPO_ID,
) -> None:
    """Upload cleaned parquet files, dataset card, and reports to HF Hub.

    Failures are logged and re-raised so the caller (typically
    ``scripts/sync_to_hub.py``) can decide whether to retry or to
    surface a non-zero exit code.
    """
    logger.info("Uploading sanitized dataset artifacts to Hugging Face: %s...", dataset_repo_id)

    # 1. Dataset Card (README.md)
    dataset_card_path = Path("reports/DATASET_AND_ANALYTICS.md")
    if dataset_card_path.exists():
        api.upload_file(
            path_or_fileobj=str(dataset_card_path),
            path_in_repo="README.md",
            repo_id=dataset_repo_id,
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
            logger.info(
                "Uploading %s (%.2f MB) -> %s...",
                local_path,
                p.stat().st_size / (1024 * 1024),
                repo_path,
            )
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            logger.info("Successfully uploaded %s", repo_path)

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
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            logger.info("Uploaded %s", repo_path)


def upload_missing_adapters(
    api: huggingface_hub.HfApi,
    adapters_dir: Path,
    *,
    model_repo_id: str = MODEL_REPO_ID,
    max_attempts: int = 5,
    retry_sleep_seconds: float = 5.0,
) -> list[str]:
    """Upload adapters present locally but missing on the Hub.

    Returns the list of adapter names that were successfully uploaded.
    Raises the last exception if an upload fails after ``max_attempts``
    retries, so the caller can surface it instead of silently dropping
    the adapter.
    """
    local_adapters = _local_adapters(Path(adapters_dir))
    logger.info("Found %d trained LoRA adapters locally in %s", len(local_adapters), adapters_dir)

    hf_files = api.list_repo_files(repo_id=model_repo_id, repo_type="model")
    uploaded_folders = {f.split("/")[0] for f in hf_files if "/" in f}

    uploaded: list[str] = []
    for d in local_adapters:
        if d.name in uploaded_folders:
            continue
        logger.info("Uploading missing adapter %s to Hugging Face Model Hub...", d.name)
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                api.upload_folder(
                    folder_path=str(d),
                    path_in_repo=d.name,
                    repo_id=model_repo_id,
                    repo_type="model",
                    ignore_patterns=["checkpoint-*"],
                )
                logger.info("Successfully uploaded %s!", d.name)
                uploaded.append(d.name)
                last_exc = None
                break
            except Exception as e:  # pragma: no cover - I/O failure path
                last_exc = e
                logger.warning(
                    "Attempt %d failed for %s: %s. Retrying in %.1fs...",
                    attempt + 1,
                    d.name,
                    e,
                    retry_sleep_seconds,
                )
                time.sleep(retry_sleep_seconds)
        if last_exc is not None:
            raise RuntimeError(
                f"Failed to upload adapter {d.name} after {max_attempts} attempts: {last_exc}"
            ) from last_exc
    return uploaded


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    api = huggingface_hub.HfApi(token=HF_TOKEN if HF_TOKEN else None)
    adapters_dir = Path("lora_adapters")

    # 1. Dataset artefacts
    try:
        upload_dataset(api)
    except Exception as e:
        logger.warning("Dataset upload to Hugging Face skipped or failed: %s", e)

    # 2. Missing LoRA adapters on the Model Hub
    try:
        upload_missing_adapters(api, adapters_dir)
    except Exception as e:
        logger.warning("LoRA adapter sync to Hugging Face skipped or failed: %s", e)

    # 3. Render and persist the local LoRA-Zoo catalogue (works even
    # when the Hub sync failed, so the GitHub release is always updated).
    zoo_index = compute_lora_zoo_index(adapters_dir)
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/lora_zoo_index.json", "w", encoding="utf-8") as f:
        json.dump(zoo_index, f, ensure_ascii=False, indent=2)
    md = build_lora_zoo_markdown(zoo_index)
    with open("reports/LORA_MODEL_ZOO.md", "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Generated LORA_MODEL_ZOO.md with %d adapters!", len(zoo_index))


if __name__ == "__main__":
    main()
