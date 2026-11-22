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
import re
import time
from pathlib import Path
from typing import Any

import huggingface_hub

from ricc.bootstrap import setup_runtime_env
from ricc.lora.policy_gate import evaluate_compliance, is_publishable
from ricc.settings import settings

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
) -> list[dict[str, Any]]:
    """Build a serialisable list of adapter records from the local ``lora_adapters/`` tree.

    Each record is suitable for both ``reports/lora_zoo_index.json`` and for
    rendering in :func:`build_lora_zoo_markdown`. Pure (no I/O beyond
    ``Path.stat``), so it is fully covered by unit tests.
    """
    results: list[dict[str, Any]] = []
    for d in _local_adapters(Path(adapters_dir)):
        if not is_publishable(d):
            logger.warning("Excluding policy-incompliant adapter from the release catalog: %s", d.name)
            continue
        safetensors = d / "adapter_model.safetensors"
        config = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        base_model_revision = str(config.get("revision") or "")
        hf_model_revision = str(config.get("hf_model_revision") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", base_model_revision):
            raise RuntimeError(f"Missing immutable base-model revision for {d.name}")
        if not re.fullmatch(r"[0-9a-f]{40}", hf_model_revision):
            raise RuntimeError(f"Missing immutable adapter revision for {d.name}")
        size_mb = round(safetensors.stat().st_size / (1024 * 1024), 2)
        results.append(
            {
                "id": d.name,
                "base_model": str(config.get("base_model_name_or_path") or "unknown"),
                "base_model_revision": base_model_revision,
                "hf_model_revision": hf_model_revision,
                "size_mb": size_mb,
                "status": "SUCCESS",
                "adapter_dir": f"lora_adapters/{d.name}",
                "hf_model_repo": model_repo_id,
            }
        )
    return results


def build_lora_zoo_markdown(
    zoo_index: list[dict[str, Any]],
    *,
    model_repo_id: str = MODEL_REPO_ID,
    corpus_size_label: str = "2.82M сообщений, 171.5k диалогов",
) -> str:
    """Render the LoRA-Zoo catalogue as a Markdown string.

    Pure function: no I/O, no datetime.now(). All values are either
    arguments or derived from the ``zoo_index`` list.
    """
    lines: list[str] = [
        "# Russian IT Community LoRA Model Zoo",
        "",
        f"**Официальный каталог {len(zoo_index)} предварительно обученных LoRA-адаптеров**, "
        f"дообученных на корпусе **RICC ({corpus_size_label})** для русскоязычного IT-дискурса, "
        "бэкенда, DevOps, AI/ML и инфраструктуры.",
        "",
        f"Все адаптеры доступны на Hugging Face Hub: [`{model_repo_id}`](https://huggingface.co/{model_repo_id}).",
        "",
        "---",
        "",
        "## Доступные LoRA-адаптеры",
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
    if zoo_index and all(
        r.get("base_model")
        and re.fullmatch(r"[0-9a-f]{40}", str(r.get("base_model_revision", "")))
        and re.fullmatch(r"[0-9a-f]{40}", str(r.get("hf_model_revision", "")))
        for r in zoo_index
    ):
        selected = zoo_index[0]
        quickstart_lines = [
            f'subfolder = "{selected["id"]}"',
            f'model_id = "{selected["base_model"]}"',
            f'model_hub_path = "{selected.get("hf_model_repo", model_repo_id)}"',
            f'base_revision = "{selected["base_model_revision"]}"',
            f'adapter_revision = "{selected["hf_model_revision"]}"',
        ]
    else:
        # Incomplete records must not be turned into runnable release examples.
        quickstart_lines = [
            "# Missing verified adapter or immutable revisions; resolve the registry first.",
            'raise RuntimeError("No verified full-epoch adapter with immutable revisions is available")',
        ]
    lines.extend(
        [
            "",
            "---",
            "",
            "## Быстрый старт: запуск проверенного full-epoch адаптера",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            *quickstart_lines,
            "tokenizer = AutoTokenizer.from_pretrained(model_hub_path, subfolder=subfolder, revision=adapter_revision)",
            'base = AutoModelForCausalLM.from_pretrained(model_id, revision=base_revision, device_map="auto", torch_dtype="auto")',
            "model = PeftModel.from_pretrained(base, model_hub_path, subfolder=subfolder, revision=adapter_revision)",
            "",
            'messages = [{"role": "user", "content": "Как настроить Nginx reverse proxy в Docker?"}]',
            'inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)',
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
    repo_root: Path | None = None,
) -> None:
    """Upload cleaned parquet files, dataset card, and reports to HF Hub.

    Failures are logged and re-raised so the caller (typically
    ``scripts/sync_to_hub.py``) can decide whether to retry or to
    surface a non-zero exit code.

    ``repo_root`` overrides the repository root used to resolve the
    artefact paths; it defaults to the checkout this module lives in, so
    the function no longer depends on the current working directory.
    """
    logger.info("Uploading sanitized dataset artifacts to Hugging Face: %s...", dataset_repo_id)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    # 0. Provenance gate: the published artefacts must match the signed
    # manifest snapshot. Without this check the release could publish a
    # dataset that was modified after the manifest was created — the
    # "provenance gate" in the SLO job compares a manifest with the very
    # files it was just created from, so it always passes.
    manifest_path = repo_root / "reports" / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Refusing to publish without the verified artifact manifest: {manifest_path}")
    try:
        from ricc.validation.artifact_manifest import verify_manifest

        # Verify every canonical parquet before the first Hub write, including
        # the dataset card. A missing or stale artifact must not cause a
        # partially published release.
        verify_manifest(repo_root, manifest_path)
        logger.info("Artifact manifest verified against all three published parquet files")
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to publish: the dataset artefacts do not match {manifest_path} ({exc}). "
            "Re-create the manifest for the artefacts that are about to be published."
        ) from exc

    # 1. Dataset Card (README.md)
    dataset_card_path = repo_root / "reports" / "DATASET_AND_ANALYTICS.md"
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
        p = repo_root / local_path
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
        p = repo_root / local_path
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            logger.info("Uploaded %s", repo_path)


def upload_model_card(
    api: huggingface_hub.HfApi,
    *,
    model_repo_id: str = MODEL_REPO_ID,
    repo_root: Path | None = None,
) -> bool:
    """Upload ``reports/HF_MODEL_CARD.md`` as the model repo README.

    Returns ``True`` when an upload happened. The card is generated by
    ``scripts/release_catalog.py``; the Hub write lives here so
    ADR 0001 keeps a single sanctioned upload path.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    card_path = repo_root / "reports" / "HF_MODEL_CARD.md"
    if not card_path.exists():
        logger.warning("Model card not found at %s; nothing to upload", card_path)
        return False
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=model_repo_id,
        repo_type="model",
    )
    logger.info("Uploaded model card (README.md) to %s", model_repo_id)
    return True


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
    # A metadata-only folder is not a published adapter. In particular, a
    # previous interrupted upload containing just README.md must be repaired,
    # not treated as complete forever.
    uploaded_folders = {f.split("/")[0] for f in hf_files if f.endswith("/adapter_model.safetensors")}

    uploaded: list[str] = []
    skipped: list[str] = []
    for d in local_adapters:
        # Policy gate: an adapter that does not satisfy the binding training
        # policy (partial epoch, time-boxed run, or no recorded provenance at
        # all) is never uploaded to the public zoo. It is skipped and logged
        # loudly instead of being published as a final adapter.
        if not is_publishable(d):
            report = evaluate_compliance(d)
            logger.warning(
                "Skipping adapter %s — it does not satisfy the training policy: %s",
                d.name,
                "; ".join(report.reasons),
            )
            skipped.append(d.name)
            continue
        if d.name in uploaded_folders:
            logger.info("Verified remote adapter weights already present: %s", d.name)
            continue
        logger.info("Uploading missing or incomplete adapter %s to Hugging Face Model Hub...", d.name)
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
    if skipped:
        logger.warning(
            "%d adapter(s) skipped for policy non-compliance: %s",
            len(skipped),
            ", ".join(skipped),
        )
    return uploaded


def main() -> None:
    setup_runtime_env()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required for a release; refusing anonymous or partial publication")
    api = huggingface_hub.HfApi(token=HF_TOKEN)
    api.whoami()
    for repo_id, repo_type in ((DATASET_REPO_ID, "dataset"), (MODEL_REPO_ID, "model")):
        # Fail before the first write when the token cannot access a target.
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    # Resolve the adapters directory relative to the REPOSITORY ROOT, not the
    # current working directory: the release job runs from the repo root, but
    # a CWD-dependent path silently produced an empty catalogue when the
    # script was invoked from elsewhere.
    adapters_dir = Path(__file__).resolve().parents[2] / "lora_adapters"

    # Verify the preflight catalog before the first remote write. This avoids
    # a partial HF publication when the release assets are not the ones tested.
    zoo_index = compute_lora_zoo_index(adapters_dir)
    if not zoo_index:
        raise RuntimeError(f"No verified adapter weights found in {adapters_dir}")
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = settings.REPORTS_DIR / "LORA_MODEL_ZOO.md"
    if not catalog_path.is_file():
        raise RuntimeError("Missing preflight-generated LORA_MODEL_ZOO.md; refusing release")
    catalog_text = catalog_path.read_text(encoding="utf-8")
    for record in zoo_index:
        for field in ("id", "base_model", "base_model_revision", "hf_model_revision"):
            if str(record[field]) not in catalog_text:
                raise RuntimeError(f"Release catalog omitted verified {field}: {record['id']}")

    # 1. Dataset artefacts — fail closed on any upload or integrity error.
    upload_dataset(api)

    # 2. Model repository card (README.md) — fail closed.
    upload_model_card(api)

    # 3. Missing LoRA adapters on the Model Hub. Policy-incompliant adapters
    # are skipped inside the function; historical Hub files are not deleted.
    upload_missing_adapters(api, adapters_dir)

    with open(settings.REPORTS_DIR / "lora_zoo_index.json", "w", encoding="utf-8") as f:
        json.dump(zoo_index, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
