"""Pure (Streamlit-free) logic for the LoRA Chat UI page.

The chat page in ``app.py`` renders the widgets; everything that must be
unit-testable without torch/streamlit lives here:

* :func:`list_chat_adapters` — which adapter directories can be loaded;
* :func:`adapter_summary` — the training provenance shown in the UI;
* :func:`build_chat_history_messages` — the message list fed to
  ``tokenizer.apply_chat_template`` (system persona + full history).

Model loading/generation itself stays in :mod:`ricc.inference` helpers and
is heavy-import-guarded; this module never imports torch at module scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ADAPTERS_DIRNAME = "lora_adapters"

CHAT_SYSTEM_PROMPT = (
    "Ты — опытный инженер и архитектор программных систем, воспитанный "
    "русскоязычным IT-сообществом. Отвечай по-русски, точным техническим языком, "
    "с примерами кода там, где они уместны."
)

# The adapter that satisfies the binding training policy (one full epoch,
# 100 % corpus coverage) is offered as the default chat partner.
POLICY_COMPLIANT_DEFAULT = "smollm2_135m_instruct"


def _adapters_root(base_dir: Path) -> Path:
    return base_dir / ADAPTERS_DIRNAME


def list_chat_adapters(base_dir: Path) -> list[str]:
    """Adapter ids that can actually be attached (weights + config present)."""
    root = _adapters_root(base_dir)
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "adapter_model.safetensors").is_file() and (entry / "adapter_config.json").is_file():
            out.append(entry.name)
    return out


def adapter_summary(base_dir: Path, adapter_id: str) -> dict[str, Any]:
    """Training provenance from ``training_meta.json`` for the UI sidebar.

    Returns a flat dict with display-ready keys; ``compliant`` is the policy
    gate verdict (False when the meta is missing/unreadable — the UI then
    labels the adapter PARTIAL instead of silently claiming full training).
    """
    meta_path = _adapters_root(base_dir) / adapter_id / "training_meta.json"
    summary: dict[str, Any] = {
        "base_model": None,
        "actual_steps": None,
        "required_steps": None,
        "corpus_coverage_pct": None,
        "train_dialogues": None,
        "train_loss": None,
        "eval_loss": None,
        "prompt_masking": None,
        "training_time_sec": None,
        "compliant": False,
    }
    if not meta_path.is_file():
        return summary
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return summary
    summary["base_model"] = meta.get("base_model")
    summary["actual_steps"] = meta.get("actual_steps")
    summary["required_steps"] = meta.get("required_steps_for_full_epoch")
    summary["corpus_coverage_pct"] = meta.get("corpus_coverage_pct")
    summary["train_dialogues"] = meta.get("train_dialogues")
    summary["train_loss"] = meta.get("train_loss")
    summary["eval_loss"] = meta.get("eval_loss")
    summary["prompt_masking"] = meta.get("prompt_masking")
    summary["training_time_sec"] = meta.get("training_time_sec")

    # Re-derive compliance the same way the release gate does, instead of
    # trusting the recorded status field.
    from ricc.lora.policy_gate import evaluate_compliance

    summary["compliant"] = evaluate_compliance(_adapters_root(base_dir) / adapter_id).compliant
    return summary


def build_chat_history_messages(
    history: list[dict[str, str]], system_prompt: str = CHAT_SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """ChatML messages for ``apply_chat_template``: system first, then history.

    The last message must be the user's new turn (the template is rendered
    with ``add_generation_prompt=True`` by the caller).
    """
    return [{"role": "system", "content": system_prompt}, *history]


def generate_reply(
    query: str,
    history: list[dict[str, str]],
    model_name: str,
    adapter_id: str | None,
    max_new_tokens: int = 384,
    temperature: float = 0.4,
    top_p: float = 0.9,
) -> str:
    """One-shot generation for the chat UI (no streaming, UI refreshes per turn).

    Heavy imports are deferred so tests can import this module without torch.
    Raises whatever the model stack raises (AdapterLoadError, OOM, ...) —
    the UI shows the error rather than silently answering from the base model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ricc.inference import validate_adapter_path

    # This module lives at the repository root, so the adapters tree is
    # a sibling of the module file (not parent.parent — that would point
    # outside the checkout when the repo is not one level below the drive).
    root_dir = Path(__file__).resolve().parent
    if not (root_dir / ADAPTERS_DIRNAME).is_dir():
        # Fallback for a packaged/installed layout: try the working directory.
        root_dir = Path.cwd()
    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if adapter_id:
        from peft import PeftModel

        adapter_path = validate_adapter_path(adapter_id, root_dir / ADAPTERS_DIRNAME)
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    full_history = [*history, {"role": "user", "content": query}]
    messages = build_chat_history_messages(full_history)

    prompt_text = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply: str = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()
