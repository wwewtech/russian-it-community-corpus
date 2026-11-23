"""Run the verified production adapter with an optional custom prompt.

Example: ``python ricc/lora/generate_demo.py --prompt "Как настроить Nginx?"``
Use ``--adapter <registry-id>`` for an explicitly experimental local adapter;
the default is resolved from the policy-compliant production registry entry.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, cast

import torch

from ricc.bootstrap import setup_runtime_env
from ricc.lora.production import resolve_adapter
from ricc.ml_runtime import generate_chat_completion, load_adapter_runtime

logger = logging.getLogger(__name__)


class _LazyTokenizer:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        from transformers import AutoTokenizer as real_tokenizer

        return cast(Any, real_tokenizer).from_pretrained(*args, **kwargs)


class _LazyBaseModel:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        from transformers import AutoModelForCausalLM as real_model

        return real_model.from_pretrained(*args, **kwargs)


class _LazyPeft:
    def __getattr__(self, name: str) -> Any:
        from peft import PeftModel

        return getattr(PeftModel, name)


AutoTokenizer = _LazyTokenizer
AutoModelForCausalLM = _LazyBaseModel
PeftModel = _LazyPeft()


def _load_legacy_local_pair(base_model_name: str | None, adapter_path: str) -> tuple[Any, Any]:
    """Compatibility path for the historical explicit base/local adapter API."""
    base = base_model_name or "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer: Any = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


def generate_answer(
    prompt: str = "Как настроить прием платежей для SaaS сервиса из РФ в 2026 году?",
    base_model_name: str | None = None,
    adapter_path: str | None = None,
    *,
    adapter_id: str = "smollm2_135m_instruct",
) -> str:
    """Generate using a registry adapter or the legacy explicit base/path pair."""
    if base_model_name is not None or adapter_path is not None:
        tokenizer, model = _load_legacy_local_pair(base_model_name, adapter_path or adapter_id)
    else:
        spec = resolve_adapter(adapter_id)
        tokenizer, model = load_adapter_runtime(spec)
    model.eval()
    messages = [
        {"role": "system", "content": "Ты опытный IT-архитектор и технический фаундер сообщества."},
        {"role": "user", "content": prompt},
    ]
    if base_model_name is not None or adapter_path is not None:
        # Preserve the old list-based mock API while sharing production
        # tensor semantics in the registry path above.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return cast(str, tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))

    response, _elapsed, _tps = generate_chat_completion(
        model,
        tokenizer,
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        max_input_length=None,
    )
    return response


def main() -> None:
    setup_runtime_env(json_logs=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default="Как настроить прием платежей для SaaS из РФ?")
    parser.add_argument("--adapter", default="smollm2_135m_instruct", help="Registry adapter ID")
    args = parser.parse_args()
    with torch.no_grad():
        response = generate_answer(prompt=args.prompt, adapter_id=args.adapter)
    print("\n" + "=" * 60)
    print(f"Ответ обученной LoRA модели:\n{response}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
