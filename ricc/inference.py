"""
Interactive CLI Chat & Unified Inference Engine.

Supports base models and registry-resolved LoRA adapters. The default is the
verified full-epoch adapter from ``lora_adapters/registry.json``; local
weights or an immutable Hub subfolder/revision are used as appropriate.
Historical local adapters remain explicitly experimental.

The module exposes three layers:

* :func:`validate_adapter_path` / :func:`build_chat_messages` /
  :func:`build_prompt` / :func:`is_exit_command` — pure helpers covered by
  :mod:`tests.test_inference` so failures surface in CI even on machines
  without a GPU/transformers stack.
* :func:`interactive_chat_session` — the I/O loop. Tries to load the model
  and tokenizer; raises :class:`RuntimeError` on adapter / weight corruption
  rather than silently downgrading to the base model.
* :func:`main` — argparse wrapper used by ``python -m ricc.inference``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ricc.bootstrap import setup_runtime_env

# Heavy third-party imports are deferred to interactive_chat_session() so
# that test modules (which only need the pure helpers) do not require
# torch / transformers / peft at collection time.


EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


def is_exit_command(text: str) -> bool:
    """Return True if ``text`` is empty / whitespace-only or matches a session-exit keyword.

    Centralised here so the same definition is used by the interactive
    loop and by the unit tests in :mod:`tests.test_inference`.
    """
    if not text or not text.strip():
        return True
    return text.strip().lower() in EXIT_COMMANDS


def validate_adapter_path(adapter_id: str, lora_root: Path) -> Path:
    """Resolve and validate a local LoRA adapter directory.

    Returns the absolute :class:`~pathlib.Path` to the adapter directory.

    Raises:
        RuntimeError: if the directory is missing, or if the expected
            ``adapter_model.safetensors`` weight file is absent.
    """
    adapter_path = (lora_root / adapter_id).resolve()
    if not adapter_path.exists():
        raise RuntimeError(
            f"LoRA adapter directory not found: {adapter_path}. "
            "Run `python scripts/generate_lora_registry.py` to regenerate, "
            "or pass adapter_id=None to use the base model."
        )
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise RuntimeError(
            f"LoRA adapter weights missing at {adapter_path}/adapter_model.safetensors. "
            "The adapter directory exists but weights are absent or corrupt."
        )
    return adapter_path


def build_chat_messages(query: str, rag_context: str = "") -> list[dict[str, str]]:
    """Build the ChatML-style messages list fed to ``tokenizer.apply_chat_template``.

    When ``rag_context`` is non-empty the system prompt is augmented with
    the retrieved knowledge-base chunks; otherwise the model receives a
    plain "senior engineer" persona prompt.
    """
    if rag_context:
        system_prompt = (
            "Ты — старший ведущий архитектор и инженер русскоязычного IT-сообщества. "
            "Используй предоставленный контекст базы знаний для точного, лаконичного ответа "
            "с примерами кода и архитектурными деталями.\n\n"
            f"КОНТЕКСТ БАЗЫ ЗНАНИЙ:\n{rag_context}"
        )
    else:
        system_prompt = "Ты — опытный инженер и архитектор программных систем. Дай точный и профессиональный ответ."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def build_prompt(messages: list[dict[str, str]], apply_template: Any) -> str:
    """Render a chat-template prompt, falling back to explicit delimiters.

    ``apply_template`` is normally ``tokenizer.apply_chat_template``. We
    accept it as a parameter so unit tests can substitute a stub that
    raises (simulating Qwen / ChatGLM-family models whose tokenizers
    reject unknown message roles) and verify the fallback path is used.
    """
    try:
        return str(apply_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        # Reconstruct using explicit <|im_start|> delimiters; without them,
        # Qwen / ChatGLM-family models treat everything as a single prompt
        # and ignore the system block.
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        return (
            "<|im_start|>system\n" + system_content + "<|im_end|>\n"
            "<|im_start|>user\n" + user_content + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def format_rag_context(hits: list[dict[str, Any]]) -> str:
    """Format a list of RAG hits into a single context string for the system prompt."""
    if not hits:
        return ""
    return "\n---\n".join(f"[{h.get('domain', 'Tech')}]: {h.get('content', '')}" for h in hits)


def _resolve_inference_adapter(adapter_id: str, root: Path) -> Any:
    """Resolve registry metadata, preserving the explicit legacy path contract."""
    from ricc.lora.production import resolve_adapter

    local_dir = root / "lora_adapters" / adapter_id
    if not (local_dir / "adapter_model.safetensors").is_file():
        return validate_adapter_path(adapter_id, root / "lora_adapters")
    return resolve_adapter(adapter_id, root=root)


def interactive_chat_session(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    adapter_id: str | None = "smollm2_135m_instruct",
    use_rag: bool = True,
    max_tokens: int = 512,
) -> None:
    """Run the interactive CLI chat loop.

    Heavy imports are deferred inside the function so importing this
    module from tests does not require torch / transformers.
    """
    # Deferred imports keep test collection light and let the pure
    # helpers above be tested in CPU-only environments.
    from transformers import TextStreamer

    from ricc.ml_runtime import load_adapter_runtime, move_inputs_to_model
    from ricc.rag.rag_pipeline import LocalRAGPipeline
    from ricc.settings import settings

    print("=" * 70)
    print("Russian IT Community LLM & RAG Interactive Terminal")
    print(f"Base Model: {model_name}")
    print(f"LoRA Adapter: {adapter_id or 'None (Base Model)'}")
    print(f"RAG Pipeline: {'ON' if use_rag else 'OFF'}")
    print("=" * 70)
    print("Type your technical question (or 'exit' / 'quit' to end session):\n")

    # 1. Load RAG
    rag_kb = None
    if use_rag:
        rag_path = settings.PARQUET_OUTPUT_DIR / "rag_knowledge_base.parquet"
        if rag_path.exists():
            print("Initializing Local RAG Knowledge Base...")
            rag_kb = LocalRAGPipeline(rag_path)
            print(f"RAG Engine ready ({len(rag_kb.df_kb):,} chunks indexed).")
        else:
            print("RAG Parquet knowledge base not found. Running without RAG.")

    # 2. Resolve and load the exact adapter and its compatible base. An
    # explicitly different --model is a research choice, not a silent fallback.
    print("Loading model and tokenizer...")
    if adapter_id:
        spec = _resolve_inference_adapter(adapter_id, settings.BASE_DIR)
        from ricc.lora.production import AdapterSpec

        if isinstance(spec, AdapterSpec) and spec.base_model != model_name:
            print(f"Using adapter-compatible base {spec.base_model} instead of requested {model_name}")
        from ricc.ml_runtime import attach_lora_adapter, load_tokenizer_and_base_model

        if isinstance(spec, AdapterSpec):
            tokenizer, model = load_adapter_runtime(spec)
            print(f"Attached adapter {spec.hub_reference} ({spec.source})")
        else:
            tokenizer, model = load_tokenizer_and_base_model(model_name)
            model = attach_lora_adapter(model, spec, model_name=model_name, adapter_id=adapter_id, verify_files=False)
            print(f"Attached legacy local adapter {spec}")
    else:
        from ricc.ml_runtime import load_tokenizer_and_base_model

        tokenizer, model = load_tokenizer_and_base_model(model_name)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            query = input("\nUser >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if is_exit_command(query):
            print("Exiting chat session.")
            break

        rag_context = ""
        if rag_kb:
            hits = rag_kb.search(query, top_k=2)
            if hits:
                rag_context = format_rag_context(hits)
                print(f"\n[RAG Retrieved {len(hits)} Context Chunks]")

        messages = build_chat_messages(query, rag_context)
        prompt_text = build_prompt(messages, tokenizer.apply_chat_template)

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = move_inputs_to_model(inputs, model)

        print("\nAssistant >> ", end="", flush=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
            )
        print()


def main() -> None:
    setup_runtime_env(json_logs=False)
    parser = argparse.ArgumentParser(description="Russian IT Community LLM & RAG Inference CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Base model for base-only inference; a registry adapter selects its own compatible base",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="smollm2_135m_instruct",
        help="Registry adapter ID (default: verified full-epoch production adapter)",
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG knowledge augmentation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated response tokens")
    args = parser.parse_args()

    interactive_chat_session(
        model_name=args.model,
        adapter_id=None if args.adapter.lower() in ("none", "") else args.adapter,
        use_rag=not args.no_rag,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
