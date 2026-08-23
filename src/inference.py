"""
Interactive CLI Chat & Unified Inference Engine.
Supports Base Models, 44+ LoRA Adapters, Flagship 7B-8B QLoRA, and Local RAG Pipeline.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from src.rag.rag_pipeline import LocalRAGPipeline


def interactive_chat_session(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_id: Optional[str] = "heavyweight_qwen2.5_coder_7b",
    use_rag: bool = True,
    max_tokens: int = 512,
):
    print("=" * 70)
    print("🤖 Russian IT Community LLM & RAG Interactive Terminal")
    print(f"📦 Base Model: {model_name}")
    print(f"🦁 LoRA Adapter: {adapter_id or 'None (Base Model)'}")
    print(f"🔍 RAG Pipeline: {'ON (325.7k knowledge chunks)' if use_rag else 'OFF'}")
    print("=" * 70)
    print("Type your technical question (or 'exit' / 'quit' to end session):\n")

    # 1. Load RAG
    rag_kb = None
    if use_rag:
        rag_path = Path("dataset_output/parquet/rag_knowledge_base.parquet")
        if rag_path.exists():
            print("🔍 Initializing Local RAG Knowledge Base...")
            rag_kb = LocalRAGPipeline(rag_path)
            print(f"✅ RAG Engine ready ({len(rag_kb.df):,} chunks indexed).")
        else:
            print("⚠️ RAG Parquet knowledge base not found. Running without RAG.")

    # 2. Load Model & Tokenizer
    print("🧠 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # 3. Attach LoRA Adapter
    if adapter_id:
        adapter_path = Path(f"lora_adapters/{adapter_id}")
        if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path))
                print(f"✅ Attached LoRA Adapter from {adapter_path}")
            except Exception as e:
                print(f"⚠️ Could not load LoRA adapter: {e}")
        else:
            print(f"⚠️ Adapter directory '{adapter_path}' not found. Using base weights.")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            query = input("\n💻 User >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting chat session.")
            break

        if not query or query.lower() in ("exit", "quit", "q"):
            print("👋 Exiting chat session.")
            break

        rag_context = ""
        if rag_kb:
            hits = rag_kb.search(query, top_k=2)
            if hits:
                rag_context = "\n---\n".join([f"[{h.get('domain', 'Tech')}]: {h.get('content', '')}" for h in hits])
                print(f"\n[🔍 RAG Retrieved {len(hits)} Context Chunks]")

        if rag_context:
            system_prompt = (
                "Ты — старший ведущий архитектор и инженер русскоязычного IT-сообщества. "
                "Используй предоставленный контекст базы знаний для точного, лаконичного ответа с примерами кода и архитектурными деталями.\n\n"
                f"КОНТЕКСТ БАЗЫ ЗНАНИЙ:\n{rag_context}"
            )
        else:
            system_prompt = "Ты — опытный инженер и архитектор программных систем. Дай точный и профессиональный ответ."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"{system_prompt}\n\nПользователь: {query}\nОтвет:"

        inputs = tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print("\n🤖 Assistant >> ", end="", flush=True)
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


def main():
    parser = argparse.ArgumentParser(description="Russian IT Community LLM & RAG Inference CLI")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--adapter", type=str, default="heavyweight_qwen2.5_coder_7b", help="LoRA Adapter ID in lora_adapters/")
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
