"""
Inference Demo with Trained Russian IT LoRA Adapter.
Run: python src/lora/generate_demo.py --prompt "Как настроить прием платежей для SaaS из РФ?"
"""

import argparse

from src.bootstrap import setup_runtime_env

setup_runtime_env()

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def generate_answer(
    prompt: str = "Как настроить прием платежей для SaaS сервиса из РФ в 2026 году?",
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "lora_adapters/russian_it_lora",
):
    print(f"\n🚀 Загрузка базовой модели: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"📦 Подключение обученного LoRA адаптера из: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    messages = [
        {"role": "system", "content": "Ты опытный IT-архитектор и технический фаундер сообщества."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print(f"\n❓ Вопрос: {prompt}")
    print("⏳ Генерация ответа модели...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(f"🤖 Ответ обученной LoRA модели:\n{response}")
    print("=" * 60 + "\n")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Как настроить прием платежей для SaaS из РФ?")
    args = parser.parse_args()
    generate_answer(prompt=args.prompt)
