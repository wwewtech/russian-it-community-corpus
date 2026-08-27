# 🦁 Russian IT Community LoRA Model Zoo & Local Adapter Catalog

Официальный каталог **58 предварительно обученных LoRA-адаптеров** (55 доменных адаптеров + 3 флагманских QLoRA моделей 7B–8B), дообученных на корпусе **RICC (2.82M очищенных сообщений, 171.5k диалогов)** для русскоязычного IT-дискурса, бэкенда, DevOps, AI/ML и системного администрирования.

Все адаптеры доступны как локально в каталоге [`lora_adapters/`](../lora_adapters/), так и на Hugging Face Hub: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora).

---

## 💻 Локальный запуск через CLI и Python

### 1. Запуск интерактивного терминала (Inference CLI):
```bash
# Запуск Qwen 2.5 1.5B с доменным LoRA-адаптером и локальным RAG
python src/inference.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter qwen2.5_1.5b_instruct

# Запуск флагманской 7B модели в 4-битном режиме (VRAM <= 6 GB)
python src/inference.py --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit --adapter heavyweight_qwen2.5_coder_7b
```

### 2. Запуск в Python коде:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Загрузка базовой модели
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 2. Подключение локального адаптера
model = PeftModel.from_pretrained(base_model, "lora_adapters/qwen2.5_1.5b_instruct")

# 3. Генерация ответа
prompt = "Как настроить Nginx reverse proxy с поддержкой WebSocket в Docker?"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Полный каталог адаптеров (58 моделей)

| # | Идентификатор адаптера | Базовая модель | Локальный каталог | Hugging Face Hub |
| :---: | :--- | :--- | :--- | :--- |
| 01 | `bloom_1b7` | `bigscience/bloom-1b7` | [`lora_adapters/bloom_1b7/`](../lora_adapters/bloom_1b7/) | [`bloom_1b7/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_1b7) |
| 02 | `bloom_560m` | `bigscience/bloom-560m` | [`lora_adapters/bloom_560m/`](../lora_adapters/bloom_560m/) | [`bloom_560m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_560m) |
| 03 | `codegen_350m_multi` | `Salesforce/codegen-350M-multi` | [`lora_adapters/codegen_350m_multi/`](../lora_adapters/codegen_350m_multi/) | [`codegen_350m_multi/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/codegen_350m_multi) |
| 04 | `deepseek_coder_1.3b_instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct` | [`lora_adapters/deepseek_coder_1.3b_instruct/`](../lora_adapters/deepseek_coder_1.3b_instruct/) | [`deepseek_coder_1.3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_coder_1.3b_instruct) |
| 05 | `deepseek_r1_distill_qwen_1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | [`lora_adapters/deepseek_r1_distill_qwen_1.5b/`](../lora_adapters/deepseek_r1_distill_qwen_1.5b/) | [`deepseek_r1_distill_qwen_1.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_r1_distill_qwen_1.5b) |
| 06 | `falcon3_1b_instruct` | `tiiuae/Falcon3-1B-Instruct` | [`lora_adapters/falcon3_1b_instruct/`](../lora_adapters/falcon3_1b_instruct/) | [`falcon3_1b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_1b_instruct) |
| 07 | `falcon3_3b_instruct` | `tiiuae/Falcon3-3B-Instruct` | [`lora_adapters/falcon3_3b_instruct/`](../lora_adapters/falcon3_3b_instruct/) | [`falcon3_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_3b_instruct) |
| 08 | `gemma_2_2b_it` | `unsloth/gemma-2-2b-it` | [`lora_adapters/gemma_2_2b_it/`](../lora_adapters/gemma_2_2b_it/) | [`gemma_2_2b_it/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gemma_2_2b_it) |
| 09 | `gpt2_large` | `openai-community/gpt2-large` | [`lora_adapters/gpt2_large/`](../lora_adapters/gpt2_large/) | [`gpt2_large/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_large) |
| 10 | `gpt2_medium` | `openai-community/gpt2-medium` | [`lora_adapters/gpt2_medium/`](../lora_adapters/gpt2_medium/) | [`gpt2_medium/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_medium) |
| 11 | `granite_3b_code_instruct` | `ibm-granite/granite-3b-code-instruct` | [`lora_adapters/granite_3b_code_instruct/`](../lora_adapters/granite_3b_code_instruct/) | [`granite_3b_code_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/granite_3b_code_instruct) |
| 12 | `heavyweight_deepseek_r1_7b` | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` | [`lora_adapters/heavyweight_deepseek_r1_7b/`](../lora_adapters/heavyweight_deepseek_r1_7b/) | [`heavyweight_deepseek_r1_7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_deepseek_r1_7b) |
| 13 | `heavyweight_llama3.1_8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | [`lora_adapters/heavyweight_llama3.1_8b/`](../lora_adapters/heavyweight_llama3.1_8b/) | [`heavyweight_llama3.1_8b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_llama3.1_8b) |
| 14 | `heavyweight_qwen2.5_coder_7b` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | [`lora_adapters/heavyweight_qwen2.5_coder_7b/`](../lora_adapters/heavyweight_qwen2.5_coder_7b/) | [`heavyweight_qwen2.5_coder_7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_qwen2.5_coder_7b) |
| 15 | `llama_3.2_1b_instruct` | `unsloth/Llama-3.2-1B-Instruct` | [`lora_adapters/llama_3.2_1b_instruct/`](../lora_adapters/llama_3.2_1b_instruct/) | [`llama_3.2_1b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_1b_instruct) |
| 16 | `llama_3.2_3b_instruct` | `unsloth/Llama-3.2-3B-Instruct` | [`lora_adapters/llama_3.2_3b_instruct/`](../lora_adapters/llama_3.2_3b_instruct/) | [`llama_3.2_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_3b_instruct) |
| 17 | `minicpm_2b_dpo` | `openbmb/MiniCPM-2B-dpo-bf16` | [`lora_adapters/minicpm_2b_dpo/`](../lora_adapters/minicpm_2b_dpo/) | [`minicpm_2b_dpo/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/minicpm_2b_dpo) |
| 18 | `opt_1.3b` | `facebook/opt-1.3b` | [`lora_adapters/opt_1.3b/`](../lora_adapters/opt_1.3b/) | [`opt_1.3b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_1.3b) |
| 19 | `opt_125m` | `facebook/opt-125m` | [`lora_adapters/opt_125m/`](../lora_adapters/opt_125m/) | [`opt_125m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_125m) |
| 20 | `opt_2.7b` | `facebook/opt-2.7b` | [`lora_adapters/opt_2.7b/`](../lora_adapters/opt_2.7b/) | [`opt_2.7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_2.7b) |
| 21 | `opt_350m` | `facebook/opt-350m` | [`lora_adapters/opt_350m/`](../lora_adapters/opt_350m/) | [`opt_350m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_350m) |
| 22 | `phi_1_5` | `microsoft/phi-1_5` | [`lora_adapters/phi_1_5/`](../lora_adapters/phi_1_5/) | [`phi_1_5/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_1_5) |
| 23 | `phi_2` | `microsoft/phi-2` | [`lora_adapters/phi_2/`](../lora_adapters/phi_2/) | [`phi_2/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_2) |
| 24 | `phi_3.5_mini_instruct` | `microsoft/Phi-3.5-mini-instruct` | [`lora_adapters/phi_3.5_mini_instruct/`](../lora_adapters/phi_3.5_mini_instruct/) | [`phi_3.5_mini_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3.5_mini_instruct) |
| 25 | `phi_3_mini_4k_instruct` | `microsoft/Phi-3-mini-4k-instruct` | [`lora_adapters/phi_3_mini_4k_instruct/`](../lora_adapters/phi_3_mini_4k_instruct/) | [`phi_3_mini_4k_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3_mini_4k_instruct) |
| 26 | `pythia_1.4b` | `EleutherAI/pythia-1.4b-deduped` | [`lora_adapters/pythia_1.4b/`](../lora_adapters/pythia_1.4b/) | [`pythia_1.4b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_1.4b) |
| 27 | `pythia_2.8b` | `EleutherAI/pythia-2.8b-deduped` | [`lora_adapters/pythia_2.8b/`](../lora_adapters/pythia_2.8b/) | [`pythia_2.8b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_2.8b) |
| 28 | `pythia_410m` | `EleutherAI/pythia-410m-deduped` | [`lora_adapters/pythia_410m/`](../lora_adapters/pythia_410m/) | [`pythia_410m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_410m) |
| 29 | `pythia_70m` | `EleutherAI/pythia-70m-deduped` | [`lora_adapters/pythia_70m/`](../lora_adapters/pythia_70m/) | [`pythia_70m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_70m) |
| 30 | `qwen1.5_0.5b_chat` | `Qwen/Qwen1.5-0.5B-Chat` | [`lora_adapters/qwen1.5_0.5b_chat/`](../lora_adapters/qwen1.5_0.5b_chat/) | [`qwen1.5_0.5b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_0.5b_chat) |
| 31 | `qwen1.5_1.8b_chat` | `Qwen/Qwen1.5-1.8B-Chat` | [`lora_adapters/qwen1.5_1.8b_chat/`](../lora_adapters/qwen1.5_1.8b_chat/) | [`qwen1.5_1.8b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_1.8b_chat) |
| 32 | `qwen2.5_0.5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | [`lora_adapters/qwen2.5_0.5b_instruct/`](../lora_adapters/qwen2.5_0.5b_instruct/) | [`qwen2.5_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_0.5b_instruct) |
| 33 | `qwen2.5_1.5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | [`lora_adapters/qwen2.5_1.5b_instruct/`](../lora_adapters/qwen2.5_1.5b_instruct/) | [`qwen2.5_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_1.5b_instruct) |
| 34 | `qwen2.5_3b_instruct` | `Qwen/Qwen2.5-3B-Instruct` | [`lora_adapters/qwen2.5_3b_instruct/`](../lora_adapters/qwen2.5_3b_instruct/) | [`qwen2.5_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_3b_instruct) |
| 35 | `qwen2.5_coder_0.5b_instruct` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | [`lora_adapters/qwen2.5_coder_0.5b_instruct/`](../lora_adapters/qwen2.5_coder_0.5b_instruct/) | [`qwen2.5_coder_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_0.5b_instruct) |
| 36 | `qwen2.5_coder_1.5b_instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [`lora_adapters/qwen2.5_coder_1.5b_instruct/`](../lora_adapters/qwen2.5_coder_1.5b_instruct/) | [`qwen2.5_coder_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_1.5b_instruct) |
| 37 | `qwen2.5_coder_3b_instruct` | `Qwen/Qwen2.5-Coder-3B-Instruct` | [`lora_adapters/qwen2.5_coder_3b_instruct/`](../lora_adapters/qwen2.5_coder_3b_instruct/) | [`qwen2.5_coder_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_3b_instruct) |
| 38 | `qwen2.5_math_1.5b_instruct` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | [`lora_adapters/qwen2.5_math_1.5b_instruct/`](../lora_adapters/qwen2.5_math_1.5b_instruct/) | [`qwen2.5_math_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_math_1.5b_instruct) |
| 39 | `qwen2_0.5b_instruct` | `Qwen/Qwen2-0.5B-Instruct` | [`lora_adapters/qwen2_0.5b_instruct/`](../lora_adapters/qwen2_0.5b_instruct/) | [`qwen2_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_0.5b_instruct) |
| 40 | `qwen2_1.5b_instruct` | `Qwen/Qwen2-1.5B-Instruct` | [`lora_adapters/qwen2_1.5b_instruct/`](../lora_adapters/qwen2_1.5b_instruct/) | [`qwen2_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_1.5b_instruct) |
| 41 | `rugpt3_large` | `ai-forever/rugpt3large_based_on_gpt2` | [`lora_adapters/rugpt3_large/`](../lora_adapters/rugpt3_large/) | [`rugpt3_large/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_large) |
| 42 | `rugpt3_medium` | `ai-forever/rugpt3medium_based_on_gpt2` | [`lora_adapters/rugpt3_medium/`](../lora_adapters/rugpt3_medium/) | [`rugpt3_medium/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_medium) |
| 43 | `rugpt3_small` | `ai-forever/rugpt3small_based_on_gpt2` | [`lora_adapters/rugpt3_small/`](../lora_adapters/rugpt3_small/) | [`rugpt3_small/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_small) |
| 44 | `russian_it_lora` | `Qwen/Qwen2.5-0.5B-Instruct` | [`lora_adapters/russian_it_lora/`](../lora_adapters/russian_it_lora/) | [`russian_it_lora/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/russian_it_lora) |
| 45 | `sber_mgpt` | `ai-forever/mGPT` | [`lora_adapters/sber_mgpt/`](../lora_adapters/sber_mgpt/) | [`sber_mgpt/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/sber_mgpt) |
| 46 | `smollm2_1.7b_instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | [`lora_adapters/smollm2_1.7b_instruct/`](../lora_adapters/smollm2_1.7b_instruct/) | [`smollm2_1.7b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_1.7b_instruct) |
| 47 | `smollm2_135m_instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | [`lora_adapters/smollm2_135m_instruct/`](../lora_adapters/smollm2_135m_instruct/) | [`smollm2_135m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_135m_instruct) |
| 48 | `smollm2_360m_instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | [`lora_adapters/smollm2_360m_instruct/`](../lora_adapters/smollm2_360m_instruct/) | [`smollm2_360m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_360m_instruct) |
| 49 | `smollm_1.7b_instruct` | `HuggingFaceTB/SmolLM-1.7B-Instruct` | [`lora_adapters/smollm_1.7b_instruct/`](../lora_adapters/smollm_1.7b_instruct/) | [`smollm_1.7b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_1.7b_instruct) |
| 50 | `smollm_135m_instruct` | `HuggingFaceTB/SmolLM-135M-Instruct` | [`lora_adapters/smollm_135m_instruct/`](../lora_adapters/smollm_135m_instruct/) | [`smollm_135m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_135m_instruct) |
| 51 | `smollm_360m_instruct` | `HuggingFaceTB/SmolLM-360M-Instruct` | [`lora_adapters/smollm_360m_instruct/`](../lora_adapters/smollm_360m_instruct/) | [`smollm_360m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_360m_instruct) |
| 52 | `stablelm_2_1_6b_chat` | `stabilityai/stablelm-2-1_6b-chat` | [`lora_adapters/stablelm_2_1_6b_chat/`](../lora_adapters/stablelm_2_1_6b_chat/) | [`stablelm_2_1_6b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_1_6b_chat) |
| 53 | `stablelm_2_zephyr_1_6b` | `stabilityai/stablelm-2-zephyr-1_6b` | [`lora_adapters/stablelm_2_zephyr_1_6b/`](../lora_adapters/stablelm_2_zephyr_1_6b/) | [`stablelm_2_zephyr_1_6b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_zephyr_1_6b) |
| 54 | `tiny_starcoder_py` | `bigcode/tiny_starcoder_py` | [`lora_adapters/tiny_starcoder_py/`](../lora_adapters/tiny_starcoder_py/) | [`tiny_starcoder_py/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tiny_starcoder_py) |
| 55 | `tinyllama_1.1b_chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | [`lora_adapters/tinyllama_1.1b_chat/`](../lora_adapters/tinyllama_1.1b_chat/) | [`tinyllama_1.1b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tinyllama_1.1b_chat) |
| 56 | `vikhr_llama_3.2_1b` | `Vikhrmodels/Vikhr-Llama-3.2-1B-instruct` | [`lora_adapters/vikhr_llama_3.2_1b/`](../lora_adapters/vikhr_llama_3.2_1b/) | [`vikhr_llama_3.2_1b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_llama_3.2_1b) |
| 57 | `vikhr_qwen_2.5_0.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct` | [`lora_adapters/vikhr_qwen_2.5_0.5b/`](../lora_adapters/vikhr_qwen_2.5_0.5b/) | [`vikhr_qwen_2.5_0.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_0.5b) |
| 58 | `vikhr_qwen_2.5_1.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` | [`lora_adapters/vikhr_qwen_2.5_1.5b/`](../lora_adapters/vikhr_qwen_2.5_1.5b/) | [`vikhr_qwen_2.5_1.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_1.5b) |

---

## ⚙️ Аппаратные требования и воспроизводимость

- **Потребление VRAM при обучении:** ~4.35 GB на NVIDIA GeForce RTX 3060 (12 GB) с gradient accumulation = 4, batch size = 1.
- **Флагманские 7B–8B модели:** используют 4-битное квантование BitsAndBytes (NF4) для инференса в пределах 6 GB VRAM.
- **Метрики адаптации:** Доменное дообучение снижает перплексию на русском инженерном тексте с 35.44 до 32.19 (-3.25 PPL), повышая соответствие лексике и архитектурным паттернам сообщества.
