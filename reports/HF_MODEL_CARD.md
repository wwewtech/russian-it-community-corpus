---
license: mit
language:
- ru
- en
library_name: peft
tags:
- russian
- lora
- qlora
- peft
- sft
- text-generation
- russian-nlp
---

# 🦁 Russian IT Community LoRA Model Zoo

**58 pre-trained adapters** (55 domain adapters + 3 flagship 7B–8B QLoRA), fine-tuned on the RICC corpus (2.91M messages, 171.5k curated SFT dialogues) for Russian-language IT discourse: backend, DevOps, AI/ML, infrastructure.

> Catalog regenerated from the Hub file tree on 2026-09-02. Source of truth: the `siblings` listing of this repository.

## ⚡ Quick Start: 3-Line Inference

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"          # any base model from the catalog
adapter_id = "wwewtech/russian-it-community-lora"
subfolder = "qwen2.5_1.5b_instruct"              # choose from the catalog below

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)

inputs = tokenizer("<|user|>\nКак настроить репликацию PostgreSQL?\n<|assistant|>\n", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📚 Full Catalog (58 Adapters)

| # | Adapter Subfolder | Base Model | Hub Link |
| :---: | :--- | :--- | :--- |
| 01 | `bloom_1b7` | `bigscience/bloom-1b7` | [`bloom_1b7/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_1b7) |
| 02 | `bloom_560m` | `bigscience/bloom-560m` | [`bloom_560m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_560m) |
| 03 | `codegen_350m_multi` | `Salesforce/codegen-350M-multi` | [`codegen_350m_multi/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/codegen_350m_multi) |
| 04 | `deepseek_coder_1.3b_instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct` | [`deepseek_coder_1.3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_coder_1.3b_instruct) |
| 05 | `deepseek_r1_distill_qwen_1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | [`deepseek_r1_distill_qwen_1.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_r1_distill_qwen_1.5b) |
| 06 | `falcon3_1b_instruct` | `tiiuae/Falcon3-1B-Instruct` | [`falcon3_1b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_1b_instruct) |
| 07 | `falcon3_3b_instruct` | `tiiuae/Falcon3-3B-Instruct` | [`falcon3_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_3b_instruct) |
| 08 | `gemma_2_2b_it` | `unsloth/gemma-2-2b-it` | [`gemma_2_2b_it/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gemma_2_2b_it) |
| 09 | `gpt2_large` | `openai-community/gpt2-large` | [`gpt2_large/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_large) |
| 10 | `gpt2_medium` | `openai-community/gpt2-medium` | [`gpt2_medium/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_medium) |
| 11 | `granite_3b_code_instruct` | `ibm-granite/granite-3b-code-instruct` | [`granite_3b_code_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/granite_3b_code_instruct) |
| 12 | `heavyweight_deepseek_r1_7b` | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` | [`heavyweight_deepseek_r1_7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_deepseek_r1_7b) |
| 13 | `heavyweight_llama3.1_8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | [`heavyweight_llama3.1_8b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_llama3.1_8b) |
| 14 | `heavyweight_qwen2.5_coder_7b` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | [`heavyweight_qwen2.5_coder_7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_qwen2.5_coder_7b) |
| 15 | `llama_3.2_1b_instruct` | `unsloth/Llama-3.2-1B-Instruct` | [`llama_3.2_1b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_1b_instruct) |
| 16 | `llama_3.2_3b_instruct` | `unsloth/Llama-3.2-3B-Instruct` | [`llama_3.2_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_3b_instruct) |
| 17 | `minicpm_2b_dpo` | `openbmb/MiniCPM-2B-dpo-bf16` | [`minicpm_2b_dpo/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/minicpm_2b_dpo) |
| 18 | `opt_1.3b` | `facebook/opt-1.3b` | [`opt_1.3b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_1.3b) |
| 19 | `opt_125m` | `facebook/opt-125m` | [`opt_125m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_125m) |
| 20 | `opt_2.7b` | `facebook/opt-2.7b` | [`opt_2.7b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_2.7b) |
| 21 | `opt_350m` | `facebook/opt-350m` | [`opt_350m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_350m) |
| 22 | `phi_1_5` | `microsoft/phi-1_5` | [`phi_1_5/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_1_5) |
| 23 | `phi_2` | `microsoft/phi-2` | [`phi_2/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_2) |
| 24 | `phi_3.5_mini_instruct` | `microsoft/Phi-3.5-mini-instruct` | [`phi_3.5_mini_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3.5_mini_instruct) |
| 25 | `phi_3_mini_4k_instruct` | `microsoft/Phi-3-mini-4k-instruct` | [`phi_3_mini_4k_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3_mini_4k_instruct) |
| 26 | `pythia_1.4b` | `EleutherAI/pythia-1.4b-deduped` | [`pythia_1.4b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_1.4b) |
| 27 | `pythia_2.8b` | `EleutherAI/pythia-2.8b-deduped` | [`pythia_2.8b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_2.8b) |
| 28 | `pythia_410m` | `EleutherAI/pythia-410m-deduped` | [`pythia_410m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_410m) |
| 29 | `pythia_70m` | `EleutherAI/pythia-70m-deduped` | [`pythia_70m/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_70m) |
| 30 | `qwen1.5_0.5b_chat` | `Qwen/Qwen1.5-0.5B-Chat` | [`qwen1.5_0.5b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_0.5b_chat) |
| 31 | `qwen1.5_1.8b_chat` | `Qwen/Qwen1.5-1.8B-Chat` | [`qwen1.5_1.8b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_1.8b_chat) |
| 32 | `qwen2.5_0.5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | [`qwen2.5_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_0.5b_instruct) |
| 33 | `qwen2.5_1.5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | [`qwen2.5_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_1.5b_instruct) |
| 34 | `qwen2.5_3b_instruct` | `Qwen/Qwen2.5-3B-Instruct` | [`qwen2.5_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_3b_instruct) |
| 35 | `qwen2.5_coder_0.5b_instruct` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | [`qwen2.5_coder_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_0.5b_instruct) |
| 36 | `qwen2.5_coder_1.5b_instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [`qwen2.5_coder_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_1.5b_instruct) |
| 37 | `qwen2.5_coder_3b_instruct` | `Qwen/Qwen2.5-Coder-3B-Instruct` | [`qwen2.5_coder_3b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_3b_instruct) |
| 38 | `qwen2.5_math_1.5b_instruct` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | [`qwen2.5_math_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_math_1.5b_instruct) |
| 39 | `qwen2_0.5b_instruct` | `Qwen/Qwen2-0.5B-Instruct` | [`qwen2_0.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_0.5b_instruct) |
| 40 | `qwen2_1.5b_instruct` | `Qwen/Qwen2-1.5B-Instruct` | [`qwen2_1.5b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_1.5b_instruct) |
| 41 | `rugpt3_large` | `ai-forever/rugpt3large_based_on_gpt2` | [`rugpt3_large/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_large) |
| 42 | `rugpt3_medium` | `ai-forever/rugpt3medium_based_on_gpt2` | [`rugpt3_medium/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_medium) |
| 43 | `rugpt3_small` | `ai-forever/rugpt3small_based_on_gpt2` | [`rugpt3_small/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_small) |
| 44 | `russian_it_lora` | `Qwen/Qwen2.5-0.5B-Instruct` | [`russian_it_lora/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/russian_it_lora) |
| 45 | `sber_mgpt` | `ai-forever/mGPT` | [`sber_mgpt/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/sber_mgpt) |
| 46 | `smollm2_1.7b_instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | [`smollm2_1.7b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_1.7b_instruct) |
| 47 | `smollm2_135m_instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | [`smollm2_135m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_135m_instruct) |
| 48 | `smollm2_360m_instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | [`smollm2_360m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_360m_instruct) |
| 49 | `smollm_1.7b_instruct` | `HuggingFaceTB/SmolLM-1.7B-Instruct` | [`smollm_1.7b_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_1.7b_instruct) |
| 50 | `smollm_135m_instruct` | `HuggingFaceTB/SmolLM-135M-Instruct` | [`smollm_135m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_135m_instruct) |
| 51 | `smollm_360m_instruct` | `HuggingFaceTB/SmolLM-360M-Instruct` | [`smollm_360m_instruct/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_360m_instruct) |
| 52 | `stablelm_2_1_6b_chat` | `stabilityai/stablelm-2-1_6b-chat` | [`stablelm_2_1_6b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_1_6b_chat) |
| 53 | `stablelm_2_zephyr_1_6b` | `stabilityai/stablelm-2-zephyr-1_6b` | [`stablelm_2_zephyr_1_6b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_zephyr_1_6b) |
| 54 | `tiny_starcoder_py` | `bigcode/tiny_starcoder_py` | [`tiny_starcoder_py/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tiny_starcoder_py) |
| 55 | `tinyllama_1.1b_chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | [`tinyllama_1.1b_chat/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tinyllama_1.1b_chat) |
| 56 | `vikhr_llama_3.2_1b` | `Vikhrmodels/Vikhr-Llama-3.2-1B-instruct` | [`vikhr_llama_3.2_1b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_llama_3.2_1b) |
| 57 | `vikhr_qwen_2.5_0.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct` | [`vikhr_qwen_2.5_0.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_0.5b) |
| 58 | `vikhr_qwen_2.5_1.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` | [`vikhr_qwen_2.5_1.5b/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_1.5b) |

## 🥇 Flagship QLoRA Models (7B–8B)

Full-precision copies also live under [`models/`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/models): `models/heavyweight_qwen2.5_coder_7b`, `models/heavyweight_deepseek_r1_7b`, `models/heavyweight_llama3.1_8b`.

## 📓 Training Data & Evaluation Status

- Training corpus: [RICC SFT Dialogues](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus) (171,520 multi-turn dialogues).
- Academic benchmark numbers (HumanEval / RuMMLU / PPL) published earlier are **withdrawn pending re-evaluation**: the harness had answer-parsing and column-mapping defects that produced implausible values (see repo commit history). Enterprise scenario scores are rubric-based heuristics, not capability measurements.
