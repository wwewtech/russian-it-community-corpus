# 🦁 Russian IT Community LoRA Model Zoo (55 Foundation Models)

> **Official Hugging Face Hub Repository:** [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora)  
> **Total Pre-trained Adapters:** 55 Domain Adapters  
> **Training Corpus:** 171.5k Curated SFT Multi-Turn Dialogues (Russian IT Community Corpus)  
> **Hardware Target:** Optimized for NVIDIA GeForce RTX 3060 (12GB VRAM) & consumer GPUs  

---

## ⚡ Quick Start: 3-Line Inference

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "wwewtech/russian-it-community-lora"
subfolder = "qwen2.5_1.5b_instruct"  # Choose from 55 adapters below

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)

inputs = tokenizer("<|user|>\nКак настроить репликацию PostgreSQL в Kubernetes?<|assistant|>\n", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📚 Full Catalog of 55 Pre-Trained Adapters

| # | Adapter Subfolder | Base Model | Architecture Family | Hugging Face Subfolder Link |
| :---: | :--- | :--- | :--- | :--- |
| **01** | `bloom_1b7` | `bigscience/bloom-1b7` | BigScience BLOOM | [`bloom_1b7`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_1b7) |
| **02** | `bloom_560m` | `bigscience/bloom-560m` | BigScience BLOOM | [`bloom_560m`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/bloom_560m) |
| **03** | `deepseek_coder_1.3b_instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct` | DeepSeek | [`deepseek_coder_1.3b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_coder_1.3b_instruct) |
| **04** | `deepseek_r1_distill_qwen_1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek | [`deepseek_r1_distill_qwen_1.5b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_r1_distill_qwen_1.5b) |
| **05** | `falcon3_1b_instruct` | `tiiuae/Falcon3-1B-Instruct` | TII Falcon | [`falcon3_1b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_1b_instruct) |
| **06** | `falcon3_3b_instruct` | `tiiuae/Falcon3-3B-Instruct` | TII Falcon | [`falcon3_3b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/falcon3_3b_instruct) |
| **07** | `gemma_2_2b_it` | `unsloth/gemma-2-2b-it` | Open Weights | [`gemma_2_2b_it`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gemma_2_2b_it) |
| **08** | `gpt2_large` | `openai-community/gpt2-large` | Generative GPT-2 / RuGPT | [`gpt2_large`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_large) |
| **09** | `gpt2_medium` | `openai-community/gpt2-medium` | Generative GPT-2 / RuGPT | [`gpt2_medium`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/gpt2_medium) |
| **10** | `granite_3b_code_instruct` | `ibm-granite/granite-3b-code-instruct` | Open Weights | [`granite_3b_code_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/granite_3b_code_instruct) |
| **11** | `heavyweight_deepseek_r1_7b` | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` | DeepSeek | [`heavyweight_deepseek_r1_7b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_deepseek_r1_7b) |
| **12** | `heavyweight_llama3.1_8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Meta LLaMA | [`heavyweight_llama3.1_8b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_llama3.1_8b) |
| **13** | `heavyweight_qwen2.5_coder_7b` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | Qwen 2.5 Coder | [`heavyweight_qwen2.5_coder_7b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_qwen2.5_coder_7b) |
| **14** | `llama_3.2_1b_instruct` | `unsloth/Llama-3.2-1B-Instruct` | Meta LLaMA | [`llama_3.2_1b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_1b_instruct) |
| **15** | `llama_3.2_3b_instruct` | `unsloth/Llama-3.2-3B-Instruct` | Meta LLaMA | [`llama_3.2_3b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/llama_3.2_3b_instruct) |
| **16** | `minicpm_2b_dpo` | `openbmb/MiniCPM-2B-dpo-bf16` | Open Weights | [`minicpm_2b_dpo`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/minicpm_2b_dpo) |
| **17** | `opt_1.3b` | `facebook/opt-1.3b` | Meta OPT | [`opt_1.3b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_1.3b) |
| **18** | `opt_125m` | `facebook/opt-125m` | Meta OPT | [`opt_125m`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_125m) |
| **19** | `opt_2.7b` | `facebook/opt-2.7b` | Meta OPT | [`opt_2.7b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_2.7b) |
| **20** | `opt_350m` | `facebook/opt-350m` | Meta OPT | [`opt_350m`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/opt_350m) |
| **21** | `phi_1_5` | `microsoft/phi-1_5` | Microsoft Phi | [`phi_1_5`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_1_5) |
| **22** | `phi_2` | `microsoft/phi-2` | Microsoft Phi | [`phi_2`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_2) |
| **23** | `phi_3.5_mini_instruct` | `microsoft/Phi-3.5-mini-instruct` | Microsoft Phi | [`phi_3.5_mini_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3.5_mini_instruct) |
| **24** | `phi_3_mini_4k_instruct` | `microsoft/Phi-3-mini-4k-instruct` | Microsoft Phi | [`phi_3_mini_4k_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/phi_3_mini_4k_instruct) |
| **25** | `pythia_1.4b` | `EleutherAI/pythia-1.4b-deduped` | EleutherAI Pythia | [`pythia_1.4b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_1.4b) |
| **26** | `pythia_2.8b` | `EleutherAI/pythia-2.8b-deduped` | EleutherAI Pythia | [`pythia_2.8b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_2.8b) |
| **27** | `pythia_410m` | `EleutherAI/pythia-410m-deduped` | EleutherAI Pythia | [`pythia_410m`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_410m) |
| **28** | `pythia_70m` | `EleutherAI/pythia-70m-deduped` | EleutherAI Pythia | [`pythia_70m`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/pythia_70m) |
| **29** | `qwen1.5_0.5b_chat` | `Qwen/Qwen1.5-0.5B-Chat` | Qwen 1.5 | [`qwen1.5_0.5b_chat`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_0.5b_chat) |
| **30** | `qwen1.5_1.8b_chat` | `Qwen/Qwen1.5-1.8B-Chat` | Qwen 1.5 | [`qwen1.5_1.8b_chat`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen1.5_1.8b_chat) |
| **31** | `qwen2.5_0.5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | Qwen 2.5 | [`qwen2.5_0.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_0.5b_instruct) |
| **32** | `qwen2.5_1.5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen 2.5 | [`qwen2.5_1.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_1.5b_instruct) |
| **33** | `qwen2.5_3b_instruct` | `Qwen/Qwen2.5-3B-Instruct` | Qwen 2.5 | [`qwen2.5_3b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_3b_instruct) |
| **34** | `qwen2.5_coder_0.5b_instruct` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Qwen 2.5 Coder | [`qwen2.5_coder_0.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_0.5b_instruct) |
| **35** | `qwen2.5_coder_1.5b_instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Qwen 2.5 Coder | [`qwen2.5_coder_1.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_1.5b_instruct) |
| **36** | `qwen2.5_coder_3b_instruct` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Qwen 2.5 Coder | [`qwen2.5_coder_3b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_coder_3b_instruct) |
| **37** | `qwen2.5_math_1.5b_instruct` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Qwen 2.5 | [`qwen2.5_math_1.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2.5_math_1.5b_instruct) |
| **38** | `qwen2_0.5b_instruct` | `Qwen/Qwen2-0.5B-Instruct` | Qwen 2 | [`qwen2_0.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_0.5b_instruct) |
| **39** | `qwen2_1.5b_instruct` | `Qwen/Qwen2-1.5B-Instruct` | Qwen 2 | [`qwen2_1.5b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/qwen2_1.5b_instruct) |
| **40** | `rugpt3_large` | `ai-forever/rugpt3large_based_on_gpt2` | Generative GPT-2 / RuGPT | [`rugpt3_large`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_large) |
| **41** | `rugpt3_medium` | `ai-forever/rugpt3medium_based_on_gpt2` | Generative GPT-2 / RuGPT | [`rugpt3_medium`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_medium) |
| **42** | `rugpt3_small` | `ai-forever/rugpt3small_based_on_gpt2` | Generative GPT-2 / RuGPT | [`rugpt3_small`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/rugpt3_small) |
| **43** | `russian_it_lora` | `Qwen/Qwen2.5-0.5B-Instruct` | Open Weights | [`russian_it_lora`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/russian_it_lora) |
| **44** | `smollm2_1.7b_instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | SmolLM2 | [`smollm2_1.7b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_1.7b_instruct) |
| **45** | `smollm2_135m_instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | SmolLM2 | [`smollm2_135m_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_135m_instruct) |
| **46** | `smollm2_360m_instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | SmolLM2 | [`smollm2_360m_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_360m_instruct) |
| **47** | `smollm_1.7b_instruct` | `HuggingFaceTB/SmolLM-1.7B-Instruct` | SmolLM | [`smollm_1.7b_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_1.7b_instruct) |
| **48** | `smollm_135m_instruct` | `HuggingFaceTB/SmolLM-135M-Instruct` | SmolLM | [`smollm_135m_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_135m_instruct) |
| **49** | `smollm_360m_instruct` | `HuggingFaceTB/SmolLM-360M-Instruct` | SmolLM | [`smollm_360m_instruct`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm_360m_instruct) |
| **50** | `stablelm_2_1_6b_chat` | `stabilityai/stablelm-2-1_6b-chat` | Stability AI StableLM | [`stablelm_2_1_6b_chat`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_1_6b_chat) |
| **51** | `stablelm_2_zephyr_1_6b` | `stabilityai/stablelm-2-zephyr-1_6b` | Stability AI StableLM | [`stablelm_2_zephyr_1_6b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/stablelm_2_zephyr_1_6b) |
| **52** | `tinyllama_1.1b_chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Open Weights | [`tinyllama_1.1b_chat`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tinyllama_1.1b_chat) |
| **53** | `vikhr_llama_3.2_1b` | `Vikhrmodels/Vikhr-Llama-3.2-1B-instruct` | Meta LLaMA | [`vikhr_llama_3.2_1b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_llama_3.2_1b) |
| **54** | `vikhr_qwen_2.5_0.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct` | Vikhr Russian NLP | [`vikhr_qwen_2.5_0.5b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_0.5b) |
| **55** | `vikhr_qwen_2.5_1.5b` | `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` | Vikhr Russian NLP | [`vikhr_qwen_2.5_1.5b`](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/vikhr_qwen_2.5_1.5b) |
