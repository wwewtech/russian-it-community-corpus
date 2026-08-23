# 🦁 Russian IT Community LoRA Model Zoo

**Официальный каталог 34 предварительно обученных LoRA-адаптеров**, дообученных на корпусе **RICC (2.91M сообщений, 171.5k диалогов)** для русскоязычного IT-дискурса, бэкенда, DevOps, AI/ML и инфраструктуры.

Все адаптеры можно загружать локально из репозитория или через Hugging Face Hub: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora).

---

## 📊 Доступные LoRA-адаптеры

| Идентификатор | Базовая модель | Семейство | Параметры | Каталог адаптера |
| :--- | :--- | :--- | :---: | :--- |
| `bloom_1b7` | **`bigscience/bloom-1b7`** | BigScience BLOOM | 1.7B | [`lora_adapters/bloom_1b7/`](file:///D:/project_x/lora_adapters/bloom_1b7/) |
| `deepseek_coder_1.3b_instruct` | **`deepseek-ai/deepseek-coder-1.3b-instruct`** | DeepSeek Coder | 1.3B | [`lora_adapters/deepseek_coder_1.3b_instruct/`](file:///D:/project_x/lora_adapters/deepseek_coder_1.3b_instruct/) |
| `deepseek_r1_distill_qwen_1.5b` | **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** | DeepSeek R1 | 1.5B | [`lora_adapters/deepseek_r1_distill_qwen_1.5b/`](file:///D:/project_x/lora_adapters/deepseek_r1_distill_qwen_1.5b/) |
| `gemma_2_2b_it` | **`unsloth/gemma-2-2b-it`** | Gemma 2 | 2.0B | [`lora_adapters/gemma_2_2b_it/`](file:///D:/project_x/lora_adapters/gemma_2_2b_it/) |
| `granite_3b_code_instruct` | **`ibm-granite/granite-3b-code-instruct`** | IBM Granite | 3.0B | [`lora_adapters/granite_3b_code_instruct/`](file:///D:/project_x/lora_adapters/granite_3b_code_instruct/) |
| `llama_3.2_1b_instruct` | **`unsloth/Llama-3.2-1B-Instruct`** | Llama 3.2 | 1.0B | [`lora_adapters/llama_3.2_1b_instruct/`](file:///D:/project_x/lora_adapters/llama_3.2_1b_instruct/) |
| `llama_3.2_3b_instruct` | **`unsloth/Llama-3.2-3B-Instruct`** | Llama 3.2 | 3.0B | [`lora_adapters/llama_3.2_3b_instruct/`](file:///D:/project_x/lora_adapters/llama_3.2_3b_instruct/) |
| `minicpm_2b_dpo` | **`openbmb/MiniCPM-2B-dpo-bf16`** | MiniCPM | 2.0B | [`lora_adapters/minicpm_2b_dpo/`](file:///D:/project_x/lora_adapters/minicpm_2b_dpo/) |
| `phi_3.5_mini_instruct` | **`microsoft/Phi-3.5-mini-instruct`** | Phi 3.5 | 3.8B | [`lora_adapters/phi_3.5_mini_instruct/`](file:///D:/project_x/lora_adapters/phi_3.5_mini_instruct/) |
| `pythia_1.4b` | **`EleutherAI/pythia-1.4b-deduped`** | EleutherAI Pythia | 1.4B | [`lora_adapters/pythia_1.4b/`](file:///D:/project_x/lora_adapters/pythia_1.4b/) |
| `pythia_2.8b` | **`EleutherAI/pythia-2.8b-deduped`** | EleutherAI Pythia | 2.8B | [`lora_adapters/pythia_2.8b/`](file:///D:/project_x/lora_adapters/pythia_2.8b/) |
| `qwen2.5_0.5b_instruct` | **`Qwen/Qwen2.5-0.5B-Instruct`** | Qwen 2.5 | 0.5B | [`lora_adapters/qwen2.5_0.5b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_0.5b_instruct/) |
| `qwen2.5_1.5b_instruct` | **`Qwen/Qwen2.5-1.5B-Instruct`** | Qwen 2.5 | 1.5B | [`lora_adapters/qwen2.5_1.5b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_1.5b_instruct/) |
| `qwen2.5_3b_instruct` | **`Qwen/Qwen2.5-3B-Instruct`** | Qwen 2.5 | 3.0B | [`lora_adapters/qwen2.5_3b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_3b_instruct/) |
| `qwen2.5_coder_0.5b_instruct` | **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** | Qwen 2.5 Coder | 0.5B | [`lora_adapters/qwen2.5_coder_0.5b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_coder_0.5b_instruct/) |
| `qwen2.5_coder_1.5b_instruct` | **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** | Qwen 2.5 Coder | 1.5B | [`lora_adapters/qwen2.5_coder_1.5b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_coder_1.5b_instruct/) |
| `qwen2.5_coder_3b_instruct` | **`Qwen/Qwen2.5-Coder-3B-Instruct`** | Qwen 2.5 Coder | 3.0B | [`lora_adapters/qwen2.5_coder_3b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_coder_3b_instruct/) |
| `qwen2.5_math_1.5b_instruct` | **`Qwen/Qwen2.5-Math-1.5B-Instruct`** | Qwen 2.5 Math | 1.5B | [`lora_adapters/qwen2.5_math_1.5b_instruct/`](file:///D:/project_x/lora_adapters/qwen2.5_math_1.5b_instruct/) |
| `rugpt3_large` | **`ai-forever/rugpt3large_based_on_gpt2`** | Sber AI | 760M | [`lora_adapters/rugpt3_large/`](file:///D:/project_x/lora_adapters/rugpt3_large/) |
| `rugpt3_medium` | **`ai-forever/rugpt3medium_based_on_gpt2`** | Sber AI | 350M | [`lora_adapters/rugpt3_medium/`](file:///D:/project_x/lora_adapters/rugpt3_medium/) |
| `rugpt3_small` | **`ai-forever/rugpt3small_based_on_gpt2`** | Sber AI | 125M | [`lora_adapters/rugpt3_small/`](file:///D:/project_x/lora_adapters/rugpt3_small/) |
| `russian_it_lora` | **`Qwen/Qwen2.5-0.5B-Instruct`** | Qwen Baseline | 0.5B | [`lora_adapters/russian_it_lora/`](file:///D:/project_x/lora_adapters/russian_it_lora/) |
| `smollm2_1.7b_instruct` | **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | SmolLM2 | 1.7B | [`lora_adapters/smollm2_1.7b_instruct/`](file:///D:/project_x/lora_adapters/smollm2_1.7b_instruct/) |
| `smollm2_135m_instruct` | **`HuggingFaceTB/SmolLM2-135M-Instruct`** | SmolLM2 | 135M | [`lora_adapters/smollm2_135m_instruct/`](file:///D:/project_x/lora_adapters/smollm2_135m_instruct/) |
| `smollm2_360m_instruct` | **`HuggingFaceTB/SmolLM2-360M-Instruct`** | SmolLM2 | 360M | [`lora_adapters/smollm2_360m_instruct/`](file:///D:/project_x/lora_adapters/smollm2_360m_instruct/) |
| `smollm_1.7b_instruct` | **`HuggingFaceTB/SmolLM-1.7B-Instruct`** | SmolLM v1 | 1.7B | [`lora_adapters/smollm_1.7b_instruct/`](file:///D:/project_x/lora_adapters/smollm_1.7b_instruct/) |
| `smollm_135m_instruct` | **`HuggingFaceTB/SmolLM-135M-Instruct`** | SmolLM v1 | 135M | [`lora_adapters/smollm_135m_instruct/`](file:///D:/project_x/lora_adapters/smollm_135m_instruct/) |
| `smollm_360m_instruct` | **`HuggingFaceTB/SmolLM-360M-Instruct`** | SmolLM v1 | 360M | [`lora_adapters/smollm_360m_instruct/`](file:///D:/project_x/lora_adapters/smollm_360m_instruct/) |
| `stablelm_2_1_6b_chat` | **`stabilityai/stablelm-2-1_6b-chat`** | Stability AI | 1.6B | [`lora_adapters/stablelm_2_1_6b_chat/`](file:///D:/project_x/lora_adapters/stablelm_2_1_6b_chat/) |
| `stablelm_2_zephyr_1_6b` | **`stabilityai/stablelm-2-zephyr-1_6b`** | Stability AI | 1.6B | [`lora_adapters/stablelm_2_zephyr_1_6b/`](file:///D:/project_x/lora_adapters/stablelm_2_zephyr_1_6b/) |
| `tinyllama_1.1b_chat` | **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** | TinyLlama | 1.1B | [`lora_adapters/tinyllama_1.1b_chat/`](file:///D:/project_x/lora_adapters/tinyllama_1.1b_chat/) |
| `vikhr_llama_3.2_1b` | **`Vikhrmodels/Vikhr-Llama-3.2-1B-instruct`** | Vikhr Russian NLP | 1.0B | [`lora_adapters/vikhr_llama_3.2_1b/`](file:///D:/project_x/lora_adapters/vikhr_llama_3.2_1b/) |
| `vikhr_qwen_2.5_0.5b` | **`Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct`** | Vikhr Russian NLP | 0.5B | [`lora_adapters/vikhr_qwen_2.5_0.5b/`](file:///D:/project_x/lora_adapters/vikhr_qwen_2.5_0.5b/) |
| `vikhr_qwen_2.5_1.5b` | **`Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct`** | Vikhr Russian NLP | 1.5B | [`lora_adapters/vikhr_qwen_2.5_1.5b/`](file:///D:/project_x/lora_adapters/vikhr_qwen_2.5_1.5b/) |