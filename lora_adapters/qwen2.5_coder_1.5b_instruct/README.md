# Russian IT Community Corpus — LoRA Adapter
## Base Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Qwen 2.5 Coder · 1.5B)

This LoRA adapter is fine-tuned on the **RICC (Russian IT Community Corpus)** dataset (2.91M messages, 171k multi-turn dialogues) across 11 developer communities.

### Usage in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "lora_adapters/qwen2.5_coder_1.5b_instruct"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

prompt = "Как настроить Nginx reverse proxy с поддержкой WebSocket и SSL в Docker?"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
