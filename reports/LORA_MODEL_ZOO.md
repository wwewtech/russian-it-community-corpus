# Russian IT Community LoRA Model Zoo

**Official release catalog: 1 verified full-epoch LoRA adapter(s).**

| # | Adapter | Base model | Size (MiB) | SHA-256 |
| ---: | --- | --- | ---: | --- |
| 1 | `smollm2_135m_instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 7.06 | `c1fbaaa3ab04313726a4c505440817e2cb967f568ba7019cd6d9240e92fa0c16` |

## Quickstart: verified full-epoch adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

subfolder = "smollm2_135m_instruct"
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
base_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
adapter_hub = "wwewtech/russian-it-community-lora"
adapter_revision = "47f5d37f9e08d921f23df2af8fbdaa614cc81bf3"

tokenizer = AutoTokenizer.from_pretrained(adapter_hub, subfolder=subfolder, revision=adapter_revision)
base = AutoModelForCausalLM.from_pretrained(model_id, revision=base_revision, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_hub, subfolder=subfolder, revision=adapter_revision)
messages = [{"role": "user", "content": "Как настроить Nginx reverse proxy в Docker?"}]
inputs = tokenizer(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt"
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Historical HF artifacts (not production releases)

Legacy folders remain for reproducibility and are NOT deleted by this catalog update. The presence of `adapter_model.safetensors` alone does not certify a final adapter.

| Status | Count |
| --- | ---: |
| Full-corpus, full-epoch final adapters | 1 |
| Time-boxed pilot adapters (NOT final) | 5 |
| Legacy adapters without training provenance (NOT final) | 52 |
| Nested legacy checkpoints under `models/` (NOT final) | 3 |
| Other partial/altered artifacts (NOT final) | 0 |

Hub audit revision: `47f5d37f9e08d921f23df2af8fbdaa614cc81bf3`.
