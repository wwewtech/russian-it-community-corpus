---
license: mit
language:
- ru
- en
library_name: peft
tags:
- russian
- lora
- peft
- sft
- text-generation
- russian-nlp
---

# Russian IT Community LoRA Model Zoo

## Production release: one verified full-epoch adapter

**`smollm2_135m_instruct`** — base model `HuggingFaceTB/SmolLM2-135M-Instruct`; 21209 / 21209 optimizer steps, 100 % coverage of 169667 trainable dialogues, no wall-clock budget.

The production entry below is verified at immutable Hub revision `47f5d37f9e08d921f23df2af8fbdaa614cc81bf3`. The SHA-256 digest of its safetensors must match the local registry exactly.

## Quick start

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

## Training policy

A real release adapter must complete the full corpus for at least one epoch. A time budget, partial epoch, capped sample pool or absent provenance disqualifies it.

## Historical HF artifacts (not production releases)

Legacy folders remain for reproducibility and are NOT deleted by this catalog update. The presence of `adapter_model.safetensors` alone does not certify a final adapter.

| Status | Count |
| --- | ---: |
| Full-corpus, full-epoch final adapters | 1 |
| Time-boxed pilot adapters (NOT final) | 5 |
| Legacy adapters without training provenance (NOT final) | 52 |
| Nested legacy checkpoints under `models/` (NOT final) | 3 |
| Other partial/altered artifacts (NOT final) | 0 |

Machine-readable per-file audit: [`hf_hub_audit.json`](https://huggingface.co/wwewtech/russian-it-community-lora/blob/main/hf_hub_audit.json).

## Provenance and limitations

- Immutable adapter revision: `47f5d37f9e08d921f23df2af8fbdaa614cc81bf3`.
- Immutable base-model revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- The adapter weights have been re-checked against the registry after the metadata repair.
- Older incomplete adapters remain experimental artifacts and MUST NOT be advertised as final models.
- Academic benchmark results previously advertised for the old zoo have been withdrawn pending re-evaluation.
- Dataset identity is independently pinned by Git+LFS revision, file size and SHA-256. Use the pinned `data/full_clean_messages.parquet`, `data/sft_dialogues.parquet` and `data/rag_knowledge_base.parquet` of the RICC dataset release, not a moving snapshot, when reproducing this model.
