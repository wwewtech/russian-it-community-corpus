---
base_model: unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- russian
- ru
---

# heavyweight_deepseek_r1_7b

Domain-adaptation LoRA adapter for `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit`, part of the Russian IT
Community Corpus (RICC) LoRA Zoo.

## Model Details

- **Developed by:** wwewtech (Russian IT Community Corpus project)
- **Model type:** LORA adapter (rank r=16, alpha=32, dropout=0.05), task type `CAUSAL_LM`
- **Target modules:** q_proj, o_proj, up_proj, k_proj, gate_proj, down_proj, v_proj
- **Adapter weights:** 154.05 MB, sha256 `b88a0749285fc01333a57ea9f763959b6a107d418926d97960a7fd4baddc88f1`
- **Language(s) (NLP):** Russian (ru), English technical terms
- **License:** MIT (repository license, see `LICENSE`); the base model retains its own license
- **Finetuned from model:** `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit`

### Model Sources

- **Repository (adapter):** https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/heavyweight_deepseek_r1_7b
- **Repository (code & pipeline):** https://github.com/wwewtech/russian-it-community-corpus
- **Training corpus:** https://huggingface.co/datasets/wwewtech/russian-it-community-corpus

## Uses

### Direct Use

Fine-tuned variant of `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` for Russian-language IT discourse
(backend, DevOps, AI/ML, infrastructure). Load with PEFT against the same
base model:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "lora_adapters/heavyweight_deepseek_r1_7b/")
```

### Out-of-Scope Use

- Not a general assistant: capability scores are **not published** (see
  benchmark retraction notice in the repository `README.md`).
- Not validated for safety-critical or legally binding advice.

## Bias, Risks and Limitations

- Training data is de-identified community chat: heuristic + probabilistic
  PII audit with a documented statistical upper bound (see
  `reports/probabilistic_pii_audit.json`), not a zero-leak guarantee.
- Chat-derived content may contain outdated or opinionated technical advice.
- Notice and takedown: see `DATASET_TERMS.md`.

## Training Details

- **Training data:** RICC SFT dialogues (171,520 curated multi-turn dialogues).
- **Training regime:** pilot domain-adaptation checkpoint — 50-100 training
  steps on sampled domain batches (statement from `reports/HF_MODEL_CARD.md`),
  **not** multi-epoch training over the whole corpus.
- **Framework:** PEFT 0.20.0 (see `adapter_config.json` in this directory).

## Evaluation

No benchmark scores are claimed for this adapter. Published academic
benchmark numbers were retracted (answer-parsing and column-mapping defects);
see the retraction notice in `README.md` before citing any evaluation figures.

## Citation

```bibtex
@misc{ricc2026,
  author = {Russian IT Community Open Research Group},
  title = {RICC: Russian IT Community Corpus},
  year = {2026},
  howpublished = {\url{https://github.com/wwewtech/russian-it-community-corpus}}
}
```
