---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- russian
- ru
---

# deepseek_r1_distill_qwen_1.5b

Domain-adaptation LoRA adapter for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, part of the Russian IT
Community Corpus (RICC) LoRA Zoo.

## Model Details

- **Developed by:** wwewtech (Russian IT Community Corpus project)
- **Model type:** LORA adapter (rank r=16, alpha=32, dropout=0.05), task type `CAUSAL_LM`
- **Target modules:** o_proj, q_proj, v_proj, k_proj
- **Adapter weights:** 16.65 MB, sha256 `9b238617018b51a07b0c312c68df900eff592b64bf5435830d7ffe1483c906cb`
- **Language(s) (NLP):** Russian (ru), English technical terms
- **License:** MIT (repository license, see `LICENSE`); the base model retains its own license
- **Finetuned from model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

### Model Sources

- **Repository (adapter):** https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/deepseek_r1_distill_qwen_1.5b
- **Repository (code & pipeline):** https://github.com/wwewtech/russian-it-community-corpus
- **Training corpus:** https://huggingface.co/datasets/wwewtech/russian-it-community-corpus

## Uses

### Direct Use

Fine-tuned variant of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for Russian-language IT discourse
(backend, DevOps, AI/ML, infrastructure). Load with PEFT against the same
base model:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "lora_adapters/deepseek_r1_distill_qwen_1.5b/")
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
