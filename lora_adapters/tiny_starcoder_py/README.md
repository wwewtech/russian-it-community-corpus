---
base_model: bigcode/tiny_starcoder_py
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- russian
- ru
---

# tiny_starcoder_py

Domain-adaptation LoRA adapter for `bigcode/tiny_starcoder_py`, part of the Russian IT
Community Corpus (RICC) LoRA Zoo.

## Model Details

- **Developed by:** wwewtech (Russian IT Community Corpus project)
- **Model type:** LORA adapter (rank r=8, alpha=16, dropout=0.05), task type `CAUSAL_LM`
- **Target modules:** wte, fc1, k_proj, c_attn, c_proj, out_proj, wpe, v_proj, q_proj, fc2
- **Adapter weights:** 6.11 MB, sha256 `ce549aeb8093f442460c73eb04a835c5d6a662cfd6670fa37ddce349e7af7a50`
- **Language(s) (NLP):** Russian (ru), English technical terms
- **License:** MIT (repository license, see `LICENSE`); the base model retains its own license
- **Finetuned from model:** `bigcode/tiny_starcoder_py`

### Model Sources

- **Repository (adapter):** https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/tiny_starcoder_py
- **Repository (code & pipeline):** https://github.com/wwewtech/russian-it-community-corpus
- **Training corpus:** https://huggingface.co/datasets/wwewtech/russian-it-community-corpus

## Uses

### Direct Use

Fine-tuned variant of `bigcode/tiny_starcoder_py` for Russian-language IT discourse
(backend, DevOps, AI/ML, infrastructure). Load with PEFT against the same
base model:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "lora_adapters/tiny_starcoder_py/")
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
