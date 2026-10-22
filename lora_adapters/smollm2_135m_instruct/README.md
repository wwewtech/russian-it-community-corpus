---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- russian
- ru
---

# smollm2_135m_instruct

Domain-adaptation LoRA adapter for `HuggingFaceTB/SmolLM2-135M-Instruct`, part of the Russian IT
Community Corpus (RICC) LoRA Zoo.

## Model Details

- **Developed by:** wwewtech (Russian IT Community Corpus project)
- **Model type:** LORA adapter (rank r=16, alpha=32, dropout=0.05), task type `CAUSAL_LM`
- **Target modules:** v_proj, o_proj, k_proj, q_proj
- **Adapter weights:** 7.06 MB, sha256 `ebf061344f9388d97e050a4cd89e617ed35194d013db865118b12803ae36e10a`
- **Language(s) (NLP):** Russian (ru), English technical terms
- **License:** MIT (repository license, see `LICENSE`); the base model retains its own license
- **Finetuned from model:** `HuggingFaceTB/SmolLM2-135M-Instruct`

### Model Sources

- **Repository (adapter):** https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/smollm2_135m_instruct
- **Repository (code & pipeline):** https://github.com/wwewtech/russian-it-community-corpus
- **Training corpus:** https://huggingface.co/datasets/wwewtech/russian-it-community-corpus

## Uses

### Direct Use

Fine-tuned variant of `HuggingFaceTB/SmolLM2-135M-Instruct` for Russian-language IT discourse
(backend, DevOps, AI/ML, infrastructure). Load with PEFT against the same
base model:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "lora_adapters/smollm2_135m_instruct/")
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
- **Training regime:** full-corpus domain adaptation — 21,420 optimizer steps over the full RICC SFT corpus with prompt masking; 100.0% of one epoch (honest coverage from `training_meta.json`, local adapter directory).
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
