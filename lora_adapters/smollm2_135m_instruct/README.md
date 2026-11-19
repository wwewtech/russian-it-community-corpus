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
- **Target modules:** q_proj, o_proj, k_proj, v_proj
- **Adapter weights:** 7.06 MB, sha256 `c1fbaaa3ab04313726a4c505440817e2cb967f568ba7019cd6d9240e92fa0c16`
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
- **Training regime:** full-corpus domain adaptation — 21,209 optimizer steps over the full RICC SFT corpus with prompt masking; 100.0% of one epoch (honest coverage from `training_meta.json`, local adapter directory).
- **Framework:** PEFT 0.17.1 (see `adapter_config.json` in this directory).

## Evaluation

Held-out perplexity measured after training (100 dialogues / 22,647 tokens from the 1,000-dialogue tail excluded from training; base perplexity 13.9678 → LoRA perplexity **7.2630**, mean NLL 2.6368 → 1.9828). This is an honest full-sequence language-model metric on unseen dialogue text; it is not a measure of chat quality.

Fresh fixed-harness academic subset on 2026-11-19 (container-isolated execution): base 16.27 → LoRA **12.89 perplexity**; HumanEval base 10.0% / LoRA 2.5% / hybrid 0% of 40 tasks; RuMMLU base 28.0% / LoRA 0.0% / hybrid 0.0% of 50 questions. The adverse RuMMLU gap is significant **p ≈ 0.0001** — do not omit it or present the adapter as better. Older broken academic numbers remain **retracted**; fresh raw outputs alone carry status `VALID` with provenance and caveats. An external expert review was not performed. Older academic benchmark numbers (HumanEval / RuMMLU) remain **retracted** for the whole zoo (answer-parsing and column-mapping defects; see the retraction notice in the repository `README.md`). Do not cite those numbers.

## Citation

```bibtex
@misc{ricc2026,
  author = {Russian IT Community Open Research Group},
  title = {RICC: Russian IT Community Corpus},
  year = {2026},
  howpublished = {\url{https://github.com/wwewtech/russian-it-community-corpus}}
}
```
