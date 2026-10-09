#!/usr/bin/env python3
"""
generate_adapter_cards.py — honest per-adapter model cards from registry.json.

Replaces stock Hugging Face template cards (dozens of
`[More Information Needed]` placeholders) and hand-written cards (which
quote stale corpus figures like "2.91M") with cards built exclusively from
verifiable sources:

* ``lora_adapters/registry.json`` — adapter manifest (slug, base model,
  r/alpha/dropout, target modules, bytes, sha256);
* ``reports/HF_MODEL_CARD.md`` — published training regime statement
  (pilot domain-adaptation checkpoints, 50-100 steps);
* ``reports/DATASET_AND_ANALYTICS.md`` — training corpus metrics;
* ``README.md`` benchmark retraction notice — evaluation status.

No numbers are invented: every claim maps to one of the files above.

Run::

    python scripts/generate_adapter_cards.py            # rewrite ALL registry-backed cards
    python scripts/generate_adapter_cards.py --check    # CI: fail if placeholders remain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "lora_adapters" / "registry.json"
PLACEHOLDER = "[More Information Needed]"


CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- russian
- ru
---

# {slug}

Domain-adaptation LoRA adapter for `{base_model}`, part of the Russian IT
Community Corpus (RICC) LoRA Zoo.

## Model Details

- **Developed by:** wwewtech (Russian IT Community Corpus project)
- **Model type:** {peft_type} adapter (rank r={r}, alpha={lora_alpha}, dropout={lora_dropout}), task type `{task_type}`
- **Target modules:** {target_modules}
- **Adapter weights:** {size_mb} MB, sha256 `{sha256}`
- **Language(s) (NLP):** Russian (ru), English technical terms
- **License:** MIT (repository license, see `LICENSE`); the base model retains its own license
- **Finetuned from model:** `{base_model}`

### Model Sources

- **Repository (adapter):** {hf_hub_url}
- **Repository (code & pipeline):** https://github.com/wwewtech/russian-it-community-corpus
- **Training corpus:** https://huggingface.co/datasets/wwewtech/russian-it-community-corpus

## Uses

### Direct Use

Fine-tuned variant of `{base_model}` for Russian-language IT discourse
(backend, DevOps, AI/ML, infrastructure). Load with PEFT against the same
base model:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "{adapter_path}")
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
- **Framework:** PEFT {peft_version} (see `adapter_config.json` in this directory).

## Evaluation

No benchmark scores are claimed for this adapter. Published academic
benchmark numbers were retracted (answer-parsing and column-mapping defects);
see the retraction notice in `README.md` before citing any evaluation figures.

## Citation

```bibtex
@misc{{ricc2026,
  author = {{Russian IT Community Open Research Group}},
  title = {{RICC: Russian IT Community Corpus}},
  year = {{2026}},
  howpublished = {{\\url{{https://github.com/wwewtech/russian-it-community-corpus}}}}
}}
```
"""


def _size_mb(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown"
    return f"{num_bytes / (1024 * 1024):.2f}"


def build_card(entry: dict) -> str:
    return CARD_TEMPLATE.format(
        slug=entry["slug"],
        base_model=entry["base_model"],
        peft_type=entry.get("peft_type") or "LORA",
        r=entry.get("r"),
        lora_alpha=entry.get("lora_alpha"),
        lora_dropout=entry.get("lora_dropout"),
        task_type=entry.get("task_type") or "CAUSAL_LM",
        target_modules=", ".join(entry.get("target_modules") or []) or "n/a",
        size_mb=_size_mb(entry.get("safetensors_bytes")),
        sha256=entry.get("safetensors_sha256") or "unknown",
        hf_hub_url=entry.get("hf_hub_url") or "",
        adapter_path=entry.get("adapter_path") or entry.get("local_path") or "",
        peft_version=entry.get("peft_version") or "unknown",
    )


def all_card_paths() -> list[Path]:
    return sorted(REPO_ROOT.glob("lora_adapters/*/README.md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if placeholder cards remain (CI mode)")
    args = parser.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_slug = {e["slug"]: e for e in registry["adapters"]}

    cards = all_card_paths()
    if args.check:
        placeholders = [p for p in cards if PLACEHOLDER in p.read_text(encoding="utf-8")]
        for path in placeholders:
            print(f"PLACEHOLDER CARD: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(f"{'FAIL' if placeholders else 'OK'}: {len(placeholders)} placeholder card(s) among {len(cards)} cards")
        return 1 if placeholders else 0

    written = 0
    for card_path in cards:
        slug = card_path.parent.name
        entry = by_slug.get(slug)
        if entry is None:
            print(f"SKIP {slug}: not in registry.json", file=sys.stderr)
            continue
        # NOTE: newline="\n" is mandatory. Path.write_text() without it translates
        # "\n" to os.linesep (CRLF on Windows), and `ruff format --check .`
        # (line-ending = "lf") flags every card as unformatted on Windows.
        with card_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(build_card(entry))
        written += 1
    print(f"rewrote {written} card(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
