# ADR 0009: Registry-driven production inference

## Status

Accepted

## Context

After the verified release pipeline, interactive inference still selected an
unrelated historical Qwen/7B adapter by default. Base and adapter sources were
not pinned together, registry entries had no single runtime representation,
and the release catalog had two independent renderers. Model inputs were moved
to any visible CUDA device rather than the actual device of a sharded model.
Some evaluation code could also sample training-seen text when a held-out tail
was absent.

## Decision

- `ricc.lora.production` resolves a registry entry into an `AdapterSpec` with
  base/adapter repositories, revisions and a Hub subfolder. The default is the
  sole `policy_compliant` full-epoch adapter.
- Local weights are preferred. Hub loading uses an immutable subfolder and
  revision. Historical adapters without immutable remote provenance require an
  explicitly selected local weights checkout and emit an experimental warning.
- `ricc.ml_runtime` loads compatible base/adapter/tokenizer sources and moves
  inputs to the actual model placement, including `hf_device_map` layouts.
- Evaluation suites share this loader and record whether their pair is pinned
  registry provenance or an explicit experimental pair. Perplexity requires a
  complete held-out tail and never silently substitutes training-seen data.
- The preflight and live release use one catalog renderer. Local weight size
  and SHA-256 remain exact instead of being reconstructed from rounded MiB.

## Consequences

The default interactive experience now selects the verified SmolLM2 adapter,
not an arbitrary historical heavyweight model. Experimental training and
inference remain possible without declaring every historical adapter final.
Existing explicit local base/path calls retain a compatibility path; they do
not become implicit Hub requests.

This ADR does not change training policy, dataset revisions or historical Hub
artifacts. A production adapter registry with multiple compliant adapters will
need an explicit selection policy rather than silently choosing the first.
