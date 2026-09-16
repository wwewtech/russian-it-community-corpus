# mypy --strict Rollout

This document tracks the per-module migration to `mypy --strict` referenced by
`pyproject.toml` ([[tool.mypy.overrides]]) and `make typecheck-strict`.

## Status (2026-09-16)

**Paths selected by the existing `make typecheck-strict` target (not a claim that the whole import graph currently passes):**

`src/config.py`, `src/bootstrap.py`, `src/ingestion/schema.py`, `src/ingestion/loader.py`,
`src/analytics/metrics.py`, `src/analytics/network.py`, `src/analytics/report_generator.py`,
`src/analytics/engine.py`, `src/taxonomy/classifier.py`, `src/taxonomy/tagger.py`,
`src/deduplication/exact_dedup.py`, `src/deduplication/minhash_lsh.py`,
`src/monitoring/drift.py`, `src/monitoring/sft_quality.py`, `src/monitoring/slo_gate.py`,
`src/rag/rag_pipeline.py`, `src/rag/__init__.py`, `src/evaluation/__init__.py`,
`src/lora/__init__.py`, `src/graph/__init__.py` (status 2026-08-29: `src.analytics.engine`
and `src.graph.*` were re-enabled after passing in isolation).

Measured locally on 2026-09-16: `python -m mypy src/validation/artifact_manifest.py
--strict --ignore-missing-imports --follow-imports=skip` passes, as does the same
isolated check for `src/monitoring/slo_gate.py`. This skips imported code; it is
not equivalent to passing the complete dependency graph. The manifest uses
`Any` at JSON validation boundaries, narrowed to TypedDict after runtime checks.

Without `--follow-imports=skip`, the manifest check reports 31 errors across
`src/pii/regex_scrubber.py`, `src/pii/ner_scrubber.py`,
`src/pii/deep_anonymizer.py`, and `src/validation/validator.py` (untyped methods,
callbacks, and a bare Pattern type). Next measurable step: annotate these
boundaries and rerun without skip. Do not suppress them merely to claim success.

## Still excluded (via `[[tool.mypy.overrides]]` in `pyproject.toml`)

| Module group | Reason | Re-enable path |
|---|---|---|
| `src.evaluation.*` | transformers/evaluate/sklearn stub gaps surface dozens of non-actionable "no overload variant" errors | Type the pure helpers first (`parse_mc_answer`, `execute_humaneval_code` already have precise signatures), then gate the suites |
| `src.lora.*` | PEFT/TRL callback plumbing, heavy third-party stubs | Type `batch_train_zoo` config handling separately from the trainer loop |
| `src.inference` | torch `generate()`/`pipeline()` untyped overloads | Extract pure text/formatting helpers and type them; keep the I/O loop last |
| `src.pipeline` | orchestrates torch-backed stages end-to-end | Becomes easy once the stages above are typed |
| `src.exporter.finalize_sync_all` | cross-store sync glue, dict-heavy payloads | Split pure index/markdown builders (already strict-friendly) from upload I/O |

## How to re-enable a module

1. Run `python -m mypy src/<module> --strict --ignore-missing-imports` and fix or
   explicitly type the surfaced errors (prefer `TypedDict`/dataclasses over loose dicts).
2. Remove the module from the `module = [...]` list under `[[tool.mypy.overrides]]`
   in `pyproject.toml`.
3. Add the module path to the `typecheck-strict` target in the `Makefile`.
4. Run `make typecheck-strict` and record the actual exit code and mypy version.
   The current GitHub workflow does not run mypy, and the dev dependency is a
   lower bound rather than a pinned version. Do not describe this as a CI gate
   until an explicit workflow step is added.
5. When testing an excluded module, first remove its matching ignore_errors
   override in a candidate change; otherwise a passing command can hide errors.

## Non-goals

- Typing the GPU/training entry points is deliberately deferred until their pure
  logic is extracted; forcing stubs for torch/peft/trl buys no safety today.
- `tests/` are not strict-typed; they are covered by ruff and runtime assertions.
