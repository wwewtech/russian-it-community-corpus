# ADR 0002: Complete mypy --strict Rollout

**Status:** Accepted
**Date:** 2026-10-14
**Authors:** RICC maintainers
**Supersedes:** —

## Context

The codebase started with mypy in "slow mode": strict checks only for a
hand-picked module list (`make typecheck-strict`), with
`[[tool.mypy.overrides]] ignore_errors` masking `src.evaluation.*` and
`src.lora.*` — the heaviest, most failure-prone code in the repo (GPU
training, benchmark harnesses).

This was a deliberate trade-off, not an accident: transformers/peft ship
weak stubs, and a big-bang strict rollout would have surfaced hundreds of
errors in code nobody was ready to refactor. The cost was real, though:

- the ignore block hid genuinely broken annotations (2 latent
  `arg-type` errors in `benchmark_comparator.py` were found the day the
  wildcard was narrowed);
- untyped dict payloads (`Collection[str]` inferred where
  `list[dict[str, Any]]` was meant) produced silent runtime-shape bugs
  that only surfaced in GPU-time evaluation runs;
- "strict, except these modules" drifts: modules rotted while excluded,
  and nobody could tell which exclusions were load-bearing.

## Decision

1. **Strict is the terminal state.** Every module under `src/` passes
   `mypy --strict`; the `ignore_errors` override block is removed
   entirely (12.2.0).
2. **The rollout is per-module, gated, and monotonic.** Modules enter
   `make typecheck-strict` one at a time, only when clean. Once in, never
   out. History: 31 modules (2026-09-17) → 40 (2026-10-12, phase 1:
   `src.lora.*` + academic benchmarks, 43 errors) → 43 (2026-10-14,
   phase 2: the three heavy eval suites, 152 errors).
3. **`cast(Any, ...)` is the single sanctioned fence** for untyped
   third-party calls (transformers/peft). Bare `# type: ignore` is banned
   on new code — it hides whole-expression errors; a cast documents the
   expected type and survives stub improvements.
4. **Data tables are annotated at the declaration site**
   (`HUMANEVAL_TASKS: list[dict[str, Any]] = [...]`), which fixes every
   downstream consumer at once instead of casting each access.
5. **The gate is enforced in three places that must stay in sync:**
   `make typecheck-strict`, the pre-commit `mypy-strict` hook
   (`pass_filenames: false` — appending staged files caused mypy
   "Duplicate module" errors), and the CI `Typecheck (Strict)` job.

## Consequences

- New modules must be written strict from the start and added to both
  lists (Makefile + pre-commit). The full recipe lives in
  `docs/development/type-checking.md`.
- Stub fixes from transformers/peft releases can turn `cast(Any, ...)`
  calls into no-ops — casts are cheap to leave and easy to bulk-remove
  later; no action required.
- Type errors are caught at PR time, not in GPU-time evaluation runs.

## References

- Rollout history and patterns: `docs/development/type-checking.md`
- CHANGELOG 12.2.0, "Changed" section
