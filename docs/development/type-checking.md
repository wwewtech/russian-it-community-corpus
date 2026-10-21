# Type Checking — mypy --strict Everywhere

As of 12.2.0 the strict rollout is **complete**: every module under `ricc/`
passes `mypy --strict`, and the `[[tool.mypy.overrides]] ignore_errors`
block in `pyproject.toml` is gone.

## Gates

| Gate | Command | Modules |
|---|---|---|
| Local / CI | `make typecheck-strict` | 43 |
| Pre-commit | `mypy-strict` hook (`.pre-commit-config.yaml`) | 43 (same list) |
| CI job | `Typecheck (Strict)` → `make typecheck-strict` | 43 |

The Makefile list and the pre-commit hook list must stay in sync — both
mirror the full `ricc/` module set.

## The `cast(Any, ...)` Pattern

Heavy ML libraries (transformers, peft) ship incomplete or untyped stubs;
calling e.g. `AutoTokenizer.from_pretrained()` under `--strict` fails with
`no-untyped-call`. The single sanctioned pattern (established in
`ricc/inference.py`, used everywhere since):

```python
from typing import Any, cast

tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True)
text: str = cast(str, tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Rules:

- `cast(Any, ...)` fences the **call**; the result is annotated with the
  concrete type you expect (`Any`, `str`, `dict[str, Any]`).
- Never write `# type: ignore` — a bare ignore hides errors in the whole
  expression. (Pre-existing `type: ignore[misc]` in
  `ricc/orchestration/prefect_flow.py` predates the policy.)
- Prefer annotating data tables at the declaration site
  (`HUMANEVAL_TASKS: list[dict[str, Any]] = [...]`) — this fixes the type
  of every downstream consumer at once, instead of casting each access.

## Dict Payloads and Fallback Branches

When a function branches and the fallback branch reads values from a
`dict[str, Any]` accumulator (which mypy sees as `object`), annotate the
locals up front and cast at the read:

```python
l_out: str
l_lat: float
if lora_model:
    l_out, l_lat, l_tps, l_vram = generate_response(lora_model, prompt)
    ...
else:
    bprev = base_evals[idx - 1]
    l_out = cast(str, bprev["out"])
    l_lat = cast(float, bprev["latency"])
```

This is the pattern used in `deep_multi_experiment_suite.py` and
`enterprise_production_eval_suite.py`.

## Adding a New Module

1. Write it strict from the start (annotate every function signature).
2. Add the path to **both** lists: `Makefile` (`typecheck-strict`) and
   `.pre-commit-config.yaml` (`mypy-strict` hook args).
3. Run `make typecheck-strict` — must pass with the new module included.
4. If the module needs a lazy third-party import, import it under
   `if TYPE_CHECKING:` for annotations and keep the runtime import
   local (see `ricc/observability/__init__.py`).

## History

| Date | Scope | Errors fixed |
|---|---|---|
| 2026-09-17 | engine, graph, inference, pipeline, finalize_sync_all, statistical_power, benchmark_comparator | (initial set, 31 modules) |
| 2026-10-12 | src.lora.\* (5 modules), official_academic_benchmarks, benchmark_comparator annotation fix | 41 + 2 |
| 2026-10-14 | deep_multi_experiment_suite, enterprise_production_eval_suite, sota_expert_benchmark_showcase | 89 + 34 + 29 |

Version history lives in `pyproject.toml` comments and the
[CHANGELOG](https://github.com/wwewtech/russian-it-community-corpus/blob/main/CHANGELOG.md).
