# Type Checking — mypy --strict Everywhere

As of 12.2.0 the strict rollout is **complete**: every module under `ricc/`,
every root entry point (`cli.py`, `main.py`, `app_helpers.py`, `chat_ui.py`),
and every script under `scripts/` passes `mypy --strict`, with no
`[[tool.mypy.overrides]] ignore_errors` blocks in `pyproject.toml`.

## Gates

| Gate | Command | Scope |
|---|---|---|
| Local / CI | `make typecheck-strict` | `ricc` (all 67 modules), root entry points, `scripts/` |
| Pre-commit | `mypy-strict` hook (`.pre-commit-config.yaml`) | Same directory/module targets |
| CI job | `Typecheck (Strict)` → `make typecheck-strict` | Full strict gate |

Because `make typecheck-strict` checks the `ricc` package and `scripts`
directory directly rather than a hand-maintained filename list, new modules are
automatically covered by strict type checking from their first commit.

## The `cast(Any, ...)` Pattern

Heavy ML libraries (transformers, peft) ship incomplete or untyped stubs;
calling e.g. `AutoTokenizer.from_pretrained()` under `--strict` fails with
`no-untyped-call`. Centralised helpers in `ricc/ml_runtime.py` fence those
calls behind typed wrappers so callers do not need to repeat `cast(Any, ...)`:

```python
from typing import Any, cast

tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True)
text: str = cast(str, tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Rules:

- `cast(Any, ...)` fences the **call**; the result is annotated with the
  concrete type you expect (`Any`, `str`, `dict[str, Any]`).
- Never write bare `# type: ignore` — it hides errors across the whole
  expression.
- Prefer annotating data tables at the declaration site
  (`HUMANEVAL_TASKS: list[dict[str, Any]] = [...]`) so every downstream consumer
  receives concrete types automatically.

## Adding a New Module

1. Write it strict from the start (annotate every function signature).
2. Run `make typecheck-strict` — `ricc/` and `scripts/` are scanned
   automatically.
3. If the module needs a lazy third-party import, import it under
   `if TYPE_CHECKING:` for annotations and keep the runtime import local
   (see `ricc/observability/__init__.py`).
