# Contributing

## Setup

```bash
git clone https://github.com/wwewtech/russian-it-community-corpus.git
cd russian-it-community-corpus
python -m venv .venv && .venv\Scripts\activate   # Windows
make install          # editable install + dev tools
pre-commit install    # hooks: ruff, mypy-strict, detect-secrets, yaml/toml checks
```

## Daily Loop

```bash
make lint       # ruff check (same pinned version as CI)
make format     # ruff format
make typecheck-strict   # mypy --strict over all 43 src modules
make test       # pytest with coverage gate
```

All four must pass locally before pushing — CI runs the same pinned
versions (`ruff==0.16.2`, `mypy==2.3.1`).

## Rules of the Codebase

1. **Strict typing is mandatory.** New modules go into both strict lists
   (Makefile + pre-commit hook). See
   [Type Checking](type-checking.md) for the `cast(Any, ...)` pattern.
2. **Single formatter**: ruff format. (The Black fallback hook was
   removed — it disagreed with ruff and blocked commits.)
3. **No bare `# type: ignore`** on new code.
4. **Errors are typed.** Raise from `ricc.errors` hierarchy; raw torch/HF
   exceptions in train/eval entry points go through
   `record_classified_ml_error()`.
5. **Metrics are deliberate.** A new metric name must be added to the
   observability module — the guard tests fail if dashboards/rules
   reference unregistered names.
6. **Security gates are hard.** bandit HIGH = 0, pip-audit = 0 vulnerable
   packages, detect-secrets = no new findings. A new HIGH finding blocks
   merge; fix or (with justification) add a scoped `# nosec` with the
   bare-code rule.
7. **Docs build in CI** (`mkdocs build --strict`): every nav entry must
   resolve.

## Commits

Conventional, scope-tagged: `feat(2-4): ...`, `sec(5): ...`,
`fix(release): ...`, `types(2-final): ...`. Version bumps touch all four
places — `pyproject.toml`, `CITATION.cff`, `ricc/__init__.py`,
`ricc/ricc/__init__.py` — enforced by `test_version_consistency`.

## Pull Requests

- Base on `main`; CI must be green (11 jobs, see [CI/CD](ci-cd.md)).
- Tests: add for every behavior change; coverage gate is 85% with
  GPU-only suites omitted from the gate.
- Update `CHANGELOG.md` (Unreleased → version at release time) and the
  relevant docs pages.
