# CI/CD

Workflow: `.github/workflows/ci.yml` — "Data Platform CI & Validation".
Triggers: push/PR to `main`/`master`, `workflow_dispatch`, release
published. Runners: pinned `ubuntu-24.04`.

## Jobs

| Job | What it does | Notes |
|---|---|---|
| Lint & Format | `ruff check` + `ruff format --check` | pinned `ruff==0.16.2` |
| Typecheck (Strict) | `make typecheck-strict` | 43 modules, `mypy==2.3.1` |
| Unit Tests | pytest + coverage, matrix 3.11/3.12/3.13/3.14 | `--cov-fail-under=85`, coverage.xml artifact |
| Data Validation & Audits | lock check, adapter cards, synthetic fixtures, validator, real-corpus PII audit (100k sample) | synthetic run is stamped, never SHIP |
| SLO Release Gate (Real Corpus) | regenerates reports from real artifacts, runs the gate | artifact `slo_verdict.json` |
| Security Supply Chain | bandit HIGH gate (0 allowed) + detect-secrets baseline + **pip-audit hard gate** (0 CVEs) | reports as artifacts, 30d |
| HF Token & Cache Check | HF_TOKEN presence + cache layout | push/dispatch only |
| SLO Release Gate | gate verdict from job-local reports | release path |
| Docker Build & Push | builds `ricc-data-platform`, pushes on main | GHCR |
| Release to Hugging Face Hub | finalize/sync/catalog on release | `contents: write` |
| LoRA zoo publish | `gh release upload` of the catalog | `--clobber` |

Typical full run: **15–18 minutes**. All 11 jobs green is the merge bar.

## Pinning Policy

- Action versions are pinned by major (`actions/checkout@v5`,
  `setup-python@v6`, `upload-artifact@v7`) and bumped by dedicated
  dependency PRs.
- Tool versions are pinned exactly in `pyproject.toml [dev]` +
  `MYPY_VERSION`/`RUFF_VERSION` env so local == hook == CI.

## Security Gates

- **bandit**: HIGH severity fails the job; the full JSON report is
  uploaded for triage. Current baseline: 0 HIGH.
- **detect-secrets**: fails on findings not in `.secrets.baseline`
  (UTF-8, regenerated 2026-10-11); baseline guarded by
  `tests/test_secrets_baseline.py`.
- **pip-audit**: hard gate since 12.2.0 — any known CVE in the
  requirements tree fails; missing report fails closed.

## Docs Gate

`mkdocs build --strict` runs in CI (docs job): every nav entry must
resolve and the build must be warning-free.

## Release Flow

`release: published` → SLO gate on real corpus → Docker push → Hub sync
(`scripts/sync_to_hub.py --apply`, sanctioned token path per
`docs/adr/0001-hf-token-handling.md`) → `gh release upload` the model
zoo catalog.
