# CI/CD

Workflow: `.github/workflows/ci.yml` — "Data Platform CI & Validation".
Triggers: push/PR to `main`/`master`, `workflow_dispatch`, release
published. Runners: pinned `ubuntu-24.04`.

## Jobs

| Job | What it does | Notes |
|---|---|---|
| Lint & Format | `ruff check` + `ruff format --check` | pinned `ruff==0.17.5` |
| Typecheck (Strict) | `make typecheck-strict` | full codebase, `mypy==2.4.1` |
| Unit Tests | pytest + coverage, matrix 3.11/3.12/3.13/3.14 | `--cov-fail-under=85`, coverage.xml artifact |
| Data Validation & Audits | lock check, adapter cards, synthetic fixtures, validator, real-corpus PII audit (100k sample) | synthetic run is stamped, never SHIP |
| SLO Release Gate (Real Corpus) | regenerates reports from real artifacts, runs the gate | artifact `slo_verdict.json` |
| Security Supply Chain | bandit HIGH gate (0 allowed) + detect-secrets baseline + **pip-audit hard gate** (0 CVEs) | reports as artifacts, 30d |
| HF Token & Cache Check | HF_TOKEN presence + cache layout | push/dispatch only |
| SLO Release Gate | gate verdict from job-local reports | release path |
| Docker Build & Push | builds `ricc-data-platform`, pushes on main | GHCR |
| Release to Hugging Face Hub | finalize/sync/catalog on release | `contents: write` |
| LoRA zoo publish | `gh release upload` of the catalog | `--clobber` |

Every required job must be green before merge or release.

## Pinning Policy

- Action versions are pinned by major (`actions/checkout@v7`,
  `setup-python@v7`, `upload-artifact@v7`) and bumped by dedicated
  dependency PRs.
- Runtime jobs install the portable `requirements.lock.txt`, not the loose
  ranges in `requirements.txt`. Refresh it with
  `python scripts/lock_requirements.py`; CI verifies the resolution and runs
  `pip check`. Unused packages are removed rather than upgraded blindly.
- Tool versions are pinned exactly in `pyproject.toml [dev]` +
  `MYPY_VERSION`/`RUFF_VERSION` env so local == hook == CI.
- After a green unprivileged CI run on a same-repository Dependabot pip PR,
  `dependabot-lock.yml` regenerates the lock and cutoff with the trusted
  default-branch resolver. It then explicitly dispatches CI for the new SHA:
  a `GITHUB_TOKEN` push does not itself create another workflow run. The
  internal recheck performs the ordinary unprivileged dependency checks
  without HF secrets; the real-corpus SLO gate and read-only release
  preflight remain mandatory on pushes to `main` and manual dispatches.

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

`release: published` → SLO gate on real corpus + HF token/write-access
preflight → Docker push → Hub sync. The release downloads the same three
parquet files and manifest that the real-corpus gate validated, verifies the
manifest before any write, materializes registry-compliant adapters and
verifies their SHA-256 digests, and refuses missing or policy-incompliant
weights. It then builds and smoke-tests the wheel and attaches the wheel,
sdist, and model-zoo catalog to the existing GitHub Release.

`workflow_dispatch` is a non-release check and never uploads release assets.
