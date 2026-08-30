# Operations

This runbook covers the day-2 procedures: how to rebuild the dataset,
regenerate the LoRA registry, push to Hugging Face, re-run CI, and
respond to common incidents. It complements the
[architecture document](architecture.md).

> **Audience:** maintainers with write access to the HF Hub repos and
> the GitHub repository. If you are a first-time contributor, read
> [CONTRIBUTING.md](../CONTRIBUTING.md) first.

---

## 1. Local environment

### Required tools

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 / 3.12 / 3.13 | CI matrix mirrors these three. Local dev typically 3.12. |
| Git | ≥ 2.40 | for LFS weights and the DVC remotes. |
| `ruff` | pinned to **0.16.2** | matches the version used in CI; do not bump casually. |
| `pytest` | ≥ 8.0 | with the `pytest-cov` plugin. |
| DVC | ≥ 3.0 | optional; only needed if you re-run the DVC pipeline. |
| Streamlit | ≥ 1.31 | for `app.py`. |

Install everything pinned:

```bash
python -m pip install -U pip
pip install -r requirements.txt
python -m pip install ruff==0.16.2
```

### Environment variables

| Variable | Required for | Default |
|----------|--------------|---------|
| `HF_TOKEN` | HF Hub push (sync_to_hub / finalize_sync_all) | falls back to `~/.huggingface/token` |
| `HF_HOME` | model cache location | `~/.cache/huggingface` |
| `RICC_PARQUET_DIR` | override dataset output location | `dataset_output/parquet` |
| `RICC_LORA_DIR` | override LoRA output location | `lora_adapters` |
| `RUN_STREAMLIT_SMOKE` | opt-in UI smoke test in CI | unset (test is skipped) |

**NEVER** put `HF_TOKEN` into `.env`, source code, docstrings, or
chat — see [ADR 0001](adr/0001-hf-token-handling.md).

---

## 2. Common workflows

### 2.1 Run the full pipeline (local)

```bash
python cli.py run                    # all 7 stages
python cli.py run --stages 6         # only stage 6 (export)
python cli.py run --stages 1 2 3     # stages 1–3
```

Outputs land in `dataset_output/{parquet,jsonl}/`. Intermediate
artifacts (pre-dedup, pre-taxonomy) live in `intermediate/` and are
gitignored.

### 2.2 Regenerate the LoRA registry

The `lora_adapters/registry.json` is the manifest pushed to the HF Hub.
It must be regenerated whenever a new adapter is added or removed:

```bash
python scripts/generate_lora_registry.py
git add lora_adapters/registry.json
git commit -m "chore(lora): regenerate registry after adding <name>"
```

The script scans `lora_adapters/*/adapter_config.json` and **skips
directories without a valid `adapter_config.json`** (i.e. incomplete
scaffolds). This is intentional — the registry is a list of *deployable*
adapters, not of every directory that ever existed on disk.

### 2.3 Push the dataset and LoRA zoo to the Hub

```bash
# Dry-run first — never skip this step on first use:
python scripts/sync_to_hub.py
python scripts/finalize_sync_all.py --dry-run

# When the dry-run output looks correct:
python scripts/sync_to_hub.py --apply
python scripts/finalize_sync_all.py --apply
```

Both scripts are idempotent: re-running them only uploads files that
changed on the remote.

### 2.4 Launch the Streamlit Data Studio

```bash
streamlit run app.py --server.port 8501
```

The app reads from `dataset_output/`, `lora_adapters/registry.json`,
and `reports/`. If any of those are missing, the corresponding panel
shows a "data not built yet" hint instead of crashing.

### 2.5 Launch the interactive chat / RAG runtime

```bash
# Default: Qwen 1.5B + 7B LoRA + RAG
python -m src.inference

# Disable LoRA, base model only
python -m src.inference --adapter none

# Disable RAG
python -m src.inference --no-rag

# Custom model + shorter replies
python -m src.inference --model Qwen/Qwen2.5-7B-Instruct --max-tokens 256
```

A CUDA-capable GPU is recommended for any non-base model. The base
1.5B model runs on CPU but is slow.

---

## 3. CI

The CI pipeline is in `.github/workflows/ci.yml` and has two jobs:

### 3.1 `lint`

Runs on every push/PR to `main` and `master`.

1. `ruff check .` — must report zero findings.
2. `ruff format --check .` — must report no files needing
   reformatting.

If `ruff format --check` complains, run `ruff format .` locally and
commit the result.

### 3.2 `test-and-validate`

Runs on a 3-version matrix (3.11, 3.12, 3.13). Steps:

1. `pip install -r requirements.txt`
2. `python -m pytest -q` — full test suite, including the 60% coverage
   floor.
3. `python cli.py benchmark` — domain benchmark generation.
4. **Conditional** PII red-team audit — only runs when
   `dataset_output/parquet/full_clean_messages.parquet` is present in
   the checkout.
5. **Conditional** `python cli.py validate` — only runs when the same
   parquet is present.
6. **Conditional** SFT dialogue quality regression — only runs when
   `dataset_output/parquet/sft_dialogues.parquet` is present.

Steps 4–6 are skipped on PR builds because the parquet files are
gitignored; they are designed to run on a self-hosted runner that has
the artifacts pre-staged. To re-enable them on a regular runner, see
[§ 3.4](#34-synthetic-parquet-fixtures-for-pr-builds).

### 3.3 Streamlit AppTest smoke test

Lives in `tests/test_app_streamlit_smoke.py` and is gated on the
`RUN_STREAMLIT_SMOKE` environment variable. It is **not** run by
default — Streamlit's AppTest framework adds ~10–15 s to the suite
and brings in `streamlit` as a hard test dependency, which we want to
keep optional.

To run it locally:

```bash
RUN_STREAMLIT_SMOKE=1 python -m pytest tests/test_app_streamlit_smoke.py -v
```

### 3.4 Synthetic parquet fixtures for PR builds

The three conditional CI steps (§ 3.2) are skipped on PR builds because
the parquet files are gitignored. There are two ways to unskip them
without committing the (very large) datasets:

**Option A — self-hosted runner with pre-staged artifacts.**
Recommended for the `main` branch. Cache
`dataset_output/parquet/{full_clean_messages,sft_dialogues,rag_knowledge_base}.parquet`
on the runner's local disk and check out the repo on top.

**Option B — synthetic fixture generator.** For PR builds, run
`scripts/build_synthetic_parquet.py` (a small in-repo script that
emits a 100-row parquet with the same schema) before the conditional
steps. The audits then run against the synthetic file in well under
a minute. This is the path forward if you want a fully green matrix
on PRs without self-hosted infrastructure.

A prototype for option B is shipped in v12.0.3. See
`scripts/build_synthetic_parquet.py` and the `synthetic-fixture` step
in `.github/workflows/ci.yml`.

### 3.5 Required status checks

For a merge to `main`, both of the following must be green:

- `lint`
- `test-and-validate` on **all three** Python versions.

The conditional PII / validate / SFT steps are *informational* on
PRs — they appear as "skipped" in the GitHub UI but do not block
merge. They MUST be green on a tagged release.

---

## 4. Incident response

### 4.1 PII leak detected post-publish

1. **Revoke the affected LoRA** in `lora_adapters/registry.json` by
   removing the entry, then re-running
   `python scripts/generate_lora_registry.py`.
2. **Re-run the PII red-team** against the rebuilt dataset:
   `python -c "from pathlib import Path; from src.validation.pii_redteam import RedTeamPIIAuditor; RedTeamPIIAuditor(Path('dataset_output/parquet/full_clean_messages.parquet')).generate_audit_certificate(Path('reports/pii_validation_report.json'))"`
3. **Push a hotfix to the Hub:**
   `python scripts/sync_to_hub.py --apply` and
   `python scripts/finalize_sync_all.py --apply`.
4. **Open a post-mortem issue** referencing the dataset card and the
   certificate in `reports/pii_validation_report.json`. Do not delete
   the certificate — it is the audit trail.

### 4.2 HF token leaked into chat / commit / log

1. **Revoke immediately** at https://huggingface.co/settings/tokens.
2. **Mint a new token** with the same scope (write to
   `wwewtech/russian-it-community-corpus` and
   `wwewtech/russian-it-community-lora`).
3. **Rotate the secret** in GitHub Actions (`Settings → Secrets and
   variables → Actions → HF_TOKEN`) and in any self-hosted runner
   environment variables.
4. **Re-read [ADR 0001](adr/0001-hf-token-handling.md)** and verify
   no other surface stores the token.

### 4.3 `ruff format` complains in CI

1. Run `ruff format .` locally.
2. Inspect the diff with `git diff`.
3. Commit with a message like
   `style: apply ruff 0.16.2 auto-format` and push.

Do not silence `ruff format` with `# fmt: off` in new code; the
auto-formatter is the source of truth.

### 4.4 Coverage dropped below 60%

1. `python -m pytest --cov=src --cov=app_helpers --cov-report=term-missing`
2. Identify the modules with the largest drop in `Missing`.
3. Add tests in the corresponding `tests/test_<module>.py`.
4. Re-run and confirm `Required test coverage of 60% reached.`

### 4.5 A CI step is stuck "skipped" forever

1. Check the step's `if:` condition in `.github/workflows/ci.yml`.
2. For PII / validate / SFT steps, verify the
   `hashFiles('dataset_output/parquet/...')` predicate resolves
   correctly on the runner. The parquet files are gitignored, so
   they will *never* resolve on a vanilla GitHub-hosted runner — see
   [§ 3.4](#34-synthetic-parquet-fixtures-for-pr-builds).

---

## 5. Release checklist (v12.x and beyond)

For each tagged release:

1. [ ] All commits on `main` since the last tag have a CHANGELOG entry.
2. [ ] `pytest` is green on 3.11, 3.12, 3.13.
3. [ ] `ruff check .` is clean.
4. [ ] `ruff format --check .` is clean.
5. [ ] PII red-team certificate (`reports/pii_validation_report.json`)
       is dated within the last 7 days and shows `passed: true`.
6. [ ] SFT quality report
       (`reports/sft_quality_report.json`) shows no sub-score below
       its floor.
7. [ ] `python scripts/sync_to_hub.py --apply` has been run and the
       remote matches local byte-for-byte.
8. [ ] `python scripts/finalize_sync_all.py --apply` has been run
       and `LoRA_ZOO.md` on the Hub reflects the local registry.
9. [ ] Docker image rebuilt and pushed
       (`docker build -t wwewtech/ricc:<tag> .`).
10. [ ] CHANGELOG.md updated with the new version section.

---

## 6. Where to ask for help

- **Code questions:** open an issue, tag `@maintainers`.
- **Security disclosure:** see [SECURITY.md](../SECURITY.md) — do not
  open a public issue.
- **Hugging Face Hub problems:** the
  [`huggingface_hub` documentation](https://huggingface.co/docs/huggingface_hub)
  is the canonical reference; the Hub status page is at
  https://status.huggingface.co.
- **CI failures:** the `gh run view <id> --log` command shows the
  full log; the most common cause is a dependency-version drift
  between local and CI (run `pip install -r requirements.txt` to
  re-sync).
