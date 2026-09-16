# Verification and remaining limitations

## Verification updates (2026-09-17)

- **Process-isolated HumanEval execution**: `src/evaluation/official_academic_benchmarks.py` migrated from `ThreadPoolExecutor` to OS-level child process isolation (`subprocess.Popen` with `-I -s` and forced `proc.kill()` on timeout) with optional rootless container execution (`_run_in_container`). Verified by `tests/test_isolated_humaneval_exec.py` (infinite loop safely terminated under timeout without process hangs). Withdrawn academic benchmark scores remain withdrawn pending fresh GPU execution.
- **Hub dataset reconciliation**: added `src/validation/hub_reconciliation.py`, `reports/pinned_hub_manifest.json`, and machine-readable `reports/dataset_reconciliation_report.json` tracking pinned HF Hub revision `81a3495de997b0690f67d175c4ebd9cfdef35b76`. Exact numerical deltas are verified and integrated into `src/monitoring/slo_gate.py` (`hub-reconciliation` check) and CI.
- **mypy strict CI gate**: resolved all 31 boundary type errors in `src/pii/regex_scrubber.py`, `src/pii/ner_scrubber.py`, `src/pii/deep_anonymizer.py`, and `src/validation/validator.py`. `make typecheck-strict` now verifies 26 modules with 0 errors without `--follow-imports=skip`. Added a dedicated blocking `typecheck` job to `.github/workflows/ci.yml` with pinned `mypy==1.8.0`.

## Local verification (2026-09-16 / 2026-09-17)

- Latest full suite before final cleanup: 469 passed, 1 skipped; coverage 94.00%, exit 0. Coverage threshold: 85% of the configured scope, not all GPU code.
- Final targeted regression run: 150 passed (artifact manifest, Hub reconciliation, isolated HumanEval execution, SLO, report consistency, coverage helpers, Docker secret exclusion).
- Ruff passed on all Python files (`ruff check .`, `ruff format --check .`).
- All three local canonical Parquet files matched reports/dataset_manifest.json on re-verification.
- reports/domain_benchmark_100.json remained unchanged in regression runs.

## Limitations / release follow-up

1. **Local vs Hub reconciliation**: The machine-readable divergence report (`reports/dataset_reconciliation_report.json`) makes the delta explicit (-397,739 rows / -34.6 MB for full corpus; +1,696 rows for SFT; +7,000 rows for RAG) against pinned Hub revision `81a3495d`. Before a new major release, execute either full re-upload to Hub under a new pinned revision tag or pull the Hub snapshot to achieve byte identity.
2. **GPU benchmark republishing**: The execution harness is now safe (isolated processes/containers with OS kill), but new academic scores (HumanEval pass@1, RuMMLU CS, PPL, ROUGE) require actual execution on a dedicated GPU node with raw outputs saved to `reports/academic_benchmarks_raw_outputs.jsonl`. Until then, retracted scores remain withdrawn.
3. **Typing coverage scope**: 26 core data, ingestion, analytics, monitoring, PII, and validation modules are strictly typed in CI. GPU/training loops (`src/lora/*`, torch `generate()`) remain excluded from strict mode until typed stubs are introduced.
4. **Docker build execution was not verified locally**. The audit found root .env was not excluded from the build context; .env and .env.* exclusions and regression tests were added. This does not prove absence of secrets in historical images or other file locations. Rotate credentials if an affected image was distributed.

These checks improve maintainability, execution safety, and artifact integrity; they are not a certification of production readiness or a guarantee of zero PII.
