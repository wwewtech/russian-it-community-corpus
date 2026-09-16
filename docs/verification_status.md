# Verification and remaining limitations

## Local verification (2026-09-16)

- Latest full suite before final cleanup: 469 passed, 1 skipped; coverage 94.00%, exit 0. Coverage threshold: 85% of the configured scope, not all GPU code.
- Final targeted regression run: 135 passed (artifact manifest, SLO, report consistency, coverage helpers, Docker secret exclusion).
- Ruff passed on the changed Python files.
- Isolated strict mypy passed for artifact_manifest and slo_gate with --ignore-missing-imports --follow-imports=skip. This does not verify the complete import graph.
- All three local canonical Parquet files matched reports/dataset_manifest.json on re-verification.
- reports/domain_benchmark_100.json remained unchanged in the full test run (recorded SHA-256 before and after: fa32e130592d53ec8bd6300e9faaa9a846a70179a8ed086ea98abbd8f5b14eaf).

Redundant local audit logs, temporary test directories and runners were removed during final cleanup. These results describe local checks, not a completed remote CI run.

## Limitations / release follow-up

1. The artifact manifest proves local byte identity, not original source lineage, training provenance, or equivalence to a pinned Hub revision. Reconcile local and published datasets before release; do not regenerate the manifest merely to bypass a mismatch.
2. No new trustworthy GPU benchmark scores were produced. The existing HumanEval harness uses a thread timeout that cannot terminate executing generated code. Do not run untrusted generated code in the host process. A future evaluation needs process/container isolation, enforced resource limits, pinned model/dataset revisions and retained raw outputs before publishing metrics. Withdrawn scores remain withdrawn.
3. Strict typing remains incremental; see mypy_strict_rollout.md. GPU/training paths are not fully verified by the CPU suite.
4. Docker build execution was not verified locally. The audit found root .env was not excluded from the build context; .env and .env.* exclusions and regression tests were added. This does not prove absence of secrets in historical images or other file locations. Rotate credentials if an affected image was distributed.

These checks improve maintainability and artifact integrity; they are not a certification of production readiness or a guarantee of zero PII.
