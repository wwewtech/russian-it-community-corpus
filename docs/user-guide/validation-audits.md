# Validation & Audits

`src/validation/` is the quality police: schema checks, zero-PII proof,
artifact integrity and Hub reconciliation. All checks feed the
[SLO gate](slo-gate.md).

## Dataset Validator

`DatasetValidator` (`validator.py`, `make validate`):

- **Parquet checks** — required columns, dtypes, non-null invariants for
  the three canonical artifacts.
- **JSONL checks** — parseability, message schema
  (`JSONLValidationError` details the offending line).
- **SFT checks** — role balance, non-empty assistant turns.
- Aggregate result: `reports/validation_results.json` with
  `overall_passed`.

## Red-Team PII Audit

`RedTeamPIIAuditor` (`pii_redteam.py`, `make audit`) adversarially
searches the full corpus for PII patterns the scrubber might have missed
— it is the "assume the anonymizer failed" counterpart of the pipeline's
anonymization stage.

## Probabilistic Audit

`scripts/run_probabilistic_pii_audit.py` (also `cli.py audit-prob`):
stratified sampling + Wilson CI (99%) + power analysis. This is the
release-blocking zero-PII proof — a leak rate above tolerance fails the
SLO gate. The 100k-sample CI configuration lives in
`.github/workflows/ci.yml` (`Data Validation & Audits` job).

## Artifact Manifest

`artifact_manifest.py` pins canonical paths + sha256 digests:

```bash
python -m src.validation.artifact_manifest create --root . --manifest reports/dataset_manifest.json
python -m src.validation.artifact_manifest verify --root . --manifest reports/dataset_manifest.json
```

`verify_manifest` is used by the SLO gate (fail-closed: missing or stale
manifest → HOLD). The module has hardening against path escape, symlink
and partial-write attacks — see `tests/test_artifact_manifest.py`.

## Hub Reconciliation

`hub_reconciliation.py` compares the local manifest against the pinned
Hugging Face Hub revision snapshot (`reports/pinned_hub_manifest.json`).
Statuses: `EXACT_MATCH`, `DOCUMENTED_DIVERGENCE` (acceptable, documented),
or mismatch (fails the gate).
