# Validation & Audits

`ricc/validation/` is the quality police: schema checks, zero-PII proof,
artifact integrity and Hub reconciliation. All checks feed the
[SLO gate](slo-gate.md).

## Dataset Validator

`DatasetValidator` (`validator.py`, `make validate`):

- **Parquet checks** — the **published contract** is enforced: the column
  names, their order and their types must match the declaration in
  `ricc/schema_version.py`, and the file must carry the current contract
  version and fingerprint. A mismatch fails the validation with the status
  `INVALID` and a list of the problems.
- **JSONL checks** — parseability, message schema
  (`JSONLValidationError` details the offending line).
- **SFT checks** — role balance, non-empty assistant turns.
- Aggregate result: `reports/validation_results.json` with
  `overall_passed`.

The contract itself is declared in `ricc/schema_version.py`, fingerprinted,
and recorded in `reports/dataset_schema.json`
(`python scripts/generate_schema_manifest.py`). See
[ADR 0006](../adr/0006-dataset-contract-versioning.md).

## Red-Team PII Audit

`RedTeamPIIAuditor` (`pii_redteam.py`, `make audit`) adversarially
searches the full corpus for PII patterns the scrubber might have missed
— it is the "assume the anonymizer failed" counterpart of the pipeline's
anonymization stage.

## Probabilistic Audit

`scripts/run_probabilistic_pii_audit.py` (also `cli.py audit-prob`):
stratified sampling + Wilson CI (99%) + power analysis. This is the
release-blocking zero-PII proof — a leak rate above tolerance fails the
SLO gate. The parameters live in `params.yaml` (`pii_audit:`) and are
read through `settings.pii_audit`; the CLI flags override them.

**It fails closed.** A corpus without the audited `text_clean` column is
reported as `FAILED_MISSING_COLUMNS`, never as PASSED: an audit that
cannot read the corpus must not certify it. The audit covers **12 leak
categories** — the seven the scrubber always checked, plus the secret
assignments, private invite links, credentialed DB URLs, Telegram forward
headers and raw user mentions it cleans but used to ignore.

**The sample size has a statistical reason.** With zero observed leaks the
one-sided 99% upper bound is ~3.29/sample_size, so the sample must be
large enough for that to stay under the tolerance: 50 000 gives 1.08e-04
(false `LEAKS_DETECTED` on a clean corpus), 100 000 gives 5.41e-05. The
same maths works the other way: a small corpus cannot be certified at all
(50 messages give an upper bound of 0.098).

**The sampling is a genuine probability sample.** Each stratum is drawn
with its own RNG; using one `random_state` for all of them made the
selection systematic — the same relative positions in every stratum — and
the confidence interval invalid.

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
