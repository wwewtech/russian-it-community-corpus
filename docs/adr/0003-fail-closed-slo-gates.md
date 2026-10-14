# ADR 0003: Fail-Closed SLO Release Gates

**Status:** Accepted
**Date:** 2026-10-14
**Authors:** RICC maintainers
**Supersedes:** —

## Context

RICC publishes a dataset whose headline property is **zero PII**. A
release decision (SHIP/HOLD) therefore has an asymmetric failure mode:
a false SHIP publishes leaked personal data and cannot be undone
(dumps are mirrored, cached, forked); a false HOLD merely delays a
release.

Before the SLO gate, release quality was judged by eyeballing several
JSON reports, each produced by a different job. That had two failure
modes:

1. **Missing proof read as success.** If a job was skipped or its report
   was never written, nothing failed — the release went out unaudited.
2. **Synthetic data could green-light a release.** CI builds 100-row
   synthetic fixtures for speed; a gate that only checks "does the
   report say PASS" cannot tell a synthetic smoke run from a real
   corpus.

## Decision

`src/monitoring/slo_gate.py` aggregates all quality signals into one
SHIP/HOLD verdict with explicit, asymmetric policy:

1. **Privacy signals fail closed.** A missing `probabilistic_pii_audit`
   report or a missing/stale `dataset_manifest.json` is HOLD, not PASS.
   No proof = no release.
2. **Informational signals fail open** (with a visible warning in the
   checks list): drift `moderate_drift` from sampling noise, absent
   `sft_quality_report`, absent `hub_reconciliation` report. Blocking on
   these would train maintainers to override the gate.
3. **Synthetic data can never SHIP.** CI fixture runs are stamped
   `_source.synthetic=true`; the gate downgrades SHIP to HOLD with an
   explicit `synthetic-smoke` check. A deliberate override requires
   changing the stamp, which is reviewable.
4. **The verdict is a single artifact** (`reports/slo_verdict.json`) with
   per-check `passed`/`detail`, consumed by CI, the release workflow and
   the `RiccSloGateHold` alert. No report eyeballing.
5. **Provenance is part of the gate**: canonical parquet artifacts are
   verified against the sha256 manifest (`verify_manifest`, itself
   hardened against path escape/symlink attacks) — a release cannot ship
   artifacts other than the audited ones.

## Consequences

- Release pauses are explainable: every HOLD names the failing check and
  the reason.
- Adding a new quality signal means adding a check with an explicit
  blocking/non-blocking classification — the asymmetry decision is made
  once, in code, not under release pressure.
- The drift check's "moderate is informational" policy is documented and
  tested (`tests/test_hybrid_rag_slo.py`); changing it is an ADR-level
  decision, not a threshold tweak.

## References

- Gate implementation: `src/monitoring/slo_gate.py`
- User guide: `docs/user-guide/slo-gate.md`
- Manifest hardening: `tests/test_artifact_manifest.py`
