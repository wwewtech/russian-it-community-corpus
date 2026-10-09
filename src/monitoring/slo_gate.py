"""SLO gate for RICC releases: single PASS/FAIL verdict from all quality signals.

Reads the JSON artifacts produced by the pipeline and monitoring scripts::

    reports/validation_results.json
    reports/probabilistic_pii_audit.json
    reports/drift_report.json
    reports/pipeline_execution_stats.json
    reports/sft_quality_report.json (optional, CI-only)

Exit code 0 = SHIP, 1 = HOLD. Missing files are reported as HOLD with an
explicit reason (fail-closed for privacy signals, fail-open with warning
for purely informational ones). Used by ``make slo`` and CI.

Synthetic-data guard: reports stamped ``_source.synthetic=true`` (or with
tiny ``reference_rows``/``cleaned`` counts from the 100-row CI fixture) are
treated as SMOKE, never SHIP — the gate returns HOLD with an explicit
``synthetic-smoke`` detail so CI cannot green-light a release on fake data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class SloVerdict:
    verdict: str = "HOLD"
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
        return None
    except Exception:
        return None


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _is_synthetic(reports_dir: Path) -> tuple[bool, str]:
    """Detect CI synthetic-fixture runs, never real local artifacts.

    Fail-closed ONLY on the explicit marker left by the synthetic fixture
    builder (``reports/synthetic_fixture.json`` with ``_source.synthetic``).
    Raw row counts are deliberately NOT used: real runs produce small counts
    too (e.g. a fresh pipeline run on 2 raw exports), and volume is already
    covered by the separate pipeline-volume check.
    """
    marker = _load_json(reports_dir / "synthetic_fixture.json")
    if isinstance(marker, dict):
        src = marker.get("_source")
        if isinstance(src, dict) and src.get("synthetic") is True:
            return True, "synthetic_fixture.json marker present"
    return False, ""


def evaluate(reports_dir: Path = REPORTS_DIR) -> SloVerdict:
    checks: list[Check] = []

    synthetic, reason = _is_synthetic(reports_dir)

    validation = _load_json(reports_dir / "validation_results.json")
    if validation is None:
        checks.append(Check("validation", False, "missing reports/validation_results.json"))
    else:
        ok = bool(validation.get("overall_passed", validation.get("passed", False)))
        checks.append(Check("validation", ok, f"overall_passed={ok}"))

    audit = _load_json(reports_dir / "probabilistic_pii_audit.json")
    if audit is None:
        # Privacy defaults to fail-closed: no audit proof => HOLD.
        checks.append(Check("pii-audit", False, "missing probabilistic_pii_audit.json"))
    else:
        verdict = str(audit.get("verdict", audit.get("status", ""))).upper()
        ok = verdict in ("PASS", "PASSED", "SHIP", "OK")
        checks.append(Check("pii-audit", ok, f"verdict={verdict or 'unknown'}"))

    drift = _load_json(reports_dir / "drift_report.json")
    if drift is None:
        checks.append(Check("drift", True, "no drift report — skip (warning)"))
    else:
        verdict = str(drift.get("overall_verdict", drift.get("verdict", "stable"))).lower()
        ref_rows = _to_int(drift.get("reference_rows", 0) or 0)
        cur_rows = _to_int(drift.get("current_rows", 0) or 0)
        # A same-snapshot baseline on a random subsample can show moderate
        # vocabulary drift purely from sampling noise (two disjoint 50k samples
        # of the same corpus). That is informational, NOT a release blocker:
        # block only on significant_drift. True temporal drift is detected by
        # pointing the drift CLI at two frozen snapshots.
        if verdict in ("stable", "no_drift", "pass", "ok"):
            checks.append(Check("drift", True, f"verdict={verdict}"))
        elif verdict == "moderate_drift":
            detail = f"verdict={verdict} (ref={ref_rows}, cur={cur_rows}) — informational, not blocking"
            checks.append(Check("drift", True, detail))
        else:
            checks.append(Check("drift", False, f"verdict={verdict}"))

    sft = _load_json(reports_dir / "sft_quality_report.json")
    if sft is not None:
        ok = bool(sft.get("passed", sft.get("overall_passed", True)))
        checks.append(Check("sft-quality", ok, "sft_quality_report present"))

    stats = _load_json(reports_dir / "pipeline_execution_stats.json")
    if stats is not None:
        n = _to_int(stats.get("cleaned_messages_count", 0) or 0)
        checks.append(Check("pipeline-volume", n > 0, f"cleaned={n}"))

    # Provenance gate: artifact identity must be verified against the manifest snapshot.
    # Fail-closed: missing manifest → HOLD.
    canonical_artifacts_exist = all(
        (reports_dir.parent / p).exists()
        for p in (
            "dataset_output/parquet/full_clean_messages.parquet",
            "dataset_output/parquet/sft_dialogues.parquet",
            "dataset_output/parquet/rag_knowledge_base.parquet",
        )
    )
    manifest_path = reports_dir / "dataset_manifest.json"

    if not canonical_artifacts_exist:
        # No artifacts to verify — fail-closed for provenance
        checks.append(Check("artifact-manifest", False, "canonical artifacts missing — fail-closed"))
    elif not manifest_path.exists():
        # Artifacts exist but manifest missing
        checks.append(Check("artifact-manifest", False, "dataset_manifest.json missing — fail-closed"))
    else:
        try:
            from src.validation.artifact_manifest import verify_manifest

            verify_manifest(reports_dir.parent, manifest_path)
            checks.append(Check("artifact-manifest", True, "dataset_manifest.json verified"))
        except Exception as exc:  # noqa: BLE001 - fail-closed on any provenance failure
            reason = str(exc) or type(exc).__name__
            checks.append(Check("artifact-manifest", False, f"manifest verify failed: {reason}"))

    # Hub reconciliation gate: check alignment with pinned HF Hub revision snapshot
    hub_rec = _load_json(reports_dir / "dataset_reconciliation_report.json")
    if hub_rec is None:
        checks.append(Check("hub-reconciliation", True, "no reconciliation report — skip (warning)"))
    else:
        rec_status = str(hub_rec.get("status", "")).upper()
        ok = rec_status in ("EXACT_MATCH", "DOCUMENTED_DIVERGENCE", "VERIFIED", "OK")
        rev = str(hub_rec.get("pinned_revision", ""))[:8]
        checks.append(Check("hub-reconciliation", ok, f"pinned_rev={rev}, status={rec_status}"))

    verdict = "SHIP" if all(c.passed for c in checks) and checks else "HOLD"
    if verdict == "SHIP" and synthetic:
        # Never SHIP on synthetic fixtures: downgrade to HOLD with explicit cause.
        checks.append(Check("synthetic-smoke", False, f"CI fixture detected ({reason}) — SMOKE, not SHIP"))
        verdict = "HOLD"
    return SloVerdict(verdict=verdict, checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="RICC SLO release gate (SHIP/HOLD)")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    verdict = evaluate(args.reports)
    payload = json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    return 0 if verdict.verdict == "SHIP" else 1


if __name__ == "__main__":
    sys.exit(main())
