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


def evaluate(reports_dir: Path = REPORTS_DIR) -> SloVerdict:
    checks: list[Check] = []

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
        ok = verdict in ("stable", "no_drift", "pass", "ok")
        checks.append(Check("drift", ok, f"verdict={verdict}"))

    sft = _load_json(reports_dir / "sft_quality_report.json")
    if sft is not None:
        ok = bool(sft.get("passed", sft.get("overall_passed", True)))
        checks.append(Check("sft-quality", ok, "sft_quality_report present"))

    stats = _load_json(reports_dir / "pipeline_execution_stats.json")
    if stats is not None:
        raw_n = stats.get("cleaned_messages_count", 0) or 0
        n = int(raw_n) if isinstance(raw_n, (int, str)) else 0
        checks.append(Check("pipeline-volume", n > 0, f"cleaned={n}"))

    # Provenance gate (fail-closed): artifact identity must be verified against
    # the manifest snapshot. Integrity/consistency problems flip the verdict to
    # HOLD because shipped numbers must correspond to the exact artifacts.
    try:
        from src.validation.artifact_manifest import verify_manifest

        verify_manifest(reports_dir.parent, reports_dir / "dataset_manifest.json")
        checks.append(Check("artifact-manifest", True, "dataset_manifest.json verified"))
    except Exception as exc:  # noqa: BLE001 - fail-closed on any provenance failure
        reason = str(exc) or type(exc).__name__
        checks.append(Check("artifact-manifest", False, f"manifest verify failed: {reason}"))

    verdict = "SHIP" if all(c.passed for c in checks) and checks else "HOLD"
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
