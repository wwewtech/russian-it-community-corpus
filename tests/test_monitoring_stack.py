"""Guards for the P4 monitoring stack (deploy/k8s + deploy/monitoring).

These tests fail the build if the monitoring manifests drift from the
metrics contract in src/observability/__init__.py:

* every metric referenced by the Grafana dashboard and PrometheusRules
  must exist in the observability module;
* the ServiceMonitor must reference the metrics port declared on the
  Service, and the Deployment must expose that port;
* PrometheusRules alerts must use severity labels the dashboard expects.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_K8S = REPO_ROOT / "deploy" / "k8s"
DEPLOY_MONITORING = REPO_ROOT / "deploy" / "monitoring"

_HAS_YAML = importlib.util.find_spec("yaml") is not None


def _load_yaml_docs(path: Path) -> list[dict]:
    if not _HAS_YAML:
        pytest.skip("pyyaml not installed")
    import yaml

    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
    assert docs, f"{path.name} contains no YAML documents"
    return docs


# ---------------------------------------------------------------------------
# Grafana dashboard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard() -> dict:
    path = DEPLOY_MONITORING / "ricc-overview.json"
    assert path.exists(), "deploy/monitoring/ricc-overview.json is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def test_dashboard_structure(dashboard: dict) -> None:
    assert dashboard.get("title") == "RICC Data Platform Overview"
    assert dashboard.get("uid") == "ricc-overview"
    panels = dashboard.get("panels", [])
    assert len(panels) >= 10, "dashboard lost panels"
    types = {p["type"] for p in panels}
    assert {"row", "timeseries", "stat"} <= types


def test_dashboard_promql_uses_real_metric_names(dashboard: dict) -> None:
    """Every ricc_* metric in the dashboard PromQL must exist in observability."""
    import src.observability  # noqa: F401  (registers the ricc_* metrics)
    from src.observability import (
        DATASET_MESSAGES_TOTAL,
        DATASET_SIZE_BYTES,
        DRIFT_JS_VALUE,
        DRIFT_PSI_VALUE,
        DRIFT_VERDICT,
        DRIFT_VOCAB_JACCARD,
        ERRORS_TOTAL,
        LORA_TRAIN_DURATION,
        LORA_TRAIN_LOSS,
        LORA_TRAIN_STEPS,
        LORA_VRAM_USAGE,
        PII_AUDIT_DURATION,
        PII_AUDIT_LEAKS_FOUND,
        PII_AUDIT_VERDICT,
        PIPELINE_MESSAGES_PROCESSED,
        PIPELINE_STAGE_DURATION,
        PIPELINE_STAGE_TOTAL,
        SLO_CHECK_RESULT,
        SLO_GATE_VERDICT,
        VALIDATION_RESULT,
    )

    # Gather the fully-qualified sample names each collector can emit
    # (counters -> *_total, histograms -> *_bucket/_count/_sum).
    metric_re = re.compile(r"^([a-z_]+?)(?:_(?:bucket|count|sum|created|total))?$")

    def _family(collector: object) -> set[str]:
        names: set[str] = set()
        for metric in getattr(collector, "describe", lambda: [])():
            names.add(metric.name)
        return names

    emitted: set[str] = set()
    for collector in (
        PIPELINE_STAGE_TOTAL,
        PIPELINE_STAGE_DURATION,
        PIPELINE_MESSAGES_PROCESSED,
        PII_AUDIT_LEAKS_FOUND,
        PII_AUDIT_VERDICT,
        PII_AUDIT_DURATION,
        DRIFT_PSI_VALUE,
        DRIFT_JS_VALUE,
        DRIFT_VOCAB_JACCARD,
        DRIFT_VERDICT,
        LORA_TRAIN_DURATION,
        LORA_TRAIN_STEPS,
        LORA_TRAIN_LOSS,
        LORA_VRAM_USAGE,
        DATASET_MESSAGES_TOTAL,
        DATASET_SIZE_BYTES,
        SLO_GATE_VERDICT,
        SLO_CHECK_RESULT,
        VALIDATION_RESULT,
        ERRORS_TOTAL,
    ):
        emitted.update(_family(collector))
        for sample in collector.collect()[0].samples:
            emitted.add(sample.name)

    panels = dashboard.get("panels", [])
    exprs: list[str] = []
    for panel in panels:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            if expr:
                exprs.append(expr)
    assert exprs, "dashboard has no PromQL targets"

    referenced: set[str] = set()
    base_re = re.compile(r"\b(ricc_[a-z_]+)\b")
    for expr in exprs:
        referenced.update(base_re.findall(expr))
    assert referenced, "dashboard PromQL references no ricc_* metrics"

    def _known(metric: str) -> bool:
        if metric in emitted:
            return True
        match = metric_re.match(metric)
        return bool(match and match.group(1) in emitted)

    unknown = {m for m in referenced if not _known(m)}
    assert not unknown, f"dashboard references metrics not in src.observability: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# ServiceMonitor
# ---------------------------------------------------------------------------


def test_servicemonitor_targets_metrics_port() -> None:
    docs = _load_yaml_docs(DEPLOY_K8S / "data-studio-servicemonitor.yaml")
    sm = docs[0]
    assert sm["apiVersion"] == "monitoring.coreos.com/v1"
    assert sm["kind"] == "ServiceMonitor"
    endpoint = sm["spec"]["endpoints"][0]
    assert endpoint["port"] == "metrics", "ServiceMonitor must scrape the named 'metrics' port"
    assert endpoint["path"] == "/metrics"


def test_service_declares_metrics_port() -> None:
    docs = _load_yaml_docs(DEPLOY_K8S / "data-studio-service.yaml")
    service = docs[0]
    ports = {p["name"]: p for p in service["spec"]["ports"]}
    assert "http" in ports
    assert "metrics" in ports, "Service must declare the named 'metrics' port for the ServiceMonitor"
    assert ports["metrics"]["port"] == 9090


def test_deployment_exposes_metrics_port() -> None:
    docs = _load_yaml_docs(DEPLOY_K8S / "data-studio-deployment.yaml")
    dep = docs[0]
    container = dep["spec"]["template"]["spec"]["containers"][0]
    ports = {p["name"]: p["containerPort"] for p in container.get("ports", [])}
    assert ports.get("metrics") == 9090, "Deployment must expose containerPort 9090 named 'metrics'"
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    assert env.get("RICC_METRICS_PORT") == "9090"


# ---------------------------------------------------------------------------
# PrometheusRules
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rules_docs() -> list[dict]:
    return _load_yaml_docs(DEPLOY_K8S / "prometheus-rules.yaml")


def test_prometheusrule_structure(rules_docs: list[dict]) -> None:
    rule = rules_docs[0]
    assert rule["apiVersion"] == "monitoring.coreos.com/v1"
    assert rule["kind"] == "PrometheusRule"
    groups = rule["spec"]["groups"]
    assert len(groups) >= 3, "expected pipeline/pii/quality/errors groups"
    for group in groups:
        assert group["rules"], f"group {group['name']} has no rules"
        for alert in group["rules"]:
            assert "alert" in alert, f"rule in {group['name']} missing alert name"
            assert "expr" in alert
            assert alert.get("labels", {}).get("severity") in ("critical", "warning", "info"), (
                f"alert {alert['alert']} has unexpected severity"
            )


def test_pii_alert_is_critical(rules_docs: list[dict]) -> None:
    """Zero-PII is the core SLO: leak alerts must page (severity: critical)."""
    rule = rules_docs[0]
    pii_alerts = [a for g in rule["spec"]["groups"] if g["name"] == "ricc.pii" for a in g["rules"]]
    assert pii_alerts, "ricc.pii group is empty"
    for alert in pii_alerts:
        assert alert["labels"]["severity"] == "critical", f"{alert['alert']} must be critical"


def test_rules_use_real_metric_names(rules_docs: list[dict]) -> None:
    rule = rules_docs[0]
    metric_re = re.compile(r"\b(ricc_[a-z_]+)\b")
    known_bases = {
        "ricc_pipeline_stage_total",
        "ricc_pii_audit_leaks_found_total",
        "ricc_pii_audit_verdict_total",
        "ricc_drift_verdict_total",
        "ricc_slo_gate_verdict_total",
        "ricc_validation_result_total",
        "ricc_errors_total",
    }
    referenced: set[str] = set()
    for group in rule["spec"]["groups"]:
        for alert in group["rules"]:
            referenced.update(metric_re.findall(str(alert["expr"])))
    unknown = referenced - known_bases
    assert not unknown, f"PrometheusRules reference unknown metrics: {sorted(unknown)}"
