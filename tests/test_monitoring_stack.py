"""Guards for the P4 monitoring stack (deploy/k8s + deploy/monitoring).

These tests fail the build if the monitoring manifests drift from the
metrics contract in ricc/observability/__init__.py:

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

from ricc.observability import (
    record_classified_ml_error,
    record_lora_result,
)

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
    import ricc.observability  # noqa: F401  (registers the ricc_* metrics)
    from ricc.observability import (
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
    assert not unknown, f"dashboard references metrics not in ricc.observability: {sorted(unknown)}"


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
        "ricc_lora_train_total",
    }
    referenced: set[str] = set()
    for group in rule["spec"]["groups"]:
        for alert in group["rules"]:
            referenced.update(metric_re.findall(str(alert["expr"])))
    unknown = referenced - known_bases
    assert not unknown, f"PrometheusRules reference unknown metrics: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# Wire-in: helpers + recorders must exist and behave (P3 taxonomy -> metrics)
# ---------------------------------------------------------------------------


def test_record_classified_ml_error_maps_oom_to_gpu_component() -> None:
    import ricc.observability as obs
    from ricc.errors import GPUOutOfMemoryError

    typed = obs.record_classified_ml_error(RuntimeError("CUDA out of memory. Tried to allocate 2GB"), model_name="Qwen")
    assert isinstance(typed, GPUOutOfMemoryError)
    assert typed.context.get("model_name") == "Qwen"


def test_record_classified_ml_error_fallback_is_training() -> None:
    from ricc.errors import TrainingError

    typed = record_classified_ml_error(ValueError("dataloader shape mismatch"), adapter_id="qwen05")
    assert isinstance(typed, TrainingError)


def test_record_lora_result_success_sets_gauges() -> None:
    from ricc.observability import LORA_TRAIN_LOSS, LORA_VRAM_USAGE

    record_lora_result(
        model="test-model",
        adapter="test-adapter",
        status="success",
        duration_sec=10.0,
        loss=0.42,
        steps=5,
        vram_bytes=1024**3,
    )
    assert LORA_TRAIN_LOSS.labels(model="test-model", adapter="test-adapter")._value.get() == pytest.approx(0.42)
    assert LORA_VRAM_USAGE.labels(model="test-model", adapter="test-adapter")._value.get() == pytest.approx(1024**3)


def test_record_lora_result_failure_does_not_clobber_gauges() -> None:
    from ricc.observability import LORA_TRAIN_LOSS

    LORA_TRAIN_LOSS.labels(model="test-model", adapter="test-adapter").set(0.11)
    record_lora_result(model="test-model", adapter="test-adapter", status="failure")
    assert LORA_TRAIN_LOSS.labels(model="test-model", adapter="test-adapter")._value.get() == pytest.approx(0.11)


def test_record_drift_metrics_sets_gauges_and_verdicts() -> None:
    from ricc.monitoring.drift import record_drift_metrics
    from ricc.observability import DRIFT_JS_VALUE, DRIFT_PSI_VALUE, DRIFT_VOCAB_JACCARD

    report = {
        "overall_verdict": "stable",
        "metrics": {
            "length_psi": {"value": 0.05, "verdict": "stable"},
            "domain_distribution": {"js_divergence_bits": 0.02, "verdict": "stable"},
            "vocabulary": {"jaccard_overlap": 0.97, "verdict": "stable"},
        },
    }
    record_drift_metrics(report)
    assert DRIFT_PSI_VALUE._value.get() == pytest.approx(0.05)
    assert DRIFT_JS_VALUE._value.get() == pytest.approx(0.02)
    assert DRIFT_VOCAB_JACCARD._value.get() == pytest.approx(0.97)


def test_record_drift_metrics_safe_on_partial_report() -> None:
    from ricc.monitoring.drift import record_drift_metrics

    # Missing keys / wrong shapes must not raise.
    record_drift_metrics({})
    record_drift_metrics({"metrics": "not-a-dict"})
    record_drift_metrics({"metrics": {"length_psi": "garbage"}})


# ---------------------------------------------------------------------------
# Compose observability profile
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_observability_profile_services(compose: dict) -> None:
    for name in ("prometheus", "grafana"):
        svc = compose.get("services", {}).get(name)
        assert svc, f"docker-compose.yml is missing the {name} service"
        assert "observability" in svc.get("profiles", []), f"{name} must belong to the observability profile"


def test_compose_prometheus_mounts_rules_and_scrapes_app(compose: dict) -> None:
    svc = compose["services"]["prometheus"]
    mounts = " ".join(svc.get("volumes", []))
    assert "prometheus.yml" in mounts
    assert "prometheus-rules.yaml" in mounts, "prometheus must load the RICC alert rules"
    commands = " ".join(str(c) for c in svc.get("command", []))
    assert "/etc/prometheus/prometheus.yml" in commands, "config.file must point at the provisioned config"


def test_compose_grafana_provisions_dashboard_and_datasource(compose: dict) -> None:
    svc = compose["services"]["grafana"]
    mounts = " ".join(svc.get("volumes", []))
    assert "ricc-overview.json" in mounts, "grafana must mount the RICC dashboard"
    assert "grafana-datasources.yaml" in mounts, "grafana must provision the Prometheus datasource"
    assert "grafana-dashboards.yaml" in mounts, "grafana must provision the dashboard provider"


def test_compose_no_shadowed_mount_targets(compose: dict) -> None:
    """No service may mount two volumes/tmpfs on the same target path.

    A tmpfs on /app/.hf_cache used to shadow the named hf-cache volume,
    silently discarding the persisted model cache on every restart.
    """
    for name, svc in compose.get("services", {}).items():
        targets: list[str] = []
        for vol in svc.get("volumes", []):
            if isinstance(vol, str):
                parts = vol.split(":")
                targets.append(parts[1] if len(parts) > 1 else parts[0])
            elif isinstance(vol, dict):
                targets.append(str(vol.get("target", "")))
        for entry in svc.get("tmpfs", []):
            targets.append(entry if isinstance(entry, str) else str(entry.get("target", "")))
        dupes = {t for t in targets if targets.count(t) > 1}
        assert not dupes, f"{name} mounts {sorted(dupes)} more than once (shadowing)"


def test_compose_prometheus_config_targets_data_studio() -> None:
    import yaml

    cfg = yaml.safe_load((DEPLOY_MONITORING / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {j["job_name"]: j for j in cfg.get("scrape_configs", [])}
    assert "ricc-data-studio" in jobs, "prometheus.yml must scrape the data-studio app"
    targets = jobs["ricc-data-studio"]["static_configs"][0]["targets"]
    assert "data-studio:9090" in targets, "app target must be the compose-network data-studio:9090"
    rule_files = cfg.get("rule_files", [])
    assert any("ricc-rules.yaml" in rf for rf in rule_files), "prometheus.yml must include the RICC alert rules"
