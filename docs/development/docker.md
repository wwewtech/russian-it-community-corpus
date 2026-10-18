# Docker

## Images & Services

`Dockerfile` (python:3.14-slim) + `docker-compose.yml`:

| Service | Purpose | Ports |
|---|---|---|
| `data-studio` | Streamlit app | 8501 (host-mapped via `$PORT`) |
| `pipeline-runner` | one-shot `python main.py` (profile `pipeline`) | — |
| `prometheus` | metrics + RICC alert rules (profile `observability`) | 9091→9090 |
| `grafana` | provisioned dashboard (profile `observability`) | 3000 |

## Data Studio

```bash
docker compose up data-studio          # http://localhost:8501
docker compose up pipeline-runner --profile pipeline
```

## Observability Profile

```bash
docker compose --profile observability up
# Prometheus:  http://localhost:9091 (scrapes data-studio:9090)
# Grafana:     http://localhost:3000 (dashboard "RICC", anon viewer)
```

Provisioned read-only from `deploy/monitoring/`
(`prometheus.yml`, `grafana-datasources.yaml`, `grafana-dashboards.yaml`,
`ricc-overview.json`); alert rules come from
`deploy/k8s/prometheus-rules.yaml`. Override ports/admin credentials via
`PROMETHEUS_PORT`, `GRAFANA_PORT`, `GRAFANA_ADMIN_USER/PASSWORD` env.

## Hardening

- Non-root `appuser` (10001), `read_only: true` rootfs, tmpfs for
  `/tmp`, `/var/tmp`, `.streamlit`; `no-new-privileges`.
- Healthcheck on `/_stcore/health` (mirrors the k8s probes).
- Resource limits: 4 CPU / 8G (studio), 8 CPU / 16G (pipeline).
- `HF_TOKEN` passed via env, never baked into the image (see
  `docs/adr/0001-hf-token-handling.md`).

## k8s Equivalents

`deploy/k8s/data-studio-{deployment,service}.yaml` mirror the hardening
(non-root, seccomp, probes, resource requests) and add the metrics port
(9090) + ServiceMonitor. See
[Observability API](../api/observability.md).

## Building

```bash
docker compose build          # both images
make docker-build             # alias
```

Dependency layer is cached separately (`requirements.txt` +
`pyproject.toml` copied first).
