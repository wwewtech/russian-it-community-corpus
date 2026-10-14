# Streamlit Data Studio

`app.py` — интерактивная витрина корпуса и платформы. Запуск:

```bash
make ui                    # = streamlit run app.py
# Docker:
docker compose up data-studio   # http://localhost:8501
```

## Sections

| Page (sidebar) | Contents |
|---|---|
| Главная и метрики | headline corpus figures, pulled from reports |
| Просмотр сообщений | paginated browsing of `full_clean_messages.parquet` |
| Диалоговая студия SFT и DPO | dialogue inspection from SFT/DPO exports |
| База знаний RAG | live hybrid search over 325k chunks |
| Zero-PII песочница | paste text → see every masked entity in real time |
| Технологический радар | hourly/yearly activity, top slang terms |
| Доменный бенчмарк | 100 benchmark questions, filterable by domain |
| Документация | renders Dataset Card, Model Card, LoRA Model Zoo |

## Data Sources

Pure helpers live in `app_helpers.py` (unit-testable without a Streamlit
runtime; `tests/test_app*.py` smoke-tests the app via `AppTest`).

## Metrics Endpoint

The app starts the Prometheus metrics server on start-up:

- port: `RICC_METRICS_PORT` (default 9090, declared in the k8s
  Deployment/Service and scraped by the ServiceMonitor);
- safe under Streamlit reruns (`maybe_start_metrics_server`);
- Docker Compose: enable the `observability` profile for Prometheus +
  Grafana (`docker compose --profile observability up`).

See [Observability API](../api/observability.md) for the metric list and
dashboards.
