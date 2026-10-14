# Analytics & Reports

`src/analytics/` computes the statistical and semantic profile of the
corpus and renders it into reports consumed by the CLI, the Streamlit
Studio and the docs pipeline.

## Engine

`DeepChatAnalyzer` (`engine.py`, ~700 LoC, strict-typed) computes:

- **Message metrics** — length distributions (chars/words/token estimates
  with percentiles), activity by hour/year, inter-arrival times.
- **Author metrics** — messages per author, participation concentration.
- **Social network analytics** — via `src/analytics/network.py`
  (reply-graph degree/centrality).
- **Topic clustering** — TF-IDF + clustering over tagged messages,
  surfaced as topic groups.
- **Longitudinal trends** — per-domain volume evolution.

Metric definitions shared with docs/reports live in
`src/analytics/metrics.py`; `ReportGenerator`
(`src/analytics/report_generator.py`) renders the JSON report consumed by
`cli.py analyze` and the Studio's tech-radar page.

## Outputs

- `reports/analytics_summary.json` — the full analysis payload.
- Markdown reports rendered from JSON sources (single source of truth —
  the `tests/test_metrics_consistency.py` guard fails the build if a
  published figure drifts from the manifest).

## Where to Look

- **CLI**: `make analyze` / `python cli.py analyze --limit 100000`.
- **Studio**: "Технологический радар" page renders hourly/yearly activity
  and top slang terms from the same JSON.

## Performance Notes

The analyzer samples NLP-heavy paths (`--limit`); full-corpus runs are
only needed when regenerating release reports (`make reports-real`).
Percentile helpers are pure functions unit-tested in
`tests/test_analytics_engine.py`.
