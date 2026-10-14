# Studio Features

## Zero-PII Sandbox

Paste any text (a demo PII-rich prompt is prefilled) and run the
two-layer scrubber (Regex + Natasha NER + morphological name forms) with
one click. The output shows:

- cleaned text side-by-side with the input;
- a JSON breakdown of every masked entity category with counts.

Use it to validate new PII patterns before adding them to the regex
layer ([PII Anonymization](../user-guide/pii-anonymization.md)).

## RAG Knowledge Base Search

Full `LocalRAGPipeline` surface: hybrid/lexical/tfidf modes, domain
filter, top-k. Scores and chunk metadata (title, date range) are shown
per hit — the same pipeline the chat CLI uses.

## Dialogue Studio (SFT / DPO)

Browse the exported SFT dialogues and DPO pairs directly from
`dataset_output/`, with per-dialogue role/turn inspection. Useful for
eyeballing [extraction quality](../user-guide/graph-extraction.md)
before training.

## Tech Radar

Renders `reports/analytics_summary.json`: hourly activity, yearly volume
dynamics (2018–2026) and top community slang terms — the community-facing
view of the [analytics engine](../user-guide/analytics-reports.md).

## Domain Benchmark

The 100 benchmark questions with domain filtering — the question bank
used by the academic benchmark harnesses.

## Reports Viewer

In-app rendering of the published documents: Dataset Card, HF Model
Card, LoRA Model Zoo catalog. Content is loaded from `reports/` and kept
consistent with the manifest by the metrics-consistency test guard.
