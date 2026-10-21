# Taxonomy & Tagging

`ricc/taxonomy/` assigns every message an IT domain and technical tags.

## Domains

Nine domains, keyword lists in `params.yaml` (`domain_taxonomy:` — single
source of truth, consumed by `TechnicalTagger` via Aho-Corasick):

`ai_ml_nlp`, `backend_databases`, `frontend_ui`, `devops_infra`,
`business_legal_fintech`, `sysadmin_security`, `career_team_management`,
`general_tech_chat`.

Matching is case-insensitive with Russian keyword support; the classifier
(`ricc/taxonomy/classifier.py`) combines keyword density with heuristics
and falls back to `general_tech_chat` when confidence is low.

## Tags

`TechnicalTagger` (`ricc/tagger.py`) extracts technology tags (docker,
kubernetes, postgres, lora, react, ...) from the same keyword tables.
Tags are stored as a list column in the parquet exports and drive the
RAG knowledge-base domain filter.

## Extending

1. Edit `params.yaml` — add keywords to an existing domain or a new
   domain block (the classifier picks it up automatically).
2. `python cli.py run` re-runs tagging; or tag incrementally via the
   pipeline stage.
3. Check domain balance: `python cli.py analyze` reports per-domain
   volumes; a heavily skewed distribution is normal for a technical
   community but feeds the [drift monitor](drift-monitoring.md) baseline.

## Where It Lands

- `full_clean_messages.parquet` — `domain`, `tags` columns.
- `rag_knowledge_base.parquet` — `topic_domain`, `topic_tags` (used by
  `LocalRAGPipeline.search(domain_filter=...)`).
- Analytics reports — per-domain volumes, trends and the tech radar.
