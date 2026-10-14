# PII Anonymization

RICC's core guarantee is **Zero-PII**: no phone numbers, emails, crypto
wallets, API keys or real names survive into the published dataset.

## Two-Layer Scrubbing

`UnifiedPIIAnonymizer` (`src/pii/anonymizer.py`) chains:

1. **Regex layer** (`src/pii/regex_scrubber.py`) — deterministic masks for
   phone numbers (RU/international formats), emails, crypto wallets
   (TRC20/ERC20/BTC), OpenAI-style API keys.
2. **NER layer** (`src/pii/ner_scrubber.py`) — Natasha/Slovnet person-name
   detection with morphological declension handling
   (`src/pii/deep_anonymizer.py` extends it with name-form dictionaries).

Layer 2 runs after layer 1, so the NER model never sees literal
contact strings, only prose.

## Pseudonymization

Authors are replaced with stable anonymous IDs
(`author_anon` / `author_id_anon`); the mapping never leaves the pipeline.
Community node names are treated as quasi-identifiers and normalized.

## Audits

| Command | What it does |
|---|---|
| `make audit` | red-team adversarial audit (`RedTeamPIIAuditor`) on the full corpus |
| `make audit-prob` | stratified probabilistic audit: sample + Wilson CI (99%) + power analysis |
| `cli.py audit-prob --sample-size 50000 --confidence 0.99` | parameterized run |

The probabilistic audit is the release-blocking proof: a leak rate above
`max_leak_tolerance` (default 1e-4) fails the [SLO gate](slo-gate.md).

## Metrics & Reports

- Findings per category → `ricc_pii_audit_leaks_found_total{category}`.
- Verdict → `ricc_pii_audit_verdict_total{verdict=PASSED|LEAKS_DETECTED}`.
- Reports: `reports/pii_validation_report.json`,
  `reports/probabilistic_pii_audit.json`.
- Any `LEAKS_DETECTED` fires a **critical** alert
  (`RiccPiiLeaksDetected`) in the monitoring stack.

## Interactive Sandbox

The Streamlit Studio has a live scrubber page (Zero-PII песочница) where
you can paste text and see every masked entity — useful for testing the
scrubber against new PII formats before adding them to the regex layer.

## Adding a New Pattern

1. Add the regex to `src/pii/regex_scrubber.py` (with a unit test).
2. Add a red-team case to `src/validation/pii_redteam.py`.
3. Re-run `make audit` and `make audit-prob`.
