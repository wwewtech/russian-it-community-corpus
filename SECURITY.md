# Security & Privacy Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.x     | :white_check_mark: |
| < 4.0   | :x:                |

---

## Reporting Security Vulnerabilities & PII Leaks

The **Russian IT Community Corpus** project is committed to ensuring zero personally identifiable information (Zero-PII) in all public datasets and maintaining secure data processing pipelines.

### 1. PII Incident / Leak Reporting
If you discover any unmasked personal identifier (real full name, unredacted phone number, personal email, private API token, or cryptocurrency private key) in any published dataset artifact:

1. **Do not create a public GitHub issue with the sensitive data.**
2. Send an email with details (file name, row index / `msg_id`, and description of the leak) to the maintainers or open a [PII Takedown Request](.github/ISSUE_TEMPLATE/pii_takedown_request.md).
3. **SLA**: Our team will acknowledge receipt within **24 hours** and remove the flagged record from the dataset within **48 hours**.

### 2. Code Vulnerabilities
If you identify a security vulnerability in the pipeline code (e.g. arbitrary code execution, path traversal in export loaders, or dependency vulnerabilities):

1. Report the vulnerability via GitHub Security Advisories or maintainer contact.
2. Please provide a minimal reproducible proof-of-concept (PoC).
3. We will triage and release a security patch promptly.

---

## PII Anonymization & Validation Policy

All dataset releases undergo automated heuristic pattern checks and NER regression audits before publication. The automated sanity report is available in [`reports/pii_validation_report.json`](reports/pii_validation_report.json).
