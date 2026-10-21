# Quality Gates

The platform ships the mechanisms that make its claims checkable. Every
one of them **fails closed**: a gate that cannot verify a thing reports a
failure, not a pass.

## The gates

| Gate | What it proves | Where | Fails closed on |
|---|---|---|---|
| **Probabilistic PII audit** | The 99 % upper bound on the message-level leak rate stays below the tolerance, over 12 categories | `ricc/validation/probabilistic_audit.py` | a corpus it cannot read; a category it does not cover |
| **Training policy gate** | A published adapter covered a full epoch of the corpus and was not time-boxed | `ricc/lora/policy_gate.py` | missing provenance; a partial epoch |
| **SLO gate** | Validation, PII audit, drift, SFT quality and artifact provenance all pass before a release | `ricc/monitoring/slo_gate.py` | a missing report; a synthetic fixture |
| **Artifact manifest** | The published Parquet files are byte-identical to the snapshot that was audited | `ricc/validation/artifact_manifest.py` | a hash mismatch |
| **Dataset contract** | The column names, order and types match the declared contract, and the version metadata is current | `ricc/schema_version.py` | a renamed, reordered or retyped column |
| **SFT quality monitor** | The real corpus (not a fixture) meets the dialogue-quality floors | `ricc/monitoring/sft_quality.py` | a corpus it could not read |
| **DVC lock** | The pipeline can be reproduced from the pinned raw exports | `dvc.lock`, `dvc.yaml` | a stage whose inputs changed |

## Why they fail closed

A gate that passes when it cannot check anything is worse than no gate:
it converts an unknown into a green light. The concrete cases this
project hit:

- The PII audit reported **PASSED** on a corpus whose `text_clean` column
  did not exist — every row became an empty string, so zero leaks were
  "found" and the report carried a 99 % confidence statement about a
  corpus it had never read.
- The `sft_quality` monitor reported a hard **FAIL** on the real corpus
  (it parsed zero of 171 520 dialogues, because the parquet stores
  `list<struct>` and the monitor only understood lists and JSON strings)
  while the synthetic CI fixture **passed** — the gate was validating
  the fixture.
- The `artifact_manifest` verification in CI compared a manifest that had
  just been created from the same files, so it could never fail.

All three now refuse to certify what they cannot verify.

## Running them

```bash
# the privacy proof
python scripts/run_probabilistic_pii_audit.py

# the training policy, for the whole zoo
python -c "from pathlib import Path; from ricc.lora.policy_gate import evaluate_compliance; \
[print(p.name, evaluate_compliance(p).compliant) for p in sorted(Path('lora_adapters').iterdir()) if p.is_dir()]"

# the dataset contract
python scripts/generate_schema_manifest.py --check

# the release verdict
python -m ricc.monitoring.slo_gate --json-out reports/slo_verdict.json
```

## In CI

| Job | Gate |
|---|---|
| `test` | the unit and integration suite, with the coverage floor |
| `coverage-regression` | the changed lines must be covered (diff-cover) |
| `validate` | the dataset validation, the PII audit and the reconciliation |
| `dvc-consistency` | the raw-export dependency list, the stage definitions and the lock file |
| `slo-gate` / `slo-gate-real` | the release verdict, on synthetic and on the real corpus |
| `security` | bandit, detect-secrets and pip-audit |

## The statistical honesty of the benchmarks

The evaluation suites compare variants that run on the **same tasks**, so
the comparison is paired and McNemar's exact test applies. On the
published raw outputs **no comparison is significant** (p ≥ 0.125), and
the report says so instead of claiming an improvement. The benchmark
suites also refuse to run without the adapter rather than writing the
base results into the LoRA column — that fallback is what produced the
previously retracted numbers.

See [ADR 0003](../adr/0003-fail-closed-slo-gates.md),
[ADR 0005](../adr/0005-training-policy-gate.md) and
[ADR 0006](../adr/0006-dataset-contract-versioning.md).
