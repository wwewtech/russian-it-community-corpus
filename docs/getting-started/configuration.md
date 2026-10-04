# Configuration

All configuration is centralized in `src/settings.py` using `pydantic-settings.BaseSettings`.

## Sources (Priority Order)

1. **Environment variables** — Highest priority, prefix `RICC__`
2. **`.env` file** — Local overrides (not committed)
3. **`params.yaml`** — DVC pipeline parameters
4. **Code defaults** — Fallback values

## Environment Variables

```bash
# Core
RICC__BASE_DIR=/path/to/project
RICC__DATA_DIR=/path/to/data
RICC__OUTPUT_DIR=/path/to/output

# MinHash LSH
RICC__MINHASH_NUM_PERM=256
RICC__MINHASH_THRESHOLD=0.85
RICC__MINHASH_SHINGLE_SIZE=4

# Dialogue Reconstruction
RICC__MAX_REPLY_TIME_GAP_HOURS=24
RICC__MIN_QUESTION_WORDS=3
RICC__MIN_ANSWER_WORDS=5
RICC__MAX_THREAD_DEPTH=20
RICC__MAX_SFT_TURNS=16

# PII Audit
RICC__PII_AUDIT__SAMPLE_SIZE=100000
RICC__PII_AUDIT__CONFIDENCE=0.995
RICC__PII_AUDIT__MAX_LEAK_TOLERANCE=1e-5

# Drift Monitoring
RICC__DRIFT__PSI_STABLE=0.05
RICC__DRIFT__PSI_MODERATE=0.15
RICC__DRIFT__JS_STABLE=0.03
RICC__DRIFT__JS_MODERATE=0.10
```

## Configuration Class

```python
from ricc import Settings, settings

# Global instance (recommended)
print(settings.PARQUET_OUTPUT_DIR)

# Custom instance
custom = Settings(
    minhash_num_perm=256,
    minhash_threshold=0.85,
    pii_audit=Settings.PIIAuditSettings(
        sample_size=100000,
        confidence=0.995,
    ),
)
```

## DVC Parameters (`params.yaml`)

```yaml
pii_audit:
  sample_size: 50000
  confidence: 0.99
  max_leak_tolerance: 0.0001

drift:
  psi_stable: 0.10
  psi_moderate: 0.25
  js_stable: 0.05
  js_moderate: 0.20
  vocab_jaccard_stable: 0.90
  vocab_jaccard_moderate: 0.75

domain_taxonomy:
  ai_ml_nlp:
    title: "AI, Machine Learning & NLP"
    keywords: ["llm", "gpt", "transformer", ...]
  backend_databases:
    title: "Backend Development & Databases"
    keywords: ["python", "postgresql", "redis", ...]
  # ...
```

## Programmatic Access

```python
from ricc import settings, get_settings

# Global instance
print(settings.BASE_DIR)
print(settings.DOMAIN_TAXONOMY)


# Dependency injection friendly
def my_func(settings: Settings = get_settings()):
    print(settings.MINHASH_NUM_PERM)
```