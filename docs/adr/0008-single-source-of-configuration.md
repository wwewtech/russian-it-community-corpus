# ADR 0008: One live configuration object, no frozen snapshots

- **Status**: Accepted
- **Date**: 2026-10-22
- **Deciders**: maintainer
- **Relates to**: `ricc/settings.py`, `docs/architecture/overview.md`

## Context

The project had **two** configuration sources, and the second one was a
trap.

`ricc/settings.py` held the real configuration: a `pydantic-settings`
`Settings` instance loaded from the environment (the `RICC__` prefix) and
from `params.yaml`. Every field is derived from a base field through a
read-only property (`parquet_output_dir` → `PARQUET_OUTPUT_DIR`).

`ricc/config.py` re-exported the same values **as a snapshot taken at
import time**:

```python
OUTPUT_DIR = settings.OUTPUT_DIR
MINHASH_NUM_PERM = settings.MINHASH_NUM_PERM
```

`ricc/pipeline.py` carried six more of the same constants, with a
comment that said it plainly: *"Backward compatibility for tests that
patch these module-level constants"*. The tests were dictating the
architecture: the production code held a second, frozen copy of the
configuration so that the tests could patch it.

The consequence was measured, not theoretical:

```
config.OUTPUT_DIR (snapshot)          : ...\dataset_output
Settings(output_dir=...).OUTPUT_DIR   : \tmp\custom-output
```

A consumer that built its own `Settings` — the documented way to
configure the library — got a **different configuration** than every
module that had imported the shim. An environment variable set after
the import was ignored by the shim and honoured by the live object.

Two copies of the same configuration is also a drift hazard: the earlier
history of this project had ~520 duplicated lines (the path constants,
the MinHash tuning, the sentiment lexicon, the stopwords and the full
8-domain taxonomy) between `config.py` and `settings.py`, at 79 %
textual similarity. The shim was the last remnant of that.

## Decision

**There is one configuration object, and it is read live.**

1. `ricc/config.py` is **removed**. The compatibility argument for it
   died with the `src.` → `ricc.` rename: every consumer is in this
   repository.
2. `ricc/pipeline.py` no longer holds module-level copies. The
   exporters, the scripts and the CLI read `settings.<FIELD>` at the
   point of use.
3. `ricc/monitoring/slo_gate.py` takes the reports directory as an
   argument that defaults to the live `settings.REPORTS_DIR`.
4. The tests patch the **underlying pydantic fields**
   (`ricc.settings.settings.output_dir`), not the derived properties
   (which are read-only) and not module constants.

`ricc/settings.py` stays the single source; `params.yaml` is the
runtime source of the nested sections and the domain taxonomy.

## Consequences

- A consumer's own `Settings` instance is now honoured everywhere.
- There is one place to change a default, and one place to read it.
- The `ricc/config.py` import path is gone. It was an internal module of
  a library that was never published under that name (the wheel was
  broken until 2026-10-21), so there is no public API break.
- The tests no longer need the production code to carry a second copy
  of the configuration. A test asserts that no module holds a
  module-level copy of a settings value, so the pattern cannot come
  back.

## Alternatives rejected

- **Keep the shim, but make it lazy.** A module-level `__getattr__`
  would forward to `settings`, so the snapshot problem would go. But a
  module whose whole purpose is to forward names adds an indirection
  and a second import path for the same values; removing it is simpler
  and the compatibility cost is zero.
- **Make `Settings` mutable and patch it in tests.** Pydantic models
  are validated on assignment, and a mutable global configuration is
  exactly what caused the original drift. The read-only properties over
  base fields give the tests a clean seam without that.
