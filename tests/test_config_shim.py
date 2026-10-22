"""Guard: there must be exactly ONE source of configuration truth.

History: ``ricc/config.py`` and ``ricc/settings.py`` each carried their
own copy of the path constants, the MinHash/graph tuning, the sentiment
lexicon, the stopwords and the full 8-domain taxonomy (~520 duplicated
lines, 79 % textual similarity). The drift between the copies was the
standing hazard: the library read ``settings`` while the CLI read
``config``.

``ricc/config.py`` survived as a "thin shim", but it re-exported the
values AS A SNAPSHOT TAKEN AT IMPORT TIME — so a consumer that built its
own ``Settings`` instance (the documented way to configure the library)
got a different configuration than every module that had imported the
shim, and an environment variable set after the import was ignored. The
shim has been removed; these tests now prevent a second source of truth
from coming back.

What is enforced here:

1. There is no ``ricc/config.py`` — the frozen snapshot is gone.
2. The hardcoded 8-domain fallback lives only in ``ricc/settings.py``.
3. ``params.yaml`` (the runtime source) covers the fallback, and the
   repository copy matches the packaged copy.
4. No module keeps a module-level copy of a settings value — the
   configuration is read from the live ``settings`` object.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SETTINGS_PY = REPO_ROOT / "ricc" / "settings.py"
CONFIG_PY = REPO_ROOT / "ricc" / "config.py"
# ``ricc/params.yaml`` is the canonical copy: it is shipped as package
# data, so an installed wheel finds it next to the code. The repository
# root keeps a copy for the DVC pipeline; a test asserts they stay in
# sync so the two can never drift.
PARAMS_YAML = REPO_ROOT / "ricc" / "params.yaml"
ROOT_PARAMS_YAML = REPO_ROOT / "params.yaml"

# Domain keys that must exist in exactly one hardcoded fallback copy.
_DOMAIN_KEYS = (
    "ai_ml_nlp",
    "backend_databases",
    "frontend_ui",
    "devops_infra",
    "business_legal_fintech",
    "sysadmin_security",
    "career_team_management",
    "general_tech_chat",
)

# The settings fields that used to be snapshotted into module-level
# constants. No module may hold a frozen copy of them any more.
_SETTINGS_FIELDS = (
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "PARQUET_OUTPUT_DIR",
    "JSONL_OUTPUT_DIR",
    "SAMPLES_OUTPUT_DIR",
    "REPORTS_DIR",
    "RAW_EXPORT_DIRS",
    "MINHASH_NUM_PERM",
    "MINHASH_THRESHOLD",
    "MINHASH_SHINGLE_SIZE",
    "MAX_REPLY_TIME_GAP_HOURS",
    "MIN_QUESTION_WORDS",
    "MIN_ANSWER_WORDS",
    "MAX_THREAD_DEPTH",
    "MAX_SFT_TURNS",
    "SENTIMENT_DICT",
    "STOPWORDS_RU",
    "DOMAIN_TAXONOMY",
)


def test_the_snapshot_shim_is_gone() -> None:
    """``ricc/config.py`` must not come back.

    It re-exported the settings as a snapshot taken at import time, so a
    consumer's own ``Settings`` instance was ignored by every module that
    had imported it. The live ``settings`` object is the only source.
    """
    assert not CONFIG_PY.exists(), (
        "ricc/config.py is back: it freezes the settings at import time, so a consumer's "
        "own Settings instance is ignored. Import from ricc.settings instead."
    )


def test_no_module_freezes_a_settings_value() -> None:
    """No module may hold a module-level copy of a settings value.

    ``ricc/pipeline.py`` used to carry six such constants "for
    backward compatibility for tests that patch these module-level
    constants" — the tests were dictating the architecture. The
    configuration is now read from the live object everywhere.
    """
    offenders: list[str] = []
    for py in sorted((REPO_ROOT / "ricc").rglob("*.py")):
        if py.name in ("settings.py", "config.py"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            # Only module-level assignments count.
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _SETTINGS_FIELDS:
                    offenders.append(f"{py.relative_to(REPO_ROOT)}: {target.id}")
    assert not offenders, f"module-level copies of the settings: {offenders} — read settings.<NAME> at use time"


def test_taxonomy_fallback_exists_in_exactly_one_place() -> None:
    """The hardcoded 8-domain fallback must live only in ricc/settings.py."""
    settings_text = SETTINGS_PY.read_text(encoding="utf-8")

    settings_hits = sum(1 for k in _DOMAIN_KEYS if f'"{k}": DomainTaxonomyItem(' in settings_text)
    assert settings_hits == 8, (
        f"settings.py hardcoded fallback has {settings_hits}/8 domains — the offline fallback must stay complete"
    )
    # No DomainTaxonomyItem literals anywhere else in ricc/
    for py in (REPO_ROOT / "ricc").rglob("*.py"):
        if py.name == "settings.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        assert "DomainTaxonomyItem(" not in text, (
            f"{py.relative_to(REPO_ROOT)} defines a second taxonomy fallback — "
            "single source of truth is ricc/settings.py"
        )


def test_params_yaml_domains_cover_the_fallback() -> None:
    """params.yaml (runtime source) must define all 8 domains."""
    import yaml

    tax = yaml.safe_load(PARAMS_YAML.read_text(encoding="utf-8"))["domain_taxonomy"]
    assert set(tax) == set(_DOMAIN_KEYS), f"params.yaml domains drifted: {sorted(tax)}"
    # And the loaded runtime taxonomy must match params.yaml exactly.
    importlib.invalidate_caches()
    from ricc.settings import settings

    for key in _DOMAIN_KEYS:
        runtime = settings.domain_taxonomy[key]
        assert runtime.title == tax[key]["title"], f"{key}: title mismatch params vs runtime"
        assert runtime.keywords == tax[key]["keywords"], f"{key}: keywords mismatch params vs runtime"


def test_root_params_yaml_matches_the_packaged_copy() -> None:
    """The repository copy and the packaged copy must not drift.

    ``ricc/params.yaml`` is shipped as package data (an installed wheel
    finds it next to the code); the repository root keeps a copy for the
    DVC pipeline. Two divergent copies would mean the development and the
    installed package disagree about the thresholds.
    """
    if not ROOT_PARAMS_YAML.exists():
        return
    assert PARAMS_YAML.read_text(encoding="utf-8") == ROOT_PARAMS_YAML.read_text(encoding="utf-8"), (
        "the packaged params.yaml drifted from the repository copy"
    )


def test_a_custom_settings_instance_is_honoured() -> None:
    """The configuration a consumer builds must be the one that is used.

    This is the defect the snapshot shim had: a consumer that built its
    own ``Settings`` got a different configuration than the modules that
    had imported the shim.
    """
    import tempfile

    from ricc.settings import Settings

    with tempfile.TemporaryDirectory() as tmpdir:
        custom = Settings(output_dir=pathlib.Path(tmpdir))
        assert pathlib.Path(tmpdir) == custom.OUTPUT_DIR
        # The derived properties follow the field, they do not freeze it.
        assert custom.parquet_output_dir == custom.PARQUET_OUTPUT_DIR
        assert custom.reports_dir == custom.REPORTS_DIR
