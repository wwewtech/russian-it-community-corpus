"""Guard: ricc.config must stay a thin shim over ricc.settings (ADR-worthy).

History: config.py and settings.py each carried their own hardcoded copies
of the path constants, MinHash/graph tuning, sentiment lexicon, stopwords
and the full 8-domain taxonomy fallback (~520 duplicated lines, 79%
textual similarity, verified deep-equal 2026-10-15). Drift between the
copies was the standing hazard — the library read `settings` while the
CLI read `config`.

These tests fail if anyone reintroduces a second source of truth:
either by editing the shim's snapshot values directly, or by adding new
hardcoded taxonomy/lexicon literals outside ricc/settings.py.
"""

from __future__ import annotations

import importlib
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PY = REPO_ROOT / "ricc" / "config.py"
SETTINGS_PY = REPO_ROOT / "ricc" / "settings.py"
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


def test_config_is_a_thin_shim() -> None:
    """config.py must delegate every constant, not duplicate it."""
    text = CONFIG_PY.read_text(encoding="utf-8")
    # The only sanctioned literals are the re-export assignments.
    assert text.count("from ricc.settings import settings") == 1
    # No hardcoded taxonomy block: the shim must not contain domain keys.
    for key in _DOMAIN_KEYS:
        assert f'"{key}"' not in text, (
            f"config.py re-hardcodes taxonomy domain '{key}' — edit ricc/settings.py instead (single source of truth)"
        )
    # No lexicon literals either (sentiment words / stopwords).
    assert "хорошо" not in text, "config.py re-hardcodes the sentiment lexicon"
    assert len(text.splitlines()) < 120, (
        f"config.py grew to {len(text.splitlines())} lines — it must stay a thin compatibility shim over ricc.settings"
    )


def test_config_snapshots_equal_settings() -> None:
    """Every re-exported constant must deep-equal its settings counterpart."""
    from ricc import config
    from ricc.settings import settings

    pairs: list[tuple[str, object, object]] = [
        ("BASE_DIR", config.BASE_DIR, settings.BASE_DIR),
        ("OUTPUT_DIR", config.OUTPUT_DIR, settings.OUTPUT_DIR),
        ("PARQUET_OUTPUT_DIR", config.PARQUET_OUTPUT_DIR, settings.PARQUET_OUTPUT_DIR),
        ("JSONL_OUTPUT_DIR", config.JSONL_OUTPUT_DIR, settings.JSONL_OUTPUT_DIR),
        ("SAMPLES_OUTPUT_DIR", config.SAMPLES_OUTPUT_DIR, settings.SAMPLES_OUTPUT_DIR),
        ("REPORTS_DIR", config.REPORTS_DIR, settings.REPORTS_DIR),
        ("RAW_EXPORT_DIRS", config.RAW_EXPORT_DIRS, settings.RAW_EXPORT_DIRS),
        ("MINHASH_NUM_PERM", config.MINHASH_NUM_PERM, settings.MINHASH_NUM_PERM),
        ("MINHASH_THRESHOLD", config.MINHASH_THRESHOLD, settings.MINHASH_THRESHOLD),
        ("MINHASH_SHINGLE_SIZE", config.MINHASH_SHINGLE_SIZE, settings.MINHASH_SHINGLE_SIZE),
        ("MAX_REPLY_TIME_GAP_HOURS", config.MAX_REPLY_TIME_GAP_HOURS, settings.MAX_REPLY_TIME_GAP_HOURS),
        ("MIN_QUESTION_WORDS", config.MIN_QUESTION_WORDS, settings.MIN_QUESTION_WORDS),
        ("MIN_ANSWER_WORDS", config.MIN_ANSWER_WORDS, settings.MIN_ANSWER_WORDS),
        ("MAX_THREAD_DEPTH", config.MAX_THREAD_DEPTH, settings.MAX_THREAD_DEPTH),
        ("MAX_SFT_TURNS", config.MAX_SFT_TURNS, settings.MAX_SFT_TURNS),
        ("SENTIMENT_DICT", config.SENTIMENT_DICT, settings.SENTIMENT_DICT),
        ("STOPWORDS_RU", config.STOPWORDS_RU, settings.STOPWORDS_RU),
        ("DOMAIN_TAXONOMY", config.DOMAIN_TAXONOMY, settings.DOMAIN_TAXONOMY),
        ("load_params()", config.load_params(), settings.load_params()),
    ]
    for name, shim, origin in pairs:
        assert shim == origin, f"config.{name} != settings.{name} — shim snapshot drifted"


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
        "params.yaml at the repository root drifted from ricc/params.yaml — "
        "copy the canonical ricc/params.yaml to the root"
    )
