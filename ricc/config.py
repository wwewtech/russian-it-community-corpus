"""Compatibility shim over :mod:`ricc.settings` — the single source of config.

Historically this module *owned* the path constants, MinHash/graph tuning
parameters, sentiment lexicon, stopwords and the domain taxonomy (with its
own hardcoded fallback copy of all eight domains — ~520 lines duplicated
from ``settings.py``, verified deep-equal at delegation time 2026-10-15).
Drift between the two copies was the standing hazard: the library reads
``settings`` while ``cli.py`` read these module constants, so a fix applied
to one layer silently missed the other.

Everything now lives in :class:`ricc.settings.Settings`. This module only
re-exports the same names so existing imports keep working:

    from ricc.config import OUTPUT_DIR, RAW_EXPORT_DIRS  # still fine

New code must import from ``ricc.settings`` directly (``from ricc.settings
import settings``). The re-exports below are snapshots taken at import
time — equivalent to the previous behavior, and patched tests target
``ricc.pipeline.*`` constants, which are likewise module-level snapshots.
"""

from typing import Any

from ricc.settings import settings

# Base Directories
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
OUTPUT_DIR = settings.OUTPUT_DIR
PARQUET_OUTPUT_DIR = settings.PARQUET_OUTPUT_DIR
JSONL_OUTPUT_DIR = settings.JSONL_OUTPUT_DIR
SAMPLES_OUTPUT_DIR = settings.SAMPLES_OUTPUT_DIR
REPORTS_DIR = settings.REPORTS_DIR

# Raw Input Data Paths (auto-discovered, with the same dated fallback as
# Settings._auto_discover_exports)
RAW_EXPORT_DIRS = settings.RAW_EXPORT_DIRS

# MinHash LSH Settings
MINHASH_NUM_PERM = settings.MINHASH_NUM_PERM
MINHASH_THRESHOLD = settings.MINHASH_THRESHOLD
MINHASH_SHINGLE_SIZE = settings.MINHASH_SHINGLE_SIZE

# Dialogue Reconstruction Settings
MAX_REPLY_TIME_GAP_HOURS = settings.MAX_REPLY_TIME_GAP_HOURS
MIN_QUESTION_WORDS = settings.MIN_QUESTION_WORDS
MIN_ANSWER_WORDS = settings.MIN_ANSWER_WORDS
MAX_THREAD_DEPTH = settings.MAX_THREAD_DEPTH
MAX_SFT_TURNS = settings.MAX_SFT_TURNS

# Sentiment & Emotional Lexicon / NLP stopwords
SENTIMENT_DICT = settings.SENTIMENT_DICT
STOPWORDS_RU = settings.STOPWORDS_RU

# Domain Taxonomy & Keywords (params.yaml with the Settings-level fallback)
DOMAIN_TAXONOMY = settings.DOMAIN_TAXONOMY


def load_params() -> dict[str, Any]:
    """Load raw ``params.yaml`` — delegates to :meth:`Settings.load_params`."""
    return settings.load_params()
