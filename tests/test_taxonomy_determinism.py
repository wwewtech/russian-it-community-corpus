"""Regression tests for deterministic domain classification.

The classifier used to break score ties by dict iteration order, which depends on
PYTHONHASHSEED.  The same text could therefore be classified differently (and export
different parquet hashes) depending on the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Text that produces an exact score tie between two domains ("docker" -> devops_infra,
# "postgres" -> backend_databases).
TIE_TEXT = "docker postgres"

# A few texts, some of which are tie-inducing, to compare classification across seeds.
SEED_TEXTS = [TIE_TEXT, "docker postgres kubernetes", "привет", "python asyncio redis"]


def _child_script() -> str:
    return (
        "import json,sys\n"
        "sys.path.insert(0, " + json.dumps(str(REPO_ROOT)) + ")\n"
        "from ricc.taxonomy.classifier import DomainClassifier\n"
        "c = DomainClassifier()\n"
        "texts = json.loads(sys.argv[1])\n"
        "out = []\n"
        "for t in texts:\n"
        "    d, conf, scores = c.classify_text(t)\n"
        "    out.append([d, conf, scores])\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )


def _classify_under_seed(seed: str) -> list:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    proc = subprocess.run(
        [sys.executable, "-c", _child_script(), json.dumps(SEED_TEXTS)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"child failed for seed {seed}: {proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestTaxonomyDeterminism:
    def test_tie_text_produces_equal_scores(self):
        from ricc.taxonomy.classifier import DomainClassifier

        domain, _conf, scores = DomainClassifier().classify_text(TIE_TEXT)
        assert scores["devops_infra"] == scores["backend_databases"]
        assert domain == "backend_databases"  # alphabetical tie-break

    def test_classification_identical_across_hash_seeds(self):
        seeds = ["1", "2", "3", "4", "5", "6", "7", "8", "12345"]
        results = {seed: _classify_under_seed(seed) for seed in seeds}
        reference = results[seeds[0]]
        for seed in seeds[1:]:
            assert results[seed] == reference, f"non-deterministic result for PYTHONHASHSEED={seed}"

    def test_tie_result_identical_across_hash_seeds(self):
        seeds = ["1", "2", "3", "4", "5", "6", "7", "8"]
        domains = {_classify_under_seed(seed)[0][0] for seed in seeds}
        assert len(domains) == 1, f"tie broken differently across seeds: {domains}"
        assert domains.pop() == "backend_databases"

    def test_same_process_call_stable_regardless_of_keyword_input_order(self):
        from ricc.taxonomy.classifier import DomainClassifier

        classifier = DomainClassifier()
        baseline = classifier.classify_text(TIE_TEXT)

        # Rebuild the classifier with the taxonomy and keyword lists in reversed order:
        # the classification outcome must not depend on the order in which
        # keywords/domains were inserted.
        reversed_taxonomy = {
            domain: {**info, "keywords": list(reversed(list(info["keywords"])))}
            for domain, info in reversed(list(classifier.taxonomy.items()))
        }
        other = DomainClassifier(taxonomy=reversed_taxonomy).classify_text(TIE_TEXT)
        assert other == baseline

    def test_repeated_calls_are_stable(self):
        from ricc.taxonomy.classifier import DomainClassifier

        classifier = DomainClassifier()
        first = classifier.classify_text(TIE_TEXT)
        for _ in range(20):
            assert classifier.classify_text(TIE_TEXT) == first


if __name__ == "__main__":
    import unittest

    unittest.main()
