"""Evaluation harnesses: academic benchmarks, enterprise suites, statistical power."""

from __future__ import annotations

# Every heuristic/rubric-based suite must stamp its outputs with this
# disclaimer: scores are concept-overlap + AST-parseability heuristics, not
# capability measurements (academic benchmarks are RETRACTED, see README).
HEURISTIC_DISCLAIMER = (
    "Rubric-based heuristic scores (concept-overlap + AST-parseability), not capability "
    "measurements. Academic benchmark numbers (HumanEval / RuMMLU / PPL / ROUGE) are "
    "RETRACTED — do not cite these scores as model capabilities."
)

HEURISTIC_DISCLAIMER_MD = (
    "> **Disclaimer:** rubric-based heuristic scores (concept-overlap + AST-parseability), "
    "not capability measurements. Academic benchmarks (HumanEval / RuMMLU / PPL / ROUGE) are "
    "RETRACTED — see README before citing."
)
