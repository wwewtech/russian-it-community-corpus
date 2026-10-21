"""Tests for the run context.

Without a run id, the log lines of two pipeline runs are
indistinguishable: there is no way to tell which run produced a given
message, and the JSON logs cannot be correlated with the artefacts of a
specific run.
"""

from __future__ import annotations

import unittest

from ricc.observability import (
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_run_context,
)


class TestRunContext(unittest.TestCase):
    def setUp(self) -> None:
        configure_logging(json_output=True)
        clear_run_context()

    def tearDown(self) -> None:
        clear_run_context()

    def test_binding_a_run_id_makes_it_visible(self):
        run_id = bind_run_context()
        self.assertTrue(run_id)
        self.assertEqual(get_run_context().get("run_id"), run_id)

    def test_an_explicit_run_id_is_kept(self):
        run_id = bind_run_context("my-run-42")
        self.assertEqual(run_id, "my-run-42")
        self.assertEqual(get_run_context().get("run_id"), "my-run-42")

    def test_extra_context_is_bound(self):
        bind_run_context("r1", pipeline_name="ricc")
        context = get_run_context()
        self.assertEqual(context.get("pipeline_name"), "ricc")

    def test_two_runs_get_different_ids(self):
        first = bind_run_context()
        clear_run_context()
        second = bind_run_context()
        self.assertNotEqual(first, second)

    def test_clearing_removes_the_context(self):
        bind_run_context("r1")
        clear_run_context()
        self.assertNotIn("run_id", get_run_context())


if __name__ == "__main__":
    unittest.main()
