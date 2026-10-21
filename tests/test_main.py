"""tests/test_main.py — the master entrypoint must be fail-closed.

``main.main()`` returns a process exit code instead of silently ignoring the
pipeline result: any typed PipelineError or a failed dataset validation must
map to exit code 1 so CI/scripts can gate on it.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from main import main


class TestMainEntrypoint(unittest.TestCase):
    def _run(self, summary=None, error=None):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch("main.MasterDataPipeline") as pipeline_cls,
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            if error is not None:
                pipeline_cls.return_value.run_all.side_effect = error
            else:
                pipeline_cls.return_value.run_all.return_value = summary
            code = main()
        return code, out.getvalue(), err.getvalue()

    def test_returns_0_on_success(self):
        code, out, _ = self._run(summary={"validation_passed": True})
        self.assertEqual(code, 0)
        self.assertIn("Starting Russian IT Community", out)

    def test_returns_1_on_validation_failure(self):
        code, _, err = self._run(summary={"validation_passed": False})
        self.assertEqual(code, 1)
        self.assertIn("validation FAILED", err)

    def test_returns_1_on_typed_pipeline_error(self):
        from ricc.errors import IngestionError

        code, _, err = self._run(error=IngestionError("0 messages ingested"))
        self.assertEqual(code, 1)
        self.assertIn("stage 'ingestion'", err)


if __name__ == "__main__":
    unittest.main()
