"""
Streamlit AppTest smoke test for ``app.py``.

Marked ``@unittest.skipUnless`` so the test only runs when
``streamlit`` is importable AND we are not in a plain ``pytest`` run
on a developer machine that does not need the UI smoke check. The CI
workflow enables it explicitly via ``RUN_STREAMLIT_SMOKE=1``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

_RUN_SMOKE = os.environ.get("RUN_STREAMLIT_SMOKE") == "1"

_SKIP_REASON = (
    "Streamlit AppTest smoke checks are opt-in; set "
    "RUN_STREAMLIT_SMOKE=1 to run them (used by the streamlit-smoke "
    "CI step)."
)


@unittest.skipUnless(_RUN_SMOKE, _SKIP_REASON)
class TestAppStreamlitSmoke(unittest.TestCase):
    """Render ``app.py`` through Streamlit's AppTest harness."""

    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment gate
            raise unittest.SkipTest(f"streamlit AppTest not available: {exc}") from exc

    def test_app_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        # The test runs app.py with no parquet / reports present, so the
        # UI must still mount and emit the empty-state warnings we wired
        # in. We do NOT assert specific text because the UI is bilingual
        # and copy may evolve; we only assert that no exception bubbles
        # up the Streamlit runtime.
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
        at.run()
        # AppTest.error contains `st.error` widgets too — e.g. the intended
        # "AUDIT REQUIRED (0 detected)" empty-state privacy card. A runtime
        # Python exception is captured in `at.exception` instead. Only the
        # latter marks a broken application mount.
        self.assertFalse(at.exception, msg=f"Unhandled app exception: {[e.value for e in at.exception]}")
        self.assertTrue(len(at.warning) > 0 or len(at.error) > 0, "expected empty-data states to be visible")
        # The LoLoRA navigation and sidebar provenance sections must mount too.
        self.assertIn("RICC Studio", [t.value for t in at.sidebar.title])


if __name__ == "__main__":
    unittest.main()
