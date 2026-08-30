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
        # An unhandled exception in app.py would show up as at.error.
        self.assertFalse(at.error, msg=f"AppTest reported errors: {[e.value for e in at.error]}")


if __name__ == "__main__":
    unittest.main()
