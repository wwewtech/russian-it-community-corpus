import json
import tempfile
import unittest
from pathlib import Path

from src.validation.hub_reconciliation import (
    PINNED_HUB_REPO_ID,
    PINNED_HUB_REVISION,
    generate_reconciliation_report,
    load_pinned_hub_manifest,
    reconcile_artifacts,
)


class TestHubReconciliation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        self.mock_local_manifest = {
            "schema_version": 1,
            "artifacts": [
                {
                    "path": "dataset_output/parquet/full_clean_messages.parquet",
                    "sha256": "localhash1",
                    "bytes": 1000,
                    "rows": 50,
                },
                {
                    "path": "dataset_output/parquet/sft_dialogues.parquet",
                    "sha256": "localhash2",
                    "bytes": 2000,
                    "rows": 100,
                },
                {
                    "path": "dataset_output/parquet/rag_knowledge_base.parquet",
                    "sha256": "localhash3",
                    "bytes": 3000,
                    "rows": 150,
                },
            ],
        }

        self.mock_hub_manifest = {
            "schema_version": 1,
            "repo_id": PINNED_HUB_REPO_ID,
            "pinned_revision": PINNED_HUB_REVISION,
            "artifacts": {
                "data/full_clean_messages.parquet": {
                    "bytes": 1200,
                    "lfs_sha256": "hubhash1",
                    "published_rows": 60,
                },
                "data/sft_dialogues.parquet": {
                    "bytes": 2000,
                    "lfs_sha256": "localhash2",
                    "published_rows": 100,
                },
                "data/rag_knowledge_base.parquet": {
                    "bytes": 3500,
                    "lfs_sha256": "hubhash3",
                    "published_rows": 170,
                },
            },
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_reconcile_divergence_detected(self) -> None:
        report = reconcile_artifacts(self.mock_local_manifest, self.mock_hub_manifest, strict=False)
        self.assertEqual(report["status"], "DOCUMENTED_DIVERGENCE")
        self.assertEqual(len(report["artifacts"]), 3)

        # Artifact 1 has divergence
        art1 = report["artifacts"][0]
        self.assertEqual(art1["delta_bytes"], -200)
        self.assertEqual(art1["delta_rows"], -10)
        self.assertFalse(art1["hashes_match"])

        # Artifact 2 matches exactly
        art2 = report["artifacts"][1]
        self.assertEqual(art2["delta_bytes"], 0)
        self.assertEqual(art2["delta_rows"], 0)
        self.assertTrue(art2["hashes_match"])

    def test_reconcile_strict_fails_on_divergence(self) -> None:
        report = reconcile_artifacts(self.mock_local_manifest, self.mock_hub_manifest, strict=True)
        self.assertEqual(report["status"], "STRICT_MISMATCH")

    def test_generate_report_io(self) -> None:
        local_p = self.root / "local.json"
        hub_p = self.root / "hub.json"
        out_p = self.root / "report.json"

        local_p.write_text(json.dumps(self.mock_local_manifest), encoding="utf-8")
        hub_p.write_text(json.dumps(self.mock_hub_manifest), encoding="utf-8")

        rep = generate_reconciliation_report(local_p, hub_p, out_p, online=False, strict=False)
        self.assertTrue(out_p.exists())
        self.assertEqual(rep["pinned_revision"], PINNED_HUB_REVISION)

    def test_load_real_pinned_manifest(self) -> None:
        real_manifest = Path("reports/pinned_hub_manifest.json")
        if real_manifest.exists():
            data = load_pinned_hub_manifest(real_manifest)
            self.assertEqual(data["pinned_revision"], PINNED_HUB_REVISION)
            self.assertIn("data/full_clean_messages.parquet", data["artifacts"])
