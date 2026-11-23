"""Static and behavioural contracts for the read-only release pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
import yaml

from scripts.lock_requirements import _resolve
from scripts.lora_provenance import download_text
from scripts.release_preflight import verify_write_permission

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))

    def test_dry_run_and_release_execute_real_gate_before_mutation(self):
        jobs = self.workflow["jobs"]
        self.assertIn("workflow_dispatch", jobs["slo-gate-real"]["if"])
        self.assertIn("release", jobs["release-preflight"]["if"])
        self.assertIn("slo-gate-real", jobs["release-preflight"]["needs"])
        self.assertEqual(jobs["release"]["needs"], ["release-preflight"])
        for required in ("security", "test-windows", "coverage-regression", "eval-suites", "docs"):
            self.assertIn(required, jobs["release-preflight"]["needs"])
        self.assertEqual(jobs["release"]["permissions"], {"contents": "write"})
        self.assertEqual(jobs["release-preflight"]["permissions"], {"contents": "read"})
        release_steps = "\n".join(str(step.get("run", "")) for step in jobs["release"]["steps"])
        self.assertLess(release_steps.index("release tag"), release_steps.index("finalize_sync_all"))
        self.assertIn("revision=adapter['hf_model_revision']", release_steps)
        preflight_upload = next(
            step
            for step in jobs["release-preflight"]["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact")
        )
        self.assertIn("dist/*.whl", preflight_upload["with"]["path"])
        self.assertNotIn("name: python-package", release_steps)

    def test_catalog_cli_matches_verified_identity(self):
        result = subprocess.run(
            [sys.executable, "-B", "scripts/release_catalog.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Generated", result.stdout)
        catalog = (ROOT / "reports/LORA_MODEL_ZOO.md").read_text(encoding="utf-8")
        self.assertIn("HuggingFaceTB/SmolLM2-135M-Instruct", catalog)
        self.assertIn("47f5d37f9e08d921f23df2af8fbdaa614cc81bf3", catalog)
        self.assertNotIn("Qwen", catalog)
        self.assertIn("Full-corpus, full-epoch final adapters | 1", catalog)

    def test_clean_staging_root_supports_manifest_and_verification(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from ricc.validation.artifact_manifest import create_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "dataset_output/parquet"
            parquet.mkdir(parents=True)
            # The preflight creates this sibling before invoking the manifest.
            (root / "reports").mkdir()
            for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
                pq.write_table(pa.table({"text": ["example"]}), parquet / f"{name}.parquet")
            path = Path("reports/dataset_manifest.json")
            create_manifest(root, path)
            self.assertEqual(len(verify_manifest(root, path)["artifacts"]), 3)

    def test_release_container_uses_writable_inline_cache_only(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/release-container.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["docker-release"]
        self.assertIn("workflow_run.event == 'release'", job["if"])
        self.assertIn("workflow_run.conclusion == 'success'", job["if"])
        self.assertIn("workflow_dispatch", workflow[True])
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn('.event == "release" and .conclusion == "success"', commands)
        self.assertIn(".head_sha == $sha", commands)
        self.assertIn(".head_branch == $tag", commands)
        build = next(step for step in job["steps"] if step.get("uses", "").startswith("docker/build-push-action"))
        self.assertEqual(build["with"]["cache-to"], "type=inline")
        self.assertEqual(build["with"]["push"], True)

    def test_hf_fetch_is_read_only_and_digest_checked(self):
        source = (ROOT / "scripts/fetch_pinned_dataset.py").read_text(encoding="utf-8")
        self.assertIn("hashlib.sha256(data).hexdigest()", source)
        self.assertIn("PINNED_HUB_REVISION", source)
        self.assertNotIn("upload_", source)

    def test_write_scope_is_not_inferred_from_public_auth_check(self):
        self.assertEqual(
            verify_write_permission(
                {"name": "owner", "auth": {"accessToken": {"role": "write"}}},
                ("owner/data", "owner/models"),
            ),
            "owner",
        )
        with self.assertRaises(RuntimeError):
            verify_write_permission(
                {"name": "reader", "auth": {"accessToken": {"role": "read"}}},
                ("owner/data", "owner/models"),
            )

    def test_dependabot_refresh_uses_trusted_resolver_and_pr_manifests(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/dependabot-lock.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["refresh-lock"]
        pregreen = workflow["jobs"]["pregreen-lock"]
        self.assertIn("pull_request_target", workflow[True])
        self.assertIn("workflow_run", workflow[True])
        self.assertIn("pull_request.head.ref, 'dependabot/pip/'", pregreen["if"])
        self.assertIn("pull_request.head.repo.full_name == github.repository", pregreen["if"])
        pregreen_commands = "\n".join(step.get("run", "") for step in pregreen["steps"])
        self.assertIn("github.event.pull_request.head.sha", yaml.safe_dump(pregreen))
        self.assertNotIn('.user.login == "dependabot[bot]"', pregreen_commands)
        self.assertIn("python scripts/lock_requirements.py --refresh", pregreen_commands)
        self.assertIn("workflow_run.event == 'pull_request'", job["if"])
        self.assertIn("workflow_run.actor.login == 'dependabot[bot]'", job["if"])
        self.assertIn("workflow_run.conclusion == 'success'", job["if"])
        self.assertEqual(job["permissions"], {"contents": "write", "pull-requests": "write", "actions": "write"})
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("python scripts/lock_requirements.py --check", commands)
        self.assertIn("cp ../trusted/scripts/lock_requirements.py", commands)
        self.assertIn("isCrossRepository", commands)
        self.assertIn("github.event.workflow_run.head_sha", yaml.safe_dump(job))
        # GITHUB_TOKEN pushes deliberately do not emit a new push event.
        # Dispatch the live PR branch and pin the discovered run by head SHA;
        # GitHub's dispatch API rejects an arbitrary commit as its ref.
        self.assertIn('gh workflow run ci.yml --ref "$PR_REF"', commands)
        self.assertIn("PR_REF: ${{ steps.pr.outputs.ref }}", yaml.safe_dump(job))
        self.assertIn("--raw-field dependency_recheck=true", commands)
        self.assertIn("git rev-parse HEAD", commands)
        self.assertIn('event=workflow_dispatch&head_sha=\'"$REFRESH_SHA"', commands)
        self.assertIn(".head_branch == env.PR_REF and .head_sha == env.REFRESH_SHA", commands)
        self.assertIn("][0].id // empty", commands)
        self.assertNotIn("][0].databaseId // empty", commands)
        self.assertIn('gh run watch "$RUN_ID" --exit-status', commands)
        ci_dispatch = self.workflow[True]["workflow_dispatch"]["inputs"]
        self.assertIn("dependency_recheck", ci_dispatch)
        real_gate = self.workflow["jobs"]["slo-gate-real"]["if"]
        self.assertIn("!inputs.dependency_recheck", real_gate)
        self.assertIn("!inputs.dependency_recheck", self.workflow["jobs"]["hf-secrets-check"]["if"])
        self.assertIn("!inputs.dependency_recheck", self.workflow["jobs"]["release-preflight"]["if"])
        dependabot = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
        pip_update = next(update for update in dependabot["updates"] if update["package-ecosystem"] == "pip")
        self.assertEqual(pip_update["exclude-paths"], ["requirements.lock.txt"])
        self.assertEqual(pip_update["cooldown"]["default-days"], 7)
        self.assertNotIn("HF_TOKEN", commands)
        self.assertNotIn("pip install -r requirements", commands)

    def test_refresh_cutoff_is_passed_to_uv(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("scripts.lock_requirements.subprocess.run", return_value=completed) as run:
            _resolve("2030-01-02T03:04:05Z")
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--exclude-newer") + 1], "2030-01-02T03:04:05Z")
        self.assertIn("--generate-hashes", args)
        self.assertIn("--universal", args)


class ProvenanceFailureTest(unittest.TestCase):
    def test_transient_metadata_failure_is_not_legacy(self):
        with (
            patch("scripts.lora_provenance.requests.get", side_effect=requests.Timeout("offline")),
            self.assertRaises(requests.Timeout),
        ):
            download_text("owner/repo", "a/training_meta.json", repo_type="model", token=None)

    def test_metadata_404_is_the_only_absence(self):
        response = MagicMock(status_code=404)
        with patch("scripts.lora_provenance.requests.get", return_value=response):
            self.assertIsNone(download_text("owner/repo", "missing.json", repo_type="model", token=None))


if __name__ == "__main__":
    unittest.main()
