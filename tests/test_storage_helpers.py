"""Storage helpers take the bucket, repo or billing project explicitly and fail without one."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ENV = (
    "DATA_BUCKET",
    "DATA_PREFIX",
    "S3_ENDPOINT_URL",
    "GCS_BILLING_PROJECT",
    "HF_125M_RESULTS_REPO",
    "HF_760M_RESULTS_REPO",
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in STORAGE_ENV}


class S3DatasetFetchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fetch = _load("fetch_s3_28", "28_fetch_s3_dataset.py")

    def test_uri_layout(self) -> None:
        self.assertEqual(self.fetch.s3_uri("b", "", "books3", "val"), "s3://b/books3/val")
        self.assertEqual(self.fetch.s3_uri("b", "/p/q/", "books3"), "s3://b/p/q/books3")

    def test_bucket_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, _clean_env(), clear=True):
            with self.assertRaisesRegex(ValueError, "--bucket"):
                self.fetch.main(["--dest-root", td, "--env-file", str(Path(td) / "none.env"), "--dry-run"])

    def test_dry_run_uses_the_given_bucket_prefix_and_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, _clean_env(), clear=True):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = self.fetch.main(
                    [
                        "--dest-root", td, "--env-file", str(Path(td) / "none.env"), "--dry-run",
                        "--datasets", "books3", "--bucket", "my-bucket", "--prefix", "data/v1",
                        "--endpoint-url", "https://s3.example.com",
                    ]
                )
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("s3://my-bucket/data/v1/books3/train", text)
        self.assertIn("s3://my-bucket/data/v1/books3/val", text)
        self.assertIn("--endpoint-url https://s3.example.com", text)
        self.assertIn("--include train.fingerprint.json", text)


class AuthorCheckpointFetchTest(unittest.TestCase):
    def test_billing_project_is_required(self) -> None:
        module = _load("fetch_author_30", "30_fetch_author_orbax.py")
        err = io.StringIO()
        with mock.patch.dict(os.environ, _clean_env(), clear=True), mock.patch.object(
            module, "load_env_file", lambda _path: None
        ), mock.patch.object(sys, "argv", ["30", "--dry-run"]), contextlib.redirect_stderr(err):
            self.assertEqual(module.main(), 2)
        self.assertIn("--billing-project", err.getvalue())


class Tier1ExportTest(unittest.TestCase):
    def test_target_repo_is_required(self) -> None:
        module = _load("export_tier1_94", "94_export_tier1_to_hf.py")
        with mock.patch.dict(os.environ, _clean_env(), clear=True), mock.patch.object(
            module, "load_env_file", lambda _path: None
        ), mock.patch.object(sys, "argv", ["94"]):
            with self.assertRaisesRegex(ValueError, "--repo-id-125m"):
                module.main()


if __name__ == "__main__":
    unittest.main()
