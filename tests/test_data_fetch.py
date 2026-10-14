"""Network-free tests for scripts/00_data_fetch.py (billing gate and budget selection)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "00_data_fetch.py"
    spec = importlib.util.spec_from_file_location("data_fetch_00", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_fetch = _load_module()


def _zarr_meta(shape: int, chunk: int) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [shape],
        "data_type": "uint32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [chunk]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    }


class FakeBucketTransport:
    """Serves gs:// URIs from a local directory tree instead of GCS."""

    def __init__(self, root: Path, billing_project: str = "test-project") -> None:
        self.root = root
        self.billing_project = billing_project
        self.copied: list[str] = []

    def _path(self, uri: str) -> Path:
        assert uri.startswith("gs://"), uri
        return self.root / uri[len("gs://") :]

    def cat(self, uri: str) -> str:
        return self._path(uri).read_text(encoding="utf-8")

    def cp(self, sources: list[str], dest: str, *, recursive: bool) -> None:
        assert not recursive
        for uri in sources:
            self.copied.append(uri)
            shutil.copy2(self._path(uri), Path(dest) / uri.rsplit("/", 1)[-1])


def _make_fake_bucket(root: Path, uri: str, *, splits: dict[str, tuple[int, int]]) -> None:
    dataset_root = root / uri[len("gs://") :]
    dataset_root.mkdir(parents=True)
    (dataset_root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}), encoding="utf-8")
    for split, (shape, chunk) in splits.items():
        split_root = dataset_root / split
        (split_root / "c").mkdir(parents=True)
        (split_root / "zarr.json").write_text(json.dumps(_zarr_meta(shape, chunk)), encoding="utf-8")
        for chunk_id in range(-(-shape // chunk)):
            (split_root / "c" / str(chunk_id)).write_bytes(f"chunk-{split}-{chunk_id}".encode())


class BillingProjectTest(unittest.TestCase):
    def test_missing_billing_project_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(data_fetch.MissingBillingProjectError, "Requester Pays"):
            data_fetch.resolve_billing_project(None, environ={})

    def test_env_fallback_and_cli_precedence(self) -> None:
        env = {data_fetch.BILLING_ENV_VAR: "from-env"}
        self.assertEqual(data_fetch.resolve_billing_project(None, environ=env), "from-env")
        self.assertEqual(data_fetch.resolve_billing_project("from-cli", environ=env), "from-cli")

    def test_main_refuses_to_download_without_billing_project(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != data_fetch.BILLING_ENV_VAR}
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, env, clear=True):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = data_fetch.main(["--dest-root", td, "--budget", "125m"])
        self.assertEqual(code, 2)
        self.assertIn("--billing-project", out.getvalue())

    def test_copy_commands_carry_the_billing_project(self) -> None:
        gcloud = data_fetch.build_cp_command(
            tool_name="gcloud",
            base_cmd=["gcloud", "storage"],
            sources=["gs://b/x"],
            dest="/tmp/x",
            billing_project="proj",
            recursive=True,
        )
        self.assertEqual(gcloud, ["gcloud", "storage", "cp", "-r", "--billing-project=proj", "gs://b/x", "/tmp/x"])
        gsutil = data_fetch.build_cp_command(
            tool_name="gsutil",
            base_cmd=["gsutil", "-m"],
            sources=["gs://b/x"],
            dest="/tmp/x",
            billing_project="proj",
            recursive=False,
        )
        self.assertEqual(gsutil, ["gsutil", "-m", "-u", "proj", "cp", "gs://b/x", "/tmp/x"])


class BudgetSelectionTest(unittest.TestCase):
    def test_token_targets_match_the_published_packages(self) -> None:
        self.assertEqual(
            data_fetch.PACKAGE_TOKEN_TARGETS,
            {
                "125m": {"dclm_filter_8k": 2_520_000_000, "books3": 126_000_000},
                "760m": {"dclm_filter_8k": 15_200_000_000, "books3": 759_000_000},
            },
        )

    def test_plan_keeps_whole_leading_chunks(self) -> None:
        plan = data_fetch.plan_split(
            source_meta=_zarr_meta(20 * 1_048_576, 1_048_576),
            target_shape=data_fetch.PACKAGE_TOKEN_TARGETS["125m"]["books3"] // 12,
            selection_policy="contiguous_prefix_chunks",
        )
        # 10.5M tokens over 1,048,576-token chunks needs 11 chunks; the last is partial.
        self.assertEqual(plan["selected_chunk_count"], 11)
        self.assertEqual(plan["selected_chunk_start"], 0)
        self.assertEqual(plan["selected_chunk_end"], 10)
        self.assertEqual(plan["exported_shape"], 10_500_000)
        self.assertEqual(plan["source_chunk_count"], 20)
        self.assertEqual(plan["selection_warning"], "")

    def test_plan_rejects_targets_beyond_the_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds source shape"):
            data_fetch.plan_split(
                source_meta=_zarr_meta(100, 10), target_shape=101, selection_policy="contiguous_prefix_chunks"
            )

    def test_plan_warns_when_target_is_inside_the_first_chunk(self) -> None:
        plan = data_fetch.plan_split(
            source_meta=_zarr_meta(100, 50), target_shape=20, selection_policy="contiguous_prefix_chunks"
        )
        self.assertEqual(plan["selected_chunk_count"], 1)
        self.assertIn("strict prefix", plan["selection_warning"])

    def test_export_budget_writes_loader_layout_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bucket = Path(td) / "bucket"
            dest = Path(td) / "data"
            uris = {"dclm_filter_8k": "gs://fake-dclm/data.zarr", "books3": "gs://fake-books/data.zarr"}
            _make_fake_bucket(bucket, uris["dclm_filter_8k"], splits={"train": (10, 4), "val": (6, 4)})
            _make_fake_bucket(bucket, uris["books3"], splits={"train": (9, 3), "val": (3, 3)})
            targets = {"tiny": {"dclm_filter_8k": 5, "books3": 3}}
            transport = FakeBucketTransport(bucket)
            with mock.patch.dict(data_fetch.DATASETS, uris), mock.patch.dict(
                data_fetch.PACKAGE_TOKEN_TARGETS, targets
            ), contextlib.redirect_stdout(io.StringIO()):
                manifest = data_fetch.export_budget(
                    budget="tiny",
                    datasets=["dclm_filter_8k", "books3"],
                    dest_root=dest,
                    transport=transport,
                    val_policy="full",
                    batch_size=2,
                    dry_run=False,
                )
                # A second run finds every chunk in place and copies nothing.
                rerun = FakeBucketTransport(bucket)
                data_fetch.export_budget(
                    budget="tiny",
                    datasets=["dclm_filter_8k"],
                    dest_root=dest,
                    transport=rerun,
                    val_policy="full",
                    batch_size=2,
                    dry_run=False,
                )

            self.assertEqual(rerun.copied, [])
            dclm = dest / "dclm_filter_8k"
            self.assertTrue((dclm / "zarr.json").is_file())
            train_meta = json.loads((dclm / "train" / "zarr.json").read_text(encoding="utf-8"))
            self.assertEqual(train_meta["shape"], [5])
            self.assertEqual(sorted(p.name for p in (dclm / "train" / "c").iterdir()), ["0", "1"])
            self.assertEqual((dclm / "train" / "c" / "1").read_bytes(), b"chunk-train-1")
            val_meta = json.loads((dclm / "val" / "zarr.json").read_text(encoding="utf-8"))
            self.assertEqual(val_meta["shape"], [6])
            self.assertEqual(sorted(p.name for p in (dclm / "val" / "c").iterdir()), ["0", "1"])
            books_train = sorted(p.name for p in (dest / "books3" / "train" / "c").iterdir())
            self.assertEqual(books_train, ["0"])

            on_disk = json.loads((dest / "export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["package_id"], "paper_budget_tiny_val-full")
            self.assertEqual(on_disk["train_selection_policy"], "contiguous_prefix_chunks")
            self.assertEqual(on_disk["billing_project"], "test-project")
            splits = {row["dataset_name"]: row["splits"] for row in manifest["datasets"]}
            self.assertEqual(splits["dclm_filter_8k"][0]["selected_chunk_count"], 2)
            self.assertEqual(splits["dclm_filter_8k"][1]["selection_policy"], "full_split_copy")
            self.assertTrue(data_fetch.check_dataset(dclm))


if __name__ == "__main__":
    unittest.main()
