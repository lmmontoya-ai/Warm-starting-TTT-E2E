"""The upstream parity scripts fail fast, with clone instructions, when the checkout is missing."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ttt.research.upstream_reference import (
    UPSTREAM_REPO_URL,
    UpstreamReferenceError,
    resolve_reference_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UpstreamReferenceTest(unittest.TestCase):
    def test_missing_checkout_names_the_clone_command(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(UpstreamReferenceError) as ctx:
                resolve_reference_root(Path(td))
        message = str(ctx.exception)
        self.assertIn(f"git clone {UPSTREAM_REPO_URL}", message)
        self.assertIn("ttte2e_reference", message)
        self.assertIn("--reference-root", message)

    def test_checkout_with_trainer_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkout = Path(td) / "e2e"
            (checkout / "ttt").mkdir(parents=True)
            (checkout / "pyproject.toml").write_text("[project]\nname='e2e'\n", encoding="utf-8")
            (checkout / "ttt" / "train.py").write_text("", encoding="utf-8")
            self.assertEqual(resolve_reference_root(Path(td), checkout), checkout.resolve())
            default_root = Path(td) / "repo"
            (default_root / "ttte2e_reference").mkdir(parents=True)
            (checkout).rename(default_root / "ttte2e_reference" / "e2e")
            self.assertEqual(
                resolve_reference_root(default_root), (default_root / "ttte2e_reference" / "e2e").resolve()
            )

    def test_parity_scripts_exit_before_doing_work(self) -> None:
        cases = {
            "41_run_reference_125m_32k_fa_smoke.py": ["--books-root", "b", "--resume-checkpoint-dir", "c"],
            "50_run_reference_125m_32k_swa_smoke.py": ["--books-root", "b", "--resume-checkpoint-dir", "c"],
            "56_run_reference_125m_s3_pretrain_smoke.py": ["--dclm-root", "d"],
        }
        for filename, extra in cases.items():
            with self.subTest(script=filename), tempfile.TemporaryDirectory() as td:
                module = _load(filename.split("_")[0] + "_parity", filename)
                argv = [filename, "--repo-root", td, "--dry-run", *extra]
                err = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err):
                    self.assertEqual(module.main(), 2)
                self.assertIn(UPSTREAM_REPO_URL, err.getvalue())
                self.assertFalse((Path(td) / "artifacts").exists())

        wrapper = _load("parity_58", "58_run_reference_entrypoint_no_wandb.py")
        with tempfile.TemporaryDirectory() as td:
            err = io.StringIO()
            argv = ["58", "--reference-root", td, "--", "+deploy=interactive"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err):
                self.assertEqual(wrapper.main(), 2)
            self.assertIn(UPSTREAM_REPO_URL, err.getvalue())


if __name__ == "__main__":
    unittest.main()
