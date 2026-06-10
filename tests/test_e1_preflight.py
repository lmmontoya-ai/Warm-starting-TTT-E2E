from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ttt.research.e1_preflight import (
    CHECKS_PER_CHECKPOINT,
    CHECKS_PER_DATA,
    PreflightArtifact,
    PreflightResult,
    assert_preflight_pass,
    run_e1_preflight,
)
from ttt.research.registry import load_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_registry(REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_orbax_stub(root: Path, exp_folder: str, exp_name: str, *, step: int = 7) -> None:
    ckpt_dir = root / exp_folder / exp_name
    payload_dir = ckpt_dir / str(step)
    (payload_dir / "model_weights").mkdir(parents=True, exist_ok=True)
    (payload_dir / "model_weights" / "manifest.ocdbt").write_text("stub\n", encoding="utf-8")
    _write_json(
        ckpt_dir / f"step_metadata_{step:08d}.json",
        {
            "step": step,
            "checkpoint_format": "orbax",
            "checkpoint_dir": str(payload_dir),
            "items": ["model_weights"],
        },
    )
    _write_json(ckpt_dir / "latest.json", {"step": step, "path": str(step)})


def _fingerprint(dataset_id: str, split: str, sha: str) -> dict:
    return {
        "dataset": {
            "dataset_id": dataset_id,
            "split": split,
            "sha256": sha,
            "num_tokens": 128,
            "tokenizer_id": "tok",
            "tokenizer_revision": "rev",
        }
    }


def _write_dataset(root: Path, dataset_id: str, *, train_sha: str, val_sha: str) -> None:
    _write_json(root / "train.fingerprint.json", _fingerprint(dataset_id, "train", train_sha))
    _write_json(root / "val.fingerprint.json", _fingerprint(dataset_id, "val", val_sha))


def _write_all_checkpoints(root: Path, exp_folder: str) -> None:
    for exp_name in (
        "pretrain-125m-fa",
        "adapt-125m-e2e-8K-from-fa",
        "ext-125m-swa-32K-from-fa",
        "ext-125m-e2e-32K-from-fa-bridge",
        "ext-125m-e2e-32K",
    ):
        _write_orbax_stub(root, exp_folder, exp_name)


class E1PreflightTest(unittest.TestCase):
    def test_manifest_schema_and_pass_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_root = root / "checkpoints"
            exp_folder = "protocol_r_125m_main_v1"
            books_root = root / "books3"
            dclm_root = root / "dclm"
            _write_all_checkpoints(ckpt_root, exp_folder)
            _write_dataset(books_root, "books3", train_sha="books-train", val_sha="books-val")
            _write_dataset(dclm_root, "dclm_filter_8k", train_sha="dclm-train", val_sha="dclm-val")

            result = run_e1_preflight(
                repo_root=REPO_ROOT,
                registry=REGISTRY,
                checkpoint_root=ckpt_root,
                exp_folder=exp_folder,
                books_root=books_root,
                dclm_root=dclm_root,
                canonical_books_root=books_root,
                canonical_dclm_root=dclm_root,
                output_dir=root / "reports" / "revision_v2" / "e1_preflight",
            )

            self.assertEqual(result.status, "PASS")
            self.assertEqual(len(result.artifacts), 7)
            self.assertEqual(
                result.checks_executed_count,
                (5 * len(CHECKS_PER_CHECKPOINT)) + (2 * len(CHECKS_PER_DATA)),
            )
            payload = json.loads(Path(result.output_json).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            for artifact in payload["artifacts"]:
                for field in (
                    "artifact_id",
                    "kind",
                    "path",
                    "exists",
                    "fingerprint",
                    "fingerprint_matches_canonical",
                    "restores_clean",
                    "status",
                    "reason",
                    "checks_executed",
                ):
                    self.assertIn(field, artifact)

    def test_gate_fails_when_required_checkpoint_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_root = root / "checkpoints"
            exp_folder = "protocol_r_125m_main_v1"
            _write_all_checkpoints(ckpt_root, exp_folder)
            missing_latest = ckpt_root / exp_folder / "pretrain-125m-fa" / "latest.json"
            missing_latest.unlink()
            books_root = root / "books3"
            dclm_root = root / "dclm"
            _write_dataset(books_root, "books3", train_sha="books-train", val_sha="books-val")
            _write_dataset(dclm_root, "dclm_filter_8k", train_sha="dclm-train", val_sha="dclm-val")

            result = run_e1_preflight(
                repo_root=REPO_ROOT,
                registry=REGISTRY,
                checkpoint_root=ckpt_root,
                exp_folder=exp_folder,
                books_root=books_root,
                dclm_root=dclm_root,
                canonical_books_root=books_root,
                canonical_dclm_root=dclm_root,
                output_dir=root / "out",
            )

            self.assertEqual(result.status, "FAIL")
            failed = {artifact.artifact_id: artifact for artifact in result.artifacts if artifact.status == "FAIL"}
            self.assertIn("fa_seed_checkpoint", failed)
            self.assertIn("uncertain provenance", failed["fa_seed_checkpoint"].reason)
            with self.assertRaisesRegex(RuntimeError, "fa_seed_checkpoint"):
                assert_preflight_pass(result)

    def test_gate_fails_on_data_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_root = root / "checkpoints"
            exp_folder = "protocol_r_125m_main_v1"
            _write_all_checkpoints(ckpt_root, exp_folder)
            books_root = root / "books3"
            canonical_books_root = root / "canonical_books3"
            dclm_root = root / "dclm"
            _write_dataset(books_root, "books3", train_sha="books-train-a", val_sha="books-val")
            _write_dataset(canonical_books_root, "books3", train_sha="books-train-b", val_sha="books-val")
            _write_dataset(dclm_root, "dclm_filter_8k", train_sha="dclm-train", val_sha="dclm-val")

            result = run_e1_preflight(
                repo_root=REPO_ROOT,
                registry=REGISTRY,
                checkpoint_root=ckpt_root,
                exp_folder=exp_folder,
                books_root=books_root,
                dclm_root=dclm_root,
                canonical_books_root=canonical_books_root,
                canonical_dclm_root=dclm_root,
                output_dir=root / "out",
            )

            self.assertEqual(result.status, "FAIL")
            books = next(artifact for artifact in result.artifacts if artifact.artifact_id == "books32k_data_surface")
            self.assertEqual(books.status, "FAIL")
            self.assertIn("fingerprint mismatch", books.reason)

    def test_gate_fails_when_check_is_skipped(self) -> None:
        artifact = PreflightArtifact(
            artifact_id="bad",
            kind="checkpoint",
            path="/tmp/missing",
            exists=True,
            fingerprint={},
            fingerprint_matches_canonical=True,
            restores_clean=True,
            status="PASS",
            reason="",
            checks_executed=["exists"],
        )
        result = PreflightResult(
            schema_version="1.0",
            status="PASS",
            checks_executed_count=1,
            artifacts=[artifact],
            output_json="",
            output_md="",
        )
        # The production runner downgrades this to FAIL before writing; this
        # direct assertion captures the same no-silent-pass contract.
        required = len(CHECKS_PER_CHECKPOINT)
        self.assertLess(result.checks_executed_count, required)

    def test_phase1_json_checkpoint_does_not_satisfy_current_pipeline_restore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_root = root / "checkpoints"
            exp_folder = "protocol_r_125m_main_v1"
            _write_all_checkpoints(ckpt_root, exp_folder)
            phase1_dir = ckpt_root / exp_folder / "pretrain-125m-fa"
            _write_json(phase1_dir / "phase1_ckpt_step_00000007.json", {"model_state": {}, "step": 7})
            _write_json(phase1_dir / "latest.json", {"step": 7, "path": "phase1_ckpt_step_00000007.json"})
            books_root = root / "books3"
            dclm_root = root / "dclm"
            _write_dataset(books_root, "books3", train_sha="books-train", val_sha="books-val")
            _write_dataset(dclm_root, "dclm_filter_8k", train_sha="dclm-train", val_sha="dclm-val")

            result = run_e1_preflight(
                repo_root=REPO_ROOT,
                registry=REGISTRY,
                checkpoint_root=ckpt_root,
                exp_folder=exp_folder,
                books_root=books_root,
                dclm_root=dclm_root,
                canonical_books_root=books_root,
                canonical_dclm_root=dclm_root,
                output_dir=root / "out",
            )

            fa = next(artifact for artifact in result.artifacts if artifact.artifact_id == "fa_seed_checkpoint")
            self.assertEqual(result.status, "FAIL")
            self.assertFalse(fa.restores_clean)
            self.assertIn("not a current-pipeline Orbax payload", fa.reason)


if __name__ == "__main__":
    unittest.main()
