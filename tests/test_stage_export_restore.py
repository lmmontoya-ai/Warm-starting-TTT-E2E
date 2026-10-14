"""Network-free round trip for scripts/40_export_stage.py --dest-dir and 46_restore_stage.py --source-dir."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from ttt.research.tier1 import read_restore_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_stage = _load("export_stage_40", "40_export_stage.py")
restore_stage = _load("restore_stage_46", "46_restore_stage.py")

PAPER = "protocol_r_125m_main_v1"
STAGE = "S0_PRETRAIN_FA_125M"
RUN = "pretrain-125m-fa"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_step(step_dir: Path, step: int) -> None:
    _write(step_dir / "_CHECKPOINT_METADATA", json.dumps({"step": step}))
    _write(step_dir / "model_weights" / "params.bin", f"weights-{step}")
    _write(step_dir / "opt_state" / "state.bin", f"opt-{step}")


def _make_local_run(root: Path, *, step: int = 4799) -> tuple[Path, Path]:
    exp_dir = root / "experiments"
    checkpoint_root = root / "checkpoints"
    run_dir = exp_dir / PAPER / STAGE / RUN
    _write(run_dir / "run_result.json", json.dumps({"status": "succeeded"}))
    _write(run_dir / "resolved_config.yaml", "training: {}\n")
    _write(run_dir / "not_exported.txt", "scratch")
    ckpt = checkpoint_root / PAPER / RUN
    _write(ckpt / "latest.json", json.dumps({"step": step}))
    _write(ckpt / f"step_metadata_{step:08d}.json", json.dumps({"step": step}))
    _make_step(ckpt / str(step), step)
    _make_step(ckpt / "100", 100)
    return exp_dir, checkpoint_root


def _make_bundle_with_two_steps(source_dir: Path) -> None:
    stage_root = source_dir / PAPER / "stages" / STAGE / RUN
    _write(stage_root / "experiment" / "resolved_config.yaml", "training: {}\n")
    _write(stage_root / "checkpoint" / "latest.json", json.dumps({"step": 3}))
    for step in (1, 3):
        _write(stage_root / "checkpoint" / f"step_metadata_{step:08d}.json", "{}")
        _make_step(stage_root / "checkpoint" / str(step), step)


def _quiet(fn, argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(argv)


class StageExportRestoreTest(unittest.TestCase):
    def test_local_export_then_restore_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp_dir, checkpoint_root = _make_local_run(root / "src")
            bundle = root / "bundle"
            code = _quiet(
                export_stage.main,
                [
                    "--paper-run-id", PAPER, "--stage-id", STAGE, "--run-id", RUN,
                    "--dest-dir", str(bundle),
                    "--exp-dir", str(exp_dir), "--checkpoint-root", str(checkpoint_root),
                    "--params-only",
                ],
            )
            self.assertEqual(code, 0)
            stage_root = bundle / PAPER / "stages" / STAGE / RUN
            self.assertTrue((stage_root / "experiment" / "run_result.json").is_file())
            self.assertFalse((stage_root / "experiment" / "not_exported.txt").exists())
            self.assertTrue((stage_root / "checkpoint" / "latest.json").is_file())
            self.assertTrue((stage_root / "checkpoint" / "step_metadata_00004799.json").is_file())
            self.assertTrue((stage_root / "checkpoint" / "4799" / "model_weights" / "params.bin").is_file())
            self.assertFalse((stage_root / "checkpoint" / "4799" / "opt_state").exists())
            self.assertFalse((stage_root / "checkpoint" / "100").exists())
            export_manifest = json.loads((stage_root / "export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(export_manifest["destination"]["kind"], "local")
            self.assertEqual(export_manifest["latest_step"], 4799)

            target_exp, target_ckpt = root / "dst" / "experiments", root / "dst" / "checkpoints"
            code = _quiet(
                restore_stage.main,
                [
                    "--source-dir", str(bundle),
                    "--source-paper-run-id", PAPER, "--source-stage-id", STAGE, "--source-run-id", RUN,
                    "--target-paper-run-id", "revision_v3_widthpreserving_v1",
                    "--exp-dir", str(target_exp), "--checkpoint-root", str(target_ckpt),
                    "--params-only",
                ],
            )
            self.assertEqual(code, 0)
            run_dir = target_exp / "revision_v3_widthpreserving_v1" / STAGE / RUN
            ckpt = target_ckpt / "revision_v3_widthpreserving_v1" / RUN
            self.assertTrue((run_dir / "resolved_config.yaml").is_file())
            self.assertEqual(json.loads((ckpt / "latest.json").read_text(encoding="utf-8"))["step"], 4799)
            self.assertEqual((ckpt / "4799" / "model_weights" / "params.bin").read_text(encoding="utf-8"), "weights-4799")
            self.assertTrue((ckpt / "4799" / "_CHECKPOINT_METADATA").is_file())
            self.assertTrue((ckpt / "step_metadata_00004799.json").is_file())
            manifest = json.loads((run_dir / "restore_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"], {"kind": "local", "source_dir": str(bundle.resolve())})
            self.assertIsNone(manifest["repo_id"])
            self.assertEqual(manifest["source_run_id"], RUN)
            self.assertEqual(manifest["restored_checkpoint_steps"], [4799])
            self.assertEqual(read_restore_manifest(run_dir), manifest)

            with self.assertRaises(FileExistsError):
                _quiet(
                    restore_stage.main,
                    [
                        "--source-dir", str(bundle),
                        "--source-paper-run-id", PAPER, "--source-stage-id", STAGE, "--source-run-id", RUN,
                        "--target-paper-run-id", "revision_v3_widthpreserving_v1",
                        "--exp-dir", str(target_exp), "--checkpoint-root", str(target_ckpt),
                    ],
                )

    def test_full_restore_keeps_optimizer_state_and_all_steps_mode_filters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "bundle"
            _make_bundle_with_two_steps(bundle)
            common = [
                "--source-dir", str(bundle),
                "--source-paper-run-id", PAPER, "--source-stage-id", STAGE, "--source-run-id", RUN,
                "--target-paper-run-id", "t",
            ]
            full_exp, full_ckpt = root / "full" / "e", root / "full" / "c"
            _quiet(restore_stage.main, [*common, "--exp-dir", str(full_exp), "--checkpoint-root", str(full_ckpt)])
            self.assertTrue((full_ckpt / "t" / RUN / "3" / "opt_state" / "state.bin").is_file())
            self.assertFalse((full_ckpt / "t" / RUN / "1").exists())

            all_exp, all_ckpt = root / "all" / "e", root / "all" / "c"
            _quiet(
                restore_stage.main,
                [*common, "--exp-dir", str(all_exp), "--checkpoint-root", str(all_ckpt),
                 "--params-only", "--all-model-checkpoints"],
            )
            restored = all_ckpt / "t" / RUN
            self.assertTrue((restored / "1" / "model_weights" / "params.bin").is_file())
            self.assertTrue((restored / "3" / "model_weights" / "params.bin").is_file())
            self.assertFalse((restored / "3" / "opt_state").exists())
            manifest = json.loads((all_exp / "t" / STAGE / RUN / "restore_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["restored_checkpoint_steps"], [1, 3])

    def test_exactly_one_source_is_required(self) -> None:
        base = [
            "--source-paper-run-id", PAPER, "--source-stage-id", STAGE, "--source-run-id", RUN,
            "--target-paper-run-id", "t",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                restore_stage.parse_args(base)
            with self.assertRaises(SystemExit):
                restore_stage.parse_args([*base, "--repo-id", "someone/repo", "--source-dir", "x"])
        self.assertEqual(restore_stage.parse_args([*base, "--repo-id", "someone/repo"]).repo_id, "someone/repo")

    def test_missing_stage_in_source_dir_names_the_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "stages/<stage_id>/<run_id>"):
                _quiet(
                    restore_stage.main,
                    [
                        "--source-dir", td,
                        "--source-paper-run-id", PAPER, "--source-stage-id", STAGE, "--source-run-id", RUN,
                        "--target-paper-run-id", "t", "--dry-run",
                    ],
                )

    def test_export_refuses_unsuccessful_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp_dir, checkpoint_root = _make_local_run(root)
            _write(exp_dir / PAPER / STAGE / RUN / "run_result.json", json.dumps({"status": "failed"}))
            with self.assertRaisesRegex(ValueError, "status=failed"):
                _quiet(
                    export_stage.main,
                    [
                        "--paper-run-id", PAPER, "--stage-id", STAGE, "--run-id", RUN,
                        "--dest-dir", str(root / "bundle"),
                        "--exp-dir", str(exp_dir), "--checkpoint-root", str(checkpoint_root),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
