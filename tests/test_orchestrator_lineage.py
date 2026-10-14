from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from omegaconf import OmegaConf

from ttt.infra.paths import path_name_error, require_path_name
from ttt.research.orchestrator import (
    OrchestratorOptions,
    build_train_command,
    override_value,
    plan_lineage_warm_start,
    run_stage,
)
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec, CheckpointRef, EvalSpec, StageSpec
from ttt.runtime import _check_continuation_has_steps, validate_run_names

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml"
LADDER_SCRIPT = REPO_ROOT / "scripts" / "23_warmstart_registry.py"


def _opts(root: Path, **kwargs) -> OrchestratorOptions:
    base = dict(
        deploy="interactive",
        runtime_mode="token_stats",
        exp_dir=root / "experiments",
        checkpoint_root=root / "checkpoints",
        profile_root=root / "profiles",
        dclm_root=root / "dclm",
        books_root=root / "books",
        exp_folder="ladder",
        wandb_entity="none",
        wandb_project="none",
        wandb_key="none",
        paper_run_id="paper",
    )
    base.update(kwargs)
    return OrchestratorOptions(**base)


class PathNameContractTest(unittest.TestCase):
    def test_plain_names_are_accepted(self) -> None:
        for value in ("warmstart_smoke", "ext-125m-e2e-32K-from-fa-direct-seed001", "a.b"):
            self.assertIsNone(path_name_error(value, field="x"), value)
            self.assertEqual(require_path_name(value, field="x"), value)

    def test_paths_are_rejected_on_every_os(self) -> None:
        for value in ("C:\\Users\\me\\exp2", "C:exp", "/tmp/exp2", "a/b", "a\\b", "..", ".", "", " x"):
            with self.subTest(value=value):
                self.assertIsNotNone(path_name_error(value, field="exp_folder"))
                with self.assertRaises(ValueError):
                    require_path_name(value, field="exp_folder")

    def test_orchestrator_options_reject_absolute_exp_folder(self) -> None:
        root = Path(tempfile.gettempdir())
        for value in ("C:\\Users\\me\\exp2", "/tmp/exp2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "exp_folder"):
                    _opts(root, exp_folder=value)

    def test_ladder_cli_rejects_absolute_exp_folder(self) -> None:
        spec = importlib.util.spec_from_file_location("warmstart_registry_cli", LADDER_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with redirect_stderr(StringIO()) as err, self.assertRaises(SystemExit) as ctx:
            module.parse_args(
                ["--exp-folder", "/tmp/exp2", "--dclm-root", "d", "--books-root", "b"]
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--exp-folder must be a single folder name", err.getvalue())

    def test_runtime_rejects_path_like_names(self) -> None:
        def cfg(**training):
            base = dict(
                exp_folder="ladder",
                exp_name="run",
                paper_run_id="",
                stage_id="",
                run_id="",
                resume_exp_name="",
                load_part="none",
            )
            base.update(training)
            return OmegaConf.create({"training": base})

        validate_run_names(cfg())
        # A path-like resume name is only an error when it is actually used.
        validate_run_names(cfg(resume_exp_name="x/y", load_part="none"))
        for bad in (
            dict(exp_folder="C:\\tmp\\exp2"),
            dict(exp_folder="/tmp/exp2"),
            dict(exp_name="a/b"),
            dict(paper_run_id="/abs"),
            dict(resume_exp_name="../other/run", load_part="params"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_run_names(cfg(**bad))


class LineageWarmStartTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stages = load_registry(REGISTRY_PATH).stage_map()
        self.root = Path(tempfile.gettempdir()) / "lineage-unit"

    def test_override_value_returns_last_assignment(self) -> None:
        overrides = ["training.load_part=params", "+training.load_part=all", "training.x=1"]
        self.assertEqual(override_value(overrides, "training.load_part"), "all")
        self.assertIsNone(override_value(overrides, "training.resume_exp_name"))

    def test_continuation_resumes_resolved_parent_and_extends_steps(self) -> None:
        stage = self.stages["S2_MINUS_CONT_125M"]
        # The registry pins the revision-v2 seed-suffixed run; a ladder run
        # produces the parent under the stage's own exp_name instead.
        self.assertIn(
            "training.resume_exp_name=ext-125m-e2e-32K-from-fa-direct-seed001", stage.extra_overrides
        )
        parent = CheckpointRef(
            checkpoint_id="S2_MINUS_125M",
            exp_folder="ladder",
            exp_name="ext-125m-e2e-32K-from-fa-direct",
            step=3,
        )
        plan = plan_lineage_warm_start(
            stage=stage, opts=_opts(self.root), parent_refs=[parent], steps=4
        )
        self.assertEqual(plan.overrides, ["training.resume_exp_name=ext-125m-e2e-32K-from-fa-direct"])
        self.assertEqual(plan.total_steps, 3 + 1 + 4)
        self.assertEqual(len(plan.notes), 2)

    def test_params_warm_start_keeps_step_budget(self) -> None:
        stage = self.stages["S2_125M"]
        parent = CheckpointRef(
            checkpoint_id="S2_ADAPT_125M", exp_folder="ladder", exp_name="adapt-125m-e2e-8K-from-fa", step=9
        )
        plan = plan_lineage_warm_start(stage=stage, opts=_opts(self.root), parent_refs=[parent], steps=4)
        self.assertEqual(plan.overrides, ["training.resume_exp_name=adapt-125m-e2e-8K-from-fa"])
        self.assertEqual(plan.total_steps, 4)
        self.assertEqual(plan.notes, [])

    def test_explicit_resume_path_and_foreign_parents_are_left_alone(self) -> None:
        stage = self.stages["S2_MINUS_CONT_125M"]
        parent = CheckpointRef(checkpoint_id="S2_MINUS_125M", exp_folder="ladder", exp_name="p", step=3)
        opts = _opts(self.root)
        cases = {
            "explicit_path": dict(
                parent_refs=[parent], explicit_resume_checkpoint_path=self.root / "ckpt" / "p"
            ),
            "other_folder": dict(
                parent_refs=[CheckpointRef(checkpoint_id="S2_MINUS_125M", exp_folder="other", exp_name="p")]
            ),
            "no_parent": dict(parent_refs=[]),
            "two_parents": dict(parent_refs=[parent, parent]),
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                plan = plan_lineage_warm_start(stage=stage, opts=opts, steps=4, **kwargs)
                self.assertEqual(plan.overrides, [])
                self.assertEqual(plan.total_steps, 4)

    def test_build_train_command_hydra_run_dir_is_opt_in(self) -> None:
        stage = self.stages["S0_PRETRAIN_FA_125M"]
        opts = _opts(self.root)
        plain = build_train_command(stage=stage, opts=opts, steps=2, run_id="r")
        self.assertFalse(any(part.startswith("hydra.run.dir=") for part in plain))
        with_dir = build_train_command(
            stage=stage, opts=opts, steps=2, run_id="r", hydra_run_dir=self.root / "run" / "hydra"
        )
        self.assertEqual(with_dir[:-1], plain)
        self.assertTrue(with_dir[-1].startswith("hydra.run.dir="))

    def test_dry_run_command_restores_lineage_parent(self) -> None:
        stage = self.stages["S2_MINUS_CONT_125M"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            opts = _opts(root, dry_run=True, dummy_dataset=True)
            parent = CheckpointRef(
                checkpoint_id="S2_MINUS_125M", exp_folder="ladder", exp_name="ext-125m-e2e-32K-from-fa-direct", step=5
            )
            result = run_stage(
                stage=stage,
                stage_map=self.stages,
                opts=opts,
                budget=BudgetSpec(budget_id="b", ext_steps=2),
                eval_spec=EvalSpec(eval_id="e"),
                repo_root=REPO_ROOT,
                parent_refs_override=[parent],
            )
            self.assertEqual(result.status, "dry_run")
            manifest = json.loads((Path(result.run_dir) / "run_manifest.json").read_text())
            command = manifest["command"]
            self.assertEqual(override_value(command, "training.resume_exp_name"), parent.exp_name)
            self.assertIn("training.total_steps=8", command)
            self.assertTrue(command[-1].startswith("hydra.run.dir="))


class RunStageOutcomeTest(unittest.TestCase):
    def test_zero_exit_without_checkpoint_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_repo = root / "fake_repo"
            (fake_repo / "ttt").mkdir(parents=True)
            # Stands in for a runtime that exits 0 but trains nothing.
            (fake_repo / "ttt" / "train.py").write_text("raise SystemExit(0)\n")
            stage = StageSpec(
                stage_id="S0",
                kind="pretrain",
                model_key="m",
                path_group="scratch",
                experiment="125m/pretrain/pretrain-125m-fa",
                exp_name="pretrain-125m-fa",
                dataset_ids=["dclm_filter_8k"],
            )
            opts = _opts(
                root,
                runtime_mode="simulate",
                python_executable=sys.executable,
                repo_root=fake_repo,
            )
            result = run_stage(
                stage=stage,
                stage_map={stage.stage_id: stage},
                opts=opts,
                budget=BudgetSpec(budget_id="b", pretrain_steps=2),
                eval_spec=EvalSpec(eval_id="e"),
                repo_root=REPO_ROOT,
            )
            self.assertEqual(result.status, "failed")
            self.assertIn("no usable checkpoint", result.error_message)
            events = (Path(result.run_dir) / "events.jsonl").read_text()
            self.assertIn("run_failed", events)


class ContinuationGuardTest(unittest.TestCase):
    def test_new_run_with_nothing_to_do_fails(self) -> None:
        root = Path(tempfile.gettempdir())
        with self.assertRaisesRegex(ValueError, "leaves nothing to run"):
            _check_continuation_has_steps(
                load_part="all",
                start_step=4,
                total_steps=4,
                resume_dir=root / "ckpt" / "parent",
                checkpoint_dir=root / "ckpt" / "child",
                resume_exp_name="parent",
            )

    def test_finished_same_run_and_params_restores_are_allowed(self) -> None:
        root = Path(tempfile.gettempdir())
        _check_continuation_has_steps(
            load_part="all",
            start_step=4,
            total_steps=4,
            resume_dir=root / "ckpt" / "run",
            checkpoint_dir=root / "ckpt" / "run",
            resume_exp_name="run",
        )
        _check_continuation_has_steps(
            load_part="params",
            start_step=0,
            total_steps=4,
            resume_dir=root / "ckpt" / "parent",
            checkpoint_dir=root / "ckpt" / "child",
            resume_exp_name="parent",
        )


if __name__ == "__main__":
    unittest.main()
