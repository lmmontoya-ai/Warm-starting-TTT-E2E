from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ttt.research.orchestrator import OrchestratorOptions, build_train_command, scaled_schedule_overrides
from ttt.research.registry import load_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "96_run_revision_v3_ladder.py"
HAS_HYDRA = importlib.util.find_spec("hydra") is not None


def _launcher():
    spec = importlib.util.spec_from_file_location("revision_v3_ladder", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _opts(**kwargs) -> OrchestratorOptions:
    base = dict(
        deploy="revision_v3_vast_h100_8x",
        runtime_mode="jax_train",
        exp_dir=Path("/tmp/exp"),
        checkpoint_root=Path("/tmp/ckpt"),
        profile_root=Path("/tmp/profiles"),
        dclm_root=Path("/tmp/dclm"),
        books_root=Path("/tmp/books"),
        exp_folder="v3",
        wandb_entity="none",
        wandb_project="none",
        wandb_key="none",
    )
    base.update(kwargs)
    return OrchestratorOptions(**base)


def _overrides(cmd: list[str], prefix: str) -> list[str]:
    return [c for c in cmd if c.startswith(prefix)]


class ScheduleOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stages = load_registry(REGISTRY_PATH).stage_map()

    def test_scaled_schedule_matches_published_recipes(self) -> None:
        # 125M pretraining: warmup 480 of 4800, inner-LR warmup 480.
        self.assertEqual(
            scaled_schedule_overrides(stage=self.stages["S3_PRETRAIN_E2E_125M"], steps=4800),
            ["training.optimizer_outer.lr_warmup_steps=480", "training.optimizer_outer.lr_decay_steps=4800", "training.ilr_warmup_steps=480"],
        )
        # Extension: 10% warmup, cosine to the last step, inner LR untouched.
        self.assertEqual(
            scaled_schedule_overrides(stage=self.stages["S2_125M"], steps=120),
            ["training.optimizer_outer.lr_warmup_steps=12", "training.optimizer_outer.lr_decay_steps=120"],
        )
        # Bridge: its own warmup and decay instead of one 480-step warmup.
        self.assertEqual(
            scaled_schedule_overrides(stage=self.stages["S2_ADAPT_125M"], steps=480),
            ["training.optimizer_outer.lr_warmup_steps=48", "training.optimizer_outer.lr_decay_steps=480", "training.ilr_warmup_steps=48"],
        )
        # FA pretraining has no inner loop.
        self.assertEqual(
            scaled_schedule_overrides(stage=self.stages["S0_PRETRAIN_FA_125M"], steps=4800),
            ["training.optimizer_outer.lr_warmup_steps=480", "training.optimizer_outer.lr_decay_steps=4800"],
        )

    def test_legacy_policy_adds_nothing(self) -> None:
        cmd = build_train_command(stage=self.stages["S2_ADAPT_125M"], opts=_opts(), steps=480, run_id="adapt")
        self.assertEqual(_overrides(cmd, "training.optimizer_outer."), [])
        self.assertEqual(_overrides(cmd, "training.ilr_warmup_steps"), [])
        self.assertEqual(_overrides(cmd, "training.accum_steps"), [])

    def test_ext_accumulation_applies_to_extensions_only(self) -> None:
        opts = _opts(ext_accum_steps=4, ext_global_batch_size=32, schedule_policy="scaled")
        ext_cmd = build_train_command(stage=self.stages["S2_MINUS_125M"], opts=opts, steps=120, run_id="direct")
        self.assertIn("training.accum_steps=4", ext_cmd)
        self.assertIn("training.global_batch_size=32", ext_cmd)
        self.assertIn("training.optimizer_outer.lr_decay_steps=120", ext_cmd)
        bridge_cmd = build_train_command(stage=self.stages["S2_ADAPT_125M"], opts=opts, steps=480, run_id="adapt")
        self.assertEqual(_overrides(bridge_cmd, "training.accum_steps"), [])
        self.assertEqual(_overrides(bridge_cmd, "training.global_batch_size"), [])

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_train_command(stage=self.stages["S2_125M"], opts=_opts(schedule_policy="linear"), steps=120, run_id="x")


class PlanRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.launcher = _launcher()
        self.stages = load_registry(REGISTRY_PATH).stage_map()

    def _args(self, *extra: str):
        return self.launcher.parse_args(["--dclm-root", "/tmp/dclm", "--books-root", "/tmp/books", *extra])

    def test_independent_bridges_and_variant_tags(self) -> None:
        args = self._args("--seeds", "1-2", "--variant", "lr1e-3", "--adapt-peak-lr", "1e-3")
        runs = self.launcher.plan_runs(requested=list(self.launcher.DEFAULT_STAGES), seeds=[1, 2], stage_map=self.stages, args=args)
        by_id = {r["run_id"]: r for r in runs}
        self.assertIn("ext-125m-fa-32K-seed001", by_id)
        self.assertIn("ext-125m-e2e-32K-seed002", by_id)
        bridge = by_id["adapt-125m-e2e-8K-from-fa-lr1e-3-seed002"]
        self.assertIn("training.optimizer_outer.lr=0.001", bridge["extra"])
        self.assertIn("training.model_seed=2", bridge["extra"])
        self.assertIn("training.data_seed=2", bridge["extra"])
        s2 = by_id["ext-125m-e2e-32K-from-fa-bridge-lr1e-3-seed002"]
        self.assertEqual(s2["resume_exp_name"], "adapt-125m-e2e-8K-from-fa-lr1e-3-seed002")
        self.assertIn("training.resume_exp_name=adapt-125m-e2e-8K-from-fa-lr1e-3-seed002", s2["extra"])
        self.assertNotIn("training.optimizer_outer.lr=0.001", s2["extra"])
        # An LR-only variant does not change the direct conversion, so it is shared.
        self.assertIn("ext-125m-e2e-32K-from-fa-direct-seed001", by_id)
        # Each seed's bridge comes before that seed's bridged extension.
        order = [r["run_id"] for r in runs]
        self.assertLess(order.index("adapt-125m-e2e-8K-from-fa-lr1e-3-seed001"), order.index("ext-125m-e2e-32K-from-fa-bridge-lr1e-3-seed001"))
        self.assertEqual(len(runs), 12)

    def test_prime_post_norm_variant_touches_fresh_prime_runs_only(self) -> None:
        args = self._args("--variant", "gain0", "--prime-post-norm-init", "0.0")
        runs = self.launcher.plan_runs(requested=["S2_ADAPT_125M", "S2_125M", "S2_MINUS_125M"], seeds=[1], stage_map=self.stages, args=args)
        extras = {r["stage_id"]: r["extra"] for r in runs}
        ids = {r["stage_id"]: r["run_id"] for r in runs}
        self.assertEqual(ids["S2_MINUS_125M"], "ext-125m-e2e-32K-from-fa-direct-gain0-seed001")
        self.assertEqual(ids["S2_ADAPT_125M"], "adapt-125m-e2e-8K-from-fa-gain0-seed001")
        self.assertIn("model.prime_post_norm_init=0.0", extras["S2_ADAPT_125M"])
        self.assertIn("model.prime_post_norm_init=0.0", extras["S2_MINUS_125M"])
        self.assertNotIn("model.prime_post_norm_init=0.0", extras["S2_125M"])

    def test_shared_bridge_is_explicit_legacy_mode(self) -> None:
        args = self._args("--shared-bridge")
        runs = self.launcher.plan_runs(requested=["S2_ADAPT_125M", "S2_125M"], seeds=[1, 2, 3], stage_map=self.stages, args=args)
        bridges = [r for r in runs if r["stage_id"] == "S2_ADAPT_125M"]
        self.assertEqual([b["run_id"] for b in bridges], ["adapt-125m-e2e-8K-from-fa"])
        self.assertEqual({r["resume_exp_name"] for r in runs if r["stage_id"] == "S2_125M"}, {"adapt-125m-e2e-8K-from-fa"})

    def test_variant_tag_is_validated(self) -> None:
        args = self._args("--variant", "bad tag/")
        with self.assertRaises(ValueError):
            self.launcher.plan_runs(requested=["S2_ADAPT_125M"], seeds=[1], stage_map=self.stages, args=args)

    def test_defaults_follow_the_published_extension_recipe(self) -> None:
        args = self._args()
        self.assertEqual((args.ext_steps, args.ext_global_batch_size, args.ext_accum_steps), (120, 32, 4))
        self.assertEqual(args.schedule_policy, "scaled")
        self.assertFalse(args.shared_bridge)


@unittest.skipUnless(HAS_HYDRA, "hydra not installed")
class LauncherDryRunTest(unittest.TestCase):
    def test_dry_run_plans_every_stage_with_fixed_schedules(self) -> None:
        launcher = _launcher()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "summary.json"
            code = launcher.main([
                "--dclm-root", str(root / "dclm"),
                "--books-root", str(root / "books"),
                "--exp-dir", str(root / "experiments"),
                "--checkpoint-root", str(root / "checkpoints"),
                "--paper-run-id", "v3_dry",
                "--stages", "S0_125M,S1_125M,S3_125M,S2_ADAPT_125M,S2_125M,S2_MINUS_125M,S3W_PRETRAIN_E2E_125M,S3W_125M",
                "--seeds", "1",
                "--dry-run",
                "--eval-after",
                "--summary-out", str(summary),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(summary.read_text())
            statuses = {row["run_id"]: row["status"] for row in payload["rows"]}
            self.assertEqual(len(statuses), 8)
            self.assertTrue(all(status == "dry_run" for status in statuses.values()), statuses)
            self.assertEqual(len(payload["eval"]), 2)
            commands = {}
            for manifest in (root / "experiments" / "v3_dry").rglob("run_manifest.json"):
                data = json.loads(manifest.read_text())
                commands[data["run_id"]] = data["command"]
            bridge = commands["adapt-125m-e2e-8K-from-fa-seed001"]
            self.assertIn("training.optimizer_outer.lr_decay_steps=480", bridge)
            self.assertIn("training.ilr_warmup_steps=48", bridge)
            s2 = commands["ext-125m-e2e-32K-from-fa-bridge-seed001"]
            self.assertIn("training.resume_exp_name=adapt-125m-e2e-8K-from-fa-seed001", s2)
            self.assertIn("training.accum_steps=4", s2)
            self.assertIn("training.total_steps=120", s2)
            wide = commands["pretrain-125m-e2e-wide"]
            self.assertIn("training.total_steps=4800", wide)
            self.assertIn("training.optimizer_outer.lr_decay_steps=4800", wide)


if __name__ == "__main__":
    unittest.main()
