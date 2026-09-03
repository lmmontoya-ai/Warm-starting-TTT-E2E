from __future__ import annotations

import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path


HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_ORBAX = importlib.util.find_spec("orbax.checkpoint") is not None


@unittest.skipUnless(HAS_JAX and HAS_ORBAX, "jax/orbax not installed")
class JaxRuntimeWarmStartGateTest(unittest.TestCase):
    """End-to-end check that the trainer refuses a warm start that is not warm."""

    def _cfg(self, root: Path, exp_name: str, *, intermediate_size: int = 128, total_steps: int = 2):
        from ttt.config import Config

        cfg = Config()
        cfg.training.exp_dir = str(root / "runs")
        cfg.training.exp_folder = "paper"
        cfg.training.exp_name = exp_name
        cfg.training.paper_run_id = "paper"
        cfg.training.stage_id = "S"
        cfg.training.run_id = exp_name
        cfg.training.total_steps = total_steps
        cfg.training.save_milestone_freq = 1
        cfg.training.global_batch_size = 2
        cfg.training.eval_batch_size = 2
        cfg.training.seq_length = 32
        cfg.training.dataset_path = str(root / "data")
        cfg.training.dataset_name = "books3"
        cfg.training.dummy_dataset = True
        cfg.training.wandb_entity = "none"
        cfg.training.wandb_project = "none"
        cfg.training.wandb_key = "none"
        cfg.training.log_wandb = False
        cfg.training.loader_workers = 1
        cfg.training.jax_eval_batches = 2
        cfg.training.spec_outer = ["**"]
        cfg.training.spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]
        cfg.model.vocab_size = 256
        cfg.model.hidden_size = 64
        cfg.model.intermediate_size = intermediate_size
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.tie_word_embeddings = True
        cfg.model.seq_len = 32
        cfg.model.mini_batch_size = 8
        cfg.model.sliding_window_size = 8
        cfg.training.checkpoint_path = str(root / "checkpoints")
        cfg.checkpoint.checkpoint_dir = str(root / "checkpoints" / "paper" / exp_name)
        cfg.checkpoint.resume_checkpoint_dir = cfg.checkpoint.checkpoint_dir
        Path(cfg.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    def _artifacts(self, root: Path, exp_name: str):
        from ttt.runtime import RunArtifacts

        run_dir = root / "runs" / "paper" / "S" / exp_name
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = RunArtifacts(
            run_dir=run_dir,
            resolved_config_path=run_dir / "resolved_config.yaml",
            unresolved_config_path=run_dir / "unresolved_config.yaml",
            metrics_path=run_dir / "metrics.jsonl",
            events_path=run_dir / "events.jsonl",
            run_manifest_path=run_dir / "run_manifest.json",
            environment_manifest_path=run_dir / "environment_manifest.json",
        )
        for path in (
            artifacts.resolved_config_path,
            artifacts.unresolved_config_path,
            artifacts.run_manifest_path,
            artifacts.environment_manifest_path,
        ):
            path.write_text("{}\n")
        return artifacts

    def _train_parent(self, root: Path, logger):
        from ttt.jax_runtime.train import run as train_run

        cfg = self._cfg(root, "parent")
        train_run(cfg=cfg, artifacts=self._artifacts(root, "parent"), logger=logger)
        self.assertTrue((Path(cfg.checkpoint.checkpoint_dir) / "latest.json").exists())

    def _child_cfg(self, root: Path, name: str, **model_overrides):
        cfg = self._cfg(root, name, **model_overrides)
        cfg.training.load_part = cfg.training.LoadPart.params
        cfg.training.resume_exp_name = "parent"
        return cfg

    def test_shape_mismatch_warm_start_is_refused_then_allowed_explicitly(self) -> None:
        from ttt.jax_runtime.train import run as train_run
        from ttt.jax_runtime.warmstart_guard import WarmStartValidationError

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logger = logging.getLogger("warmstart_gate")
            self._train_parent(root, logger)

            cfg = self._child_cfg(root, "child_mismatch", intermediate_size=96)
            artifacts = self._artifacts(root, "child_mismatch")
            with self.assertRaises(WarmStartValidationError) as ctx:
                train_run(cfg=cfg, artifacts=artifacts, logger=logger)
            self.assertIn("fresh initialization", str(ctx.exception))
            report = json.loads((artifacts.run_dir / "restore_report.json").read_text())
            self.assertGreater(report["mismatched_params"], 0)
            self.assertEqual(report["mode"], "fallback_partial")
            self.assertIn("ffn", report["by_component"])

            cfg = self._child_cfg(root, "child_mismatch_ok", intermediate_size=96)
            cfg.training.warmstart_allow_shape_mismatch = True
            cfg.training.warmstart_max_initial_loss = 0.0
            train_run(cfg=cfg, artifacts=self._artifacts(root, "child_mismatch_ok"), logger=logger)

    def test_same_architecture_warm_start_passes_and_reports_full_coverage(self) -> None:
        from ttt.jax_runtime.train import run as train_run

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logger = logging.getLogger("warmstart_gate")
            self._train_parent(root, logger)
            cfg = self._child_cfg(root, "child_same")
            # A tiny random-ish model on dummy data sits near ln(256) ~ 5.5, so a
            # generous ceiling keeps the loss gate active but non-failing here.
            cfg.training.warmstart_max_initial_loss = 20.0
            artifacts = self._artifacts(root, "child_same")
            train_run(cfg=cfg, artifacts=artifacts, logger=logger)
            report = json.loads((artifacts.run_dir / "restore_report.json").read_text())
            self.assertEqual(report["fresh_params"], 0)
            self.assertEqual(report["restored_fraction"], 1.0)
            events = [json.loads(line) for line in artifacts.events_path.read_text().splitlines() if line.strip()]
            started = [e for e in events if e.get("event") == "run_started"]
            self.assertTrue(started and "coverage" in started[0]["restore"])

    def test_initial_loss_gate_refuses_random_looking_start(self) -> None:
        from ttt.jax_runtime.train import run as train_run
        from ttt.jax_runtime.warmstart_guard import WarmStartValidationError

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logger = logging.getLogger("warmstart_gate")
            self._train_parent(root, logger)
            cfg = self._child_cfg(root, "child_loss")
            cfg.training.warmstart_max_initial_loss = 0.01
            with self.assertRaises(WarmStartValidationError) as ctx:
                train_run(cfg=cfg, artifacts=self._artifacts(root, "child_loss"), logger=logger)
            self.assertIn("first training loss", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
