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
class JaxRuntimeSmokeTest(unittest.TestCase):
    def _base_cfg(self, root: Path):
        from ttt.config import Config

        cfg = Config()
        cfg.training.exp_dir = str(root / "runs")
        cfg.training.exp_folder = "paper"
        cfg.training.exp_name = "exp"
        cfg.training.paper_run_id = "paper"
        cfg.training.stage_id = "S0"
        cfg.training.run_id = "exp"
        cfg.training.total_steps = 2
        cfg.training.save_milestone_freq = 1
        cfg.training.global_batch_size = 2
        cfg.training.eval_batch_size = 2
        cfg.training.seq_length = 32
        cfg.training.dataset_path = str(root / "data")
        cfg.training.dataset_name = "books3"
        cfg.training.data_split = "train"
        cfg.training.eval_split = "val"
        cfg.training.dummy_dataset = True
        cfg.training.wandb_entity = "none"
        cfg.training.wandb_project = "none"
        cfg.training.wandb_key = "none"
        cfg.training.loader_workers = 1
        cfg.training.jax_eval_batches = 2
        cfg.training.spec_outer = ["**"]
        cfg.training.spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]
        cfg.model.vocab_size = 256
        cfg.model.hidden_size = 64
        cfg.model.intermediate_size = 128
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.tie_word_embeddings = True
        cfg.model.seq_len = 32
        cfg.training.checkpoint_path = str(root / "checkpoints")
        cfg.checkpoint.checkpoint_dir = str(root / "checkpoints" / "paper" / "exp")
        cfg.checkpoint.resume_checkpoint_dir = cfg.checkpoint.checkpoint_dir
        Path(cfg.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    def _artifacts(self, root: Path):
        from ttt.runtime import RunArtifacts

        run_dir = root / "runs" / "paper" / "S0" / "exp"
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

    def test_train_then_eval_smoke(self) -> None:
        from ttt.jax_runtime.eval import run as eval_run
        from ttt.jax_runtime.train import run as train_run

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._base_cfg(root)
            artifacts = self._artifacts(root)
            logger = logging.getLogger("jax_smoke")
            logger.setLevel(logging.INFO)

            train_run(cfg=cfg, artifacts=artifacts, logger=logger)

            latest_path = Path(cfg.checkpoint.checkpoint_dir) / "latest.json"
            self.assertTrue(latest_path.exists())
            latest = json.loads(latest_path.read_text())
            self.assertEqual(latest["step"], 1)

            cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_eval
            cfg.training.resume_checkpoint_path = cfg.checkpoint.checkpoint_dir
            eval_run(cfg=cfg, artifacts=artifacts, logger=logger)

            self.assertTrue((artifacts.run_dir / "per_position_nll.npy").exists())
            self.assertIn("eval_loss", artifacts.metrics_path.read_text())
            self.assertIn("eval_loss_ce", artifacts.metrics_path.read_text())

    def test_author_checkpoint_path_resolution(self) -> None:
        from ttt.jax_runtime.checkpoint import resolve_resume_checkpoint_dir

        author_root = Path(
            "/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/artifacts/author_checkpoints/760m_fa"
        )
        if not author_root.exists():
            self.skipTest("Author checkpoint artifact not present locally.")

        with tempfile.TemporaryDirectory() as td:
            cfg = self._base_cfg(Path(td))
            cfg.training.resume_checkpoint_path = str(author_root)
            resolved = resolve_resume_checkpoint_dir(
                cfg,
                current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
            )
            self.assertIsNotNone(resolved)
            self.assertTrue(str(resolved).endswith("/28999"))


if __name__ == "__main__":
    unittest.main()
