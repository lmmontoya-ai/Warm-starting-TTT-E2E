from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class JaxRuntimeSmokeTest(unittest.TestCase):
    def test_train_then_eval_smoke(self) -> None:
        from ttt.config import Config
        from ttt.jax_runtime.eval import run as eval_run
        from ttt.jax_runtime.train import run as train_run
        from ttt.runtime import RunArtifacts

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "runs" / "exp"
            run_dir.mkdir(parents=True, exist_ok=True)

            ckpt_dir = root / "checkpoints" / "paper" / "exp"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            cfg = Config()
            cfg.training.exp_dir = str(root / "runs")
            cfg.training.exp_folder = "paper"
            cfg.training.exp_name = "exp"
            cfg.training.paper_run_id = "paper"
            cfg.training.stage_id = "S0"
            cfg.training.run_id = "exp"
            cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_train
            cfg.training.train_mode = cfg.training.TrainMode.pretrain
            cfg.training.total_steps = 2
            cfg.training.save_milestone_freq = 1
            cfg.training.global_batch_size = 2
            cfg.training.eval_batch_size = 2
            cfg.training.seq_length = 32
            cfg.training.jax_max_seq_tokens = 16
            cfg.training.jax_vocab_size_cap = 256
            cfg.training.jax_hidden_size_cap = 64
            cfg.training.jax_num_layers_cap = 2
            cfg.training.jax_intermediate_size_cap = 128
            cfg.training.jax_eval_batches = 2
            cfg.training.dataset_path = str(root / "data")
            cfg.training.dataset_name = "books3"
            cfg.training.data_split = "train"
            cfg.training.eval_split = "val"
            cfg.training.dummy_dataset = True
            cfg.training.wandb_entity = "none"
            cfg.training.wandb_project = "none"
            cfg.training.wandb_key = "none"

            cfg.training.checkpoint_path = str(root / "checkpoints")
            cfg.checkpoint.checkpoint_dir = str(ckpt_dir)
            cfg.checkpoint.resume_checkpoint_dir = str(root / "checkpoints" / "paper" / "")

            artifacts = RunArtifacts(
                run_dir=run_dir,
                resolved_config_path=run_dir / "resolved_config.yaml",
                unresolved_config_path=run_dir / "unresolved_config.yaml",
                metrics_path=run_dir / "metrics.jsonl",
                events_path=run_dir / "events.jsonl",
                run_manifest_path=run_dir / "run_manifest.json",
                environment_manifest_path=run_dir / "environment_manifest.json",
            )
            for p in [
                artifacts.resolved_config_path,
                artifacts.unresolved_config_path,
                artifacts.run_manifest_path,
                artifacts.environment_manifest_path,
            ]:
                p.write_text("{}\n")

            import logging

            logger = logging.getLogger("jax_smoke")
            logger.setLevel(logging.INFO)

            train_run(cfg=cfg, artifacts=artifacts, logger=logger)
            self.assertTrue((ckpt_dir / "latest.json").exists())

            cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_eval
            eval_run(cfg=cfg, artifacts=artifacts, logger=logger)

            self.assertTrue(artifacts.metrics_path.exists())
            content = artifacts.metrics_path.read_text()
            self.assertIn("loss", content)


if __name__ == "__main__":
    unittest.main()
