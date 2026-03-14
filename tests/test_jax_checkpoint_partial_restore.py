from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_ORBAX = importlib.util.find_spec("orbax.checkpoint") is not None


@unittest.skipUnless(HAS_JAX and HAS_ORBAX, "jax/orbax not installed")
class JaxCheckpointPartialRestoreTest(unittest.TestCase):
    def test_params_restore_exact_shapes_prefers_checkpoint_values(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            ck = OrbaxCheckpointer(root)
            ck.save(
                step=3,
                model_weights={
                    "match": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                },
                opt_state=None,
                metrics={"loss": 1.0},
                metadata={},
            )
            restored = ck.load(
                step=3,
                targets={
                    "model_weights": {
                        "match": jnp.zeros((2, 3), dtype=jnp.float32),
                    }
                },
                restore=TrainingConfig.LoadPart.params,
            )

            self.assertEqual(restored.step, 3)
            self.assertTrue((restored.model_weights["match"] == jnp.arange(6, dtype=jnp.float32).reshape(2, 3)).all())
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(skipped, [])

    def test_params_restore_skips_shape_mismatches(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            ck = OrbaxCheckpointer(root)
            ck.save(
                step=7,
                model_weights={
                    "match": jnp.full((2, 3), 7.0, dtype=jnp.float32),
                    "mismatch": jnp.full((4,), 9.0, dtype=jnp.float32),
                },
                opt_state=None,
                metrics={"loss": 1.0},
                metadata={},
            )
            restored = ck.load(
                step=7,
                targets={
                    "model_weights": {
                        "match": jnp.zeros((2, 3), dtype=jnp.float32),
                        "mismatch": jnp.zeros((5,), dtype=jnp.float32),
                    }
                },
                restore=TrainingConfig.LoadPart.params,
            )

            self.assertEqual(restored.step, 7)
            self.assertTrue((restored.model_weights["match"] == 7.0).all())
            self.assertEqual(restored.model_weights["mismatch"].shape, (5,))
            self.assertTrue((restored.model_weights["mismatch"] == 0.0).all())
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(len(skipped), 1)
            self.assertIn("mismatch", skipped[0])

    def test_params_restore_falls_back_when_exact_target_mapping_raises(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import LATEST_FILENAME, OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            root.mkdir(parents=True, exist_ok=True)
            (root / LATEST_FILENAME).write_text(json.dumps({"step": 9}), encoding="utf-8")
            (root / "step_metadata_00000009.json").write_text(json.dumps({"step": 9}), encoding="utf-8")

            ck = OrbaxCheckpointer.__new__(OrbaxCheckpointer)
            ck.checkpoint_dir = root
            ck.manager = mock.Mock()
            ck.manager.latest_step.return_value = 9
            ck.manager.item_metadata.return_value = {
                "model_weights": {
                    "match": jnp.ones((2, 3), dtype=jnp.float32),
                    "mismatch": jnp.ones((4,), dtype=jnp.float32),
                }
            }
            ck.manager.restore.return_value = {
                "model_weights": {
                    "match": jnp.full((2, 3), 5.0, dtype=jnp.float32),
                    "mismatch": jnp.full((4,), 9.0, dtype=jnp.float32),
                }
            }

            with mock.patch(
                "ttt.jax_runtime.checkpoint.fetch_from_eqx_module",
                side_effect=ValueError("shape mismatch"),
            ):
                restored = ck.load(
                    step=9,
                    targets={
                        "model_weights": {
                            "match": jnp.zeros((2, 3), dtype=jnp.float32),
                            "mismatch": jnp.zeros((5,), dtype=jnp.float32),
                        }
                    },
                    restore=TrainingConfig.LoadPart.params,
                )

            self.assertEqual(restored.step, 9)
            self.assertTrue((restored.model_weights["match"] == 5.0).all())
            self.assertEqual(restored.model_weights["mismatch"].shape, (5,))
            self.assertTrue((restored.model_weights["mismatch"] == 0.0).all())
            self.assertEqual(ck.manager.restore.call_count, 1)
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(len(skipped), 1)
            self.assertIn("mismatch", skipped[0])


if __name__ == "__main__":
    unittest.main()
