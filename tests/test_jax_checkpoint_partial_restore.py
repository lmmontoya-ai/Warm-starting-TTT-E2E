from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_ORBAX = importlib.util.find_spec("orbax.checkpoint") is not None


@unittest.skipUnless(HAS_JAX and HAS_ORBAX, "jax/orbax not installed")
class JaxCheckpointPartialRestoreTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
